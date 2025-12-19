import os
import re
import shutil
import subprocess

import numpy as np
import pandas as pd

from Bio.PDB import MMCIFParser, PDBIO

from BPfold.util.RNA_kit import connects2dbn, connects2mat, dbn2connects, cal_metric


def dbn2connects(dbn, strict=True):
    alphabet = ''.join([chr(ord('A')+i) for i in range(26)])
    alphabet_low = alphabet.lower()
    syms = ('([{<' + alphabet)[::-1]
    syms_conj = (')]}>' + alphabet_low)[::-1]
    left2right = {p: q for p, q in zip(syms, syms_conj)}
    right2left = {p: q for p, q in zip(syms_conj, syms)}

    pair_dic = {}
    stack_dic = {char: [] for char in left2right}
    for i, char in enumerate(dbn):
        idx = i+1
        if char=='.':
            pair_dic[idx] = 0
        elif char in left2right:
            stack_dic[char].append((idx, char))
        elif char in right2left:
            cur_stack = stack_dic[right2left[char]]
            if len(cur_stack)==0:
                if strict:
                    raise Exception(f'[Error] Invalid brackets: {dbn}')
                else:
                    pair_dic[idx] = 0
                    continue
            p, ch = cur_stack.pop()
            pair_dic[p] = idx
            pair_dic[idx] = p
        else:
            raise Exception(f'[Error] Unknown DBN representation: dbn[{i}]={char}: {dbn}')
    if any(stack for k, stack in stack_dic.items()) and strict:
        raise Exception(f'[Error] Brackets dismatch: {dbn}')
    connects = [pair_dic[i] for i in range(1, 1+ len(dbn))]
    return connects

def get_seq_from_PDB_by_rnatools(pdb_path):
    '''
        May be wrong.
    '''
    CMD = f'rna_pdb_tools.py --get-seq {pdb_path}'
    res = subprocess.Popen(CMD, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    cont = res.stdout.read()
    res.stdout.close()
    # print(f'get seq from {pdb_path}: "{cont}"')
    lines = cont.split('\n')
    name = lines[0][2:]
    chain = lines[1][1:]
    seq = lines[2]
    # print(name, chain, seq)
    return seq


def get_seq_and_SS_from_PDB_by_onepiece(pdb_path):
    '''
        get multiple chains
    '''
    tmp_dir = 'tmp'
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_file = os.path.join(tmp_dir, 'tmp_op.ss')
    op_src = '/public2/home/heqinzhu/gitrepo/RNA/RNA3d/onePiece/src'
    os.system(f'java -cp {op_src} Zhu_onepiece {pdb_path} {tmp_file} > log_op 2>&1')
    with open(tmp_file) as fp:
        head, seq, ss = fp.read().split('\n')
        return seq, ss


def get_SS_from_PDB_by_briqx(pdb_path):
    CMD = f'pdbInfo -i {pdb_path} -ss'
    res = subprocess.Popen(CMD, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    text = res.stdout.read()
    res.stdout.close()
    dbn = text.strip(' \n\r\t')
    if set(dbn.upper()).issubset(set('AUGCTN')):
        return dbn
    else:
        return None


def get_SS_from_PDB_by_RNAVIEW(pdb_path):
    '''
        will generate files in data_dir, awful
    '''
    pattern = r'pair-type\s+\d+\s+(?P<left>\d+)\s+(?P<right>\d+) [\-\+][\-\+]c'

    length_pattern = r'The total base pairs =\s+\d+ \(from\s+(?P<length>\d+) bases\)'
    tmp_dir = '.tmp'
    os.makedirs(tmp_dir, exist_ok=True)
    dest_path = os.path.join(tmp_dir, os.path.basename(pdb_path))
    if os.path.exists(dest_path):
        os.remove(dest_path)
    shutil.copy(pdb_path, dest_path)
    CMD = f'rnaview {dest_path}'
    res = subprocess.Popen(CMD, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    text = res.stdout.read()
    res.stdout.close()

    length = None
    for L in re.findall(length_pattern, text):
        length = int(L)
    connects = [0]*length
    for left, right in re.findall(pattern, text):
        left, right = int(left), int(right)
        connects[left-1] = right-1
        connects[right-1] = left-1
    return connects


def TMscore_cal(model, native, bin_dir='.'):
    '''
    wget https://zhanggroup.org/TM-score/TMscore.cpp
    g++ -static -O3 -ffast-math -lm -o TMscore TMscore.cpp 

    wget -https://zhanggroup.org/TM-score/TMscore_cpp.gz 
    ungzip TMscore_cpp.gz
    chmod +x TMscore_cpp
    '''
    ## Run TM-score to compare 'model' and 'native':
    CMD = f'{bin_dir}/TMscore_cpp -seq {model} {native}'
    ## Run TM-score to compare two complex structures with multiple chains
    ## Compare all chains with the same chain identifier
    CMD_C = f'{bin_dir}/TMscore_cpp -seq -c {model} {native}'

    res = subprocess.Popen(CMD, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    text = res.stdout.read()
    res.stdout.close()
    pattern = r'RMSD of  the common residues=\s+(?P<RMSD>\d+\.\d+).*?TM-score    = (?P<TMscore>[01]\.\d+).*?GDT-TS-score=\s+(?P<GDT_TS>[01]\.\d+).*?(?P<align_pred_seq>[\-aucg]+)\n[\s:]+\n(?P<align_gt_seq>[\-augc]+)\n'
    for m in re.findall(pattern, text, re.DOTALL):
        RMSD, tmscore, GDT_TS, align_pred_seq, align_gt_seq = m
        return {
                'RMSD': float(RMSD), 
                'TMscore': float(tmscore), 
                'GDT-TS': float(GDT_TS), 
                'align_pred_seq': align_pred_seq.upper(), 
                'align_gt_seq': align_gt_seq.upper(),
               }
    print(f'[Warning]: No RMSD metric found: model={model}, native={native}')
    return dict()


def rnatool_RMSD(target, pred):
    # --target_selection A:1-48+52-63
    # --model_selection A:1-48+52-63
    # --target_ignore_selection A/57/O2\'
    CMD = 'time rna_calc_rmsd.py -t {target} {pred}'
    CMD = 'rna_calc_rmsd.py -t {target} {pred}'

    cur_cmd = CMD.format(
                         target=target,
                         pred=pred,
                        )
    try:
        res = subprocess.Popen(cur_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        print(f'$ {cur_cmd}')
        cont = res.stdout.read()
        print(cont)
        metric = None
        for line in cont.split('\n'):
            if line.startswith(f):
                metric = float(line.strip(' \n\r\t').split(',')[1])
                break
        res.stdout.close()
    except Exception as e:
        print(e)
    finally:
        if metric is None or (metric - -1)<=0:
            return np.nan
        else:
            return metric
        # os.system(cur_cmd)


def briqx_RMSD(pdb1, pdb2):
    cmd = f'rna_rms -allatom {pdb1} {pdb2}'
    res = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    # print(f'$ {cmd}')
    cont = res.stdout.read()
    # print(cont)
    metric = np.nan
    for line in cont.split('\n'):
        if 'rms_' in line:
            metric = float(line.strip(' \n\r\t').split(':')[1])
            break
    res.stdout.close()
    return metric


def rnatool_note():
    pass
    # rna_pdb_tools.py --get-chain A seq.pdb
    # rna_pdb_tools.py --get-seq seq.pdb
    # rna_pdb_tools.py --get-ss seq.pdb


    # Tertiary structure comparison
    # rna_calc_rmsd.py - calculate RMSDs of structures to the target
    # rna_calc_evo_rmsd.py - calculate RMSD between structures based on a given alignment and selected residues as defined in the "x line",
    # rna_calc_inf.py - including multiprocessing based on ClaRNA (in Python 2!)
    # rna_clanstix.py - a tool for visualizing RNA 3D structures based on pairwise structural similarity with Clans,
    # rna_prediction_significance.py - calculate significance of an RNA tertiary structure prediction.
    # Tertiary structure formats
    # diffpdb - a simple tool to compare text-content of PDB files,

    # rna_pdb_merge_into_one.py - merge single files into an NMR-style multiple model file PDB file.

    # Tertiary structure analysis
    # clarna_app.py - a wrapper to ClaRNA, See also PyMOL4RNA, Python 2!
    # rna_x3dna.py - a wrapper to 3dna, See also PyMOL4RNA,
    # ClashCalc.py - a simple clash score calculator, used in NPDock, requires BioPython,
    # Tertiary structure processing
    # rna_refinement.py - a wrapper for QRNAS (Quick Refinement of Nucleic Acids)


def cal_all_metrics(pred_pdb, gt_pdb, eval_SS=True, bin_dir='.'):
    def pad_dbn_by_aligned_seq(seq, ss,):
        pad_ss = []
        ct = 0
        for base in seq.upper():
            if base in 'AUGCNT' and ct<len(ss):
                pad_ss.append(ss[ct])
                ct+=1
            else:
                pad_ss.append('.')
        return pad_ss

    def pad_connects_by_aligned_seq(seq, ss):
        seq = seq.upper()
        pad_ss = []

        ## map original seq to aligned seq, 0-indexed
        ori_base_num = 0
        idx_map = {}
        for i, base in enumerate(seq):
            if base in 'AUGCNT':
                idx_map[ori_base_num] = i
                ori_base_num += 1

        ct = 0
        for base in seq:
            if base in 'AUGCNT' and ct<len(ss):
                # print(seq, base, ct)
                # print(idx_map)
                # print(ss)
                conn = 0 if ss[ct]==0 or (ss[ct]-1 not in idx_map) else idx_map[ss[ct]-1]+1
                pad_ss.append(conn)
                ct+=1
            else:
                pad_ss.append(0)
        return pad_ss

    if not os.path.exists(pred_pdb):
        print('Missing pred:', pred_pdb)
        m_dic = {
                'RMSD': 30, 
                'TMscore': 0, 
                'GDT-TS': 0, 
                'align_pred_seq': 'N', 
                'align_gt_seq': 'N',
                'epsilon_RMSD': 30,
               }
        if eval_SS:
            m_dic['F1'] = 0
            m_dic['Precision'] = 0
            m_dic['Recall'] = 0
            m_dic['MCC'] = 0
            m_dic['gt_dbn'] = '.'
            m_dic['pred_dbn'] = '.'
        return m_dic

    m_dic = TMscore_cal(pred_pdb, gt_pdb, bin_dir=bin_dir)
    m_dic['epsilon_RMSD'] = epsilon_RMSD(pred_pdb, gt_pdb)
    # print(m_dic)
    print(m_dic['align_gt_seq'], 'align gt seq')
    print(m_dic['align_pred_seq'], 'align pred seq')
    if eval_SS:
        # try:
        #     pred_dbn = get_SS_from_PDB_by_briqx(pred_pdb)
        #     gt_dbn = get_SS_from_PDB_by_briqx(gt_pdb)
        #     pred_ss = dbn2connects(pred_dbn)
        #     gt_ss = dbn2connects(gt_dbn)
        # except Exception as e:
        #     print(e, pred_pdb, gt_pdb)
        #     pred_ss = get_SS_from_PDB_by_RNAVIEW(pred_pdb)
        #     gt_ss = get_SS_from_PDB_by_RNAVIEW(gt_pdb)
        #     pred_dbn = connects2dbn(pred_ss)
        #     gt_dbn = connects2dbn(gt_ss)

        pred_ss = get_SS_from_PDB_by_RNAVIEW(pred_pdb)
        gt_ss = get_SS_from_PDB_by_RNAVIEW(gt_pdb)
        pred_dbn = connects2dbn(pred_ss)
        gt_dbn = connects2dbn(gt_ss)

        ## broken data
        if os.path.basename(gt_pdb) == 'PZ23.pdb':
            gt_ss.insert(13, 0)
            gt_dbn = gt_dbn[:12]+'.' + gt_dbn[12:]

        print(gt_dbn, 'gt dbn')
        print(pred_dbn, 'pred dbn')

        align_gt_ss = pad_connects_by_aligned_seq(m_dic['align_gt_seq'], gt_ss)
        align_pred_ss = pad_connects_by_aligned_seq(m_dic['align_pred_seq'], pred_ss)
        SS_metric_dic = cal_metric(connects2mat(align_pred_ss), connects2mat(align_gt_ss)) # F1, P, R, INF, MCC
        print('SS metric:', SS_metric_dic)
        m_dic['F1'] = SS_metric_dic['F1']
        m_dic['Precision'] = SS_metric_dic['Precision']
        m_dic['Recall'] = SS_metric_dic['Recall']
        m_dic['MCC'] = SS_metric_dic['MCC']
        m_dic['gt_dbn'] = gt_dbn
        m_dic['pred_dbn'] = pred_dbn
        print(m_dic)
    return m_dic


def cal_all_pdbs(dest, pred_pre, gt_pre, models, eval_SS=True, bin_dir='.', dataset_names=None):
    tmp_dir = '.tmp_cif2pdb'
    os.makedirs(tmp_dir, exist_ok=True)
    all_data = []
    for model in models:
        pred_all_dataset = os.path.join(pred_pre, model)
        if not os.path.isdir(pred_all_dataset):
            continue
        if dataset_names is None:
            dataset_names = {}
            for dataset in os.listdir(pred_all_dataset):
                dataset_names[dataset] = []
                for f in os.listdir(os.path.join(pred_all_dataset, dataset)):
                    if f.endswith('.pdb') or f.endswith('.cif'):
                        if '_' in f:
                            name = f.split('_')[0]
                        else:
                            name = f[:f.rfind('.')]
                        dataset_names[dataset].append(name)
        for dataset, names in dataset_names.items():
            pred_dir = os.path.join(pred_all_dataset, dataset)
            gt_dir = os.path.join(gt_pre, dataset)
            for name in names:
                gt_pdb = os.path.join(gt_dir, name+'.pdb')
                pred_pdb = os.path.join(pred_dir, name+'.pdb')
                pred_cif = os.path.join(pred_dir, name+'.cif')

                if os.path.exists(pred_cif):
                    new_pdb = os.path.join(tmp_dir, name+'.pdb')
                    cif2pdb(pred_cif, new_pdb)
                    pred_pdb = new_pdb
                print()
                print('gt_pdb:', gt_pdb)
                print('pred_pdb:', pred_pdb)
                cur_data = cal_all_metrics(pred_pdb, gt_pdb, eval_SS, bin_dir=bin_dir)
                all_data.append(dict(model=model, dataset=dataset, name=name, **cur_data))
    df = pd.DataFrame(all_data)
    df.to_csv(dest, index=False)



def cif2pdb(src, dest):
    structure_id = "example"
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure(structure_id, src)
    io = PDBIO()
    io.set_structure(structure)
    io.save(dest)


def prepare_af3_pred(dest, src):
    datasets = ['CASP15_RNAs', '20_RNA_Puzzles']
    select_func = lambda name: datasets[1] if name.lower().startswith('pz') else datasets[0]
    if os.path.exists(dest):
        shutil.rmtree(dest)
    for dataset in datasets:
        os.makedirs(os.path.join(dest, dataset), exist_ok=True)
    for target_name in os.listdir(src):
        src_dir = os.path.join(src, target_name)
        file_name = [name for name in os.listdir(src_dir) if name.endswith('model_0.cif')][0]
        src_path = os.path.join(src_dir, file_name)

        dataset_name = select_func(target_name)
        dest_file_name = f'{target_name.upper()}.pdb'.replace('FREE', 'Free').replace('BOUND', 'Bound')
        dest_path = os.path.join(dest, dataset_name, dest_file_name)
        cif2pdb(src_path, dest_path)


### Additional evaluation    

## #1
def stereochemical(pdb_path):
    ''' stereochemical quality
    # https://sw-tools.rcsb.org/apps/MAXIT/source.html
    # tar -xvf maxit-v10-linux64.tar.gz
    # cd maxit-v10-linux64
    # chmod +x maxit
    # export PATH=$PWD:$PATH
    '''
    cache_dir = '.RNAeval_cache'
    os.makedirs(cache_dir, exist_ok=True)
    out_path = os.path.join(cache_dir, os.path.basename(pdb_path)+'.maxit.txt')
    err_path = os.path.join(cache_dir, os.path.basename(pdb_path)+'.maxit.err')
    os.system(f'maxit -input {pdb_path} > {out_path} 2> {err_Path}')
    os.system(f"grep -Ei 'close|bond length|bond angle|planar|chirality|polymer' {out_path}")
    raise NotImplementedError
    return 


def knot_artifact(pdb_path, ntrials=200):
    '''
    # 3. Knotted artifacts in predicted 3D RNA structures
    - https://github.com/ilbsm/CASP15_knotted_artifacts
    - pip install topoly
    '''
    import topoly
    from topoly import alexander
    coords = topoly.read_xyz(pdb_path, atoms=["P","O5'","C5'","C4'","C3'","O3'"])
    result = alexander(coords, ntrials=ntrials) # random 闭合, compute Alexander polynomial
    return result ## <50%: unknot, if knot, output knot type, such as "3_1" for trefoil knot



def entanglement():
    # RNA 3D structure entanglement
    # https://www.cs.put.poznan.pl/mantczak/spider.zip
    raise NotImplementedError


## #2
def epsilon_RMSD(pdb_path, ref_path):
    '''
        epsilon RMSD
        pip install mdtraj
        pip install barnaba
    '''
    import barnaba as bb 
    try:
        return bb.ermsd(ref_path, pdb_path)
    except Exception as e:
        print(e, pdb_path, ref_path)
        return np.nan


if __name__ == '__main__':
    synthetic_names = ['R1126', 'R1128', 'R1136', 'R1138']
    metric_names = ['RMSD', 'F1']
    pred_pre = 'pred_results'
    gt_pre = '/public/share/heqinzhu_share/structRFM/rna3d_vis/native'

    # prepare_af3_pred(os.path.join(pred_pre, 'af3'), 'af3_results')

    out_name = 'RNA3d_metrics.csv'
    all_metric_path =  f'all_{out_name}'

    models = [
              'af3',
              'trRosettaRNA',
              'structRFM',
             ]
    cal_all_pdbs(all_metric_path, pred_pre, gt_pre, models)
