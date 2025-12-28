import os
import re

FILE_DIR = os.path.dirname(os.path.abspath(__file__))


def stereochemical(pdb_path):
    ''' stereochemical quality
    # https://sw-tools.rcsb.org/apps/MAXIT/source.html
    # tar -xvf maxit-v10-linux64.tar.gz
    # cd maxit-v10-linux64
    # chmod +x maxit
    # export PATH=$PWD:$PATH
    update maxit/src/Ndb2Pdb_Remark_500s_Util.C
    '''
    maxit_dir = f'{FILE_DIR}/tools/maxit-v10.200-prod-src'
    os.environ['RCSBROOT'] = maxit_dir
    cache_dir = '.maxit_cache'
    os.makedirs(cache_dir, exist_ok=True)

    out_path = os.path.join(cache_dir, os.path.basename(pdb_path)+'.maxit.txt')
    out_pdb = os.path.join(cache_dir, os.path.basename(pdb_path)+'.pdb')
    log_path = os.path.join(cache_dir, os.path.basename(pdb_path)+'.maxit.log')
    CMD = f'{maxit_dir}/bin/process_entry -input {pdb_path} -input_format pdb -output {out_pdb} -output_format pdb -log {log_path} 2> {out_path}'
    os.system(CMD)
    pattern = r'\[STEREOCHEM SUMMARY\] close=(?P<close>\d+) bond length=(?P<bond_length>\d+) bond angle=(?P<bond_angle>\d+) planar=(?P<planar>\d+) chirality=(?P<chirality>\d+) polymer=(?P<polymer>\d+)'
    keys = ['close', 'bond_length', 'bond_angle', 'planar', 'chirality', 'polymer']
    with open(out_path) as fp:
        text = fp.read()
        for tup in re.findall(pattern, text):
            return {k: v for k, v in zip(keys, tup)}


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


if __name__ == '__main__':
    pdb_path = '../structRFM_labs/RNAeval_results/structRFM/CASP16/R1260.pdb'
    pdb_path = '/public/share/heqinzhu_share/structRFM/Zfold_test_data/CASP16/pdb/R1203.pdb'
    ret = stereochemical(pdb_path)
    print(ret)
