import os , sys
import json
import pprint 
import pickle
import argparse

import tqdm
import numpy as np 
import pandas as pd
import torch 


from structRFM.infer import structRFM_infer
from task1_ss.evaluate_heatmap import mat2connects, connects2dbn, cal_metric_pairwise, attnmap_to_cont, post_process_heatmap, connects2mat, dbn2connects
from BPfold.util.RNA_kit import dispart_nc_pairs


def read_seq_ss_data(path):
    if path.endswith('.pkl') or path.endswith('.pickle'):
        for name, seq, mat, idx in pickle.load(open(path, 'rb')):
            yield name, seq, mat2connects(mat), idx
    elif path.endswith('.txt'):
        # name, seq, dbn
        with open(path) as fp:
            lines = fp.readlines()
            for i in range(0,len(lines), 3):
                yield lines[i][1:].strip('\n'), lines[i+1].strip('\n'), dbn2connects(lines[i+2].strip('\n')), i//3
    else:
        raise NotImplementedError


def apc_postprocess(pred_matrix, thresh):
    pred_matrix = attnmap_to_cont(pred_matrix)
    # apply threshold
    pred_matrix[pred_matrix<thresh] = 0
    pred_matrix[pred_matrix>=thresh] = 1
    return pred_matrix


def cal_metric(pred_matrix, gt_connects, seq, thresh, postprocess_wo_thresh=False):
    if postprocess_wo_thresh:
        pred_matrix = post_process_heatmap(seq, pred_matrix) # my postprocess, use their
    else:
        pred_matrix = apc_postprocess(pred_matrix, thresh)

    # cal metric  ## TODO, use their metric func
    pred_connects = mat2connects(pred_matrix) # convert matrix to connects (list of pairs)
    
    mcc, inf, f1, p, r = cal_metric_pairwise(pred_connects, gt_connects)
   
    assert f1 >= 0.0 and f1 <= 1.0, f"Unexpected F1: {f1}"

    # print dbn
    # print('gt  ', connects2dbn(gt_connects))
    # print('pred', connects2dbn(pred_connects))
    
    return {
            'MCC': mcc, 
            'INF': inf, 
            'F1': f1, 
            'P': p, 
            'R': r
           }
    

def list_depth(lst):
    if not isinstance(lst, list):
        return 0
    elif not lst:
        return 1
    else:
        return 1 + max(list_depth(item) for item in lst)
    
    
# global parameters 
num_heads = 12 
num_layers = 12 
num_thresh = 1000  # 0.001 step, from 0 to 1
thresh_values = np.linspace(0, 1, num_thresh)  


def get_attentions(model, dest, seq, seq_name, max_length=514):
    cur_dest = os.path.join(dest, 'attn_pkl')
    os.makedirs(cur_dest, exist_ok=True)
    cur_path = os.path.join(cur_dest, seq_name+'.pkl')
    #  attention
    if os.path.exists(cur_path):
        attentions = pickle.load(open(cur_path, 'rb'))
    else:
        if len(seq)>max_length-2:
            seq = seq[:max_length-2]
        features, attentions = model.extract_raw_feature(seq, return_all=True, output_attentions=True)
        attentions = tuple([atten[:, :, 1:-1, 1:-1].cpu() for atten in attentions])
        with open(cur_path, "wb") as f:
            pickle.dump(attentions, f)
    return attentions


def get_layer_outputs(model, dest, seq, seq_name, max_length=514):
    cur_dest = os.path.join(dest, 'layer_outputs')
    os.makedirs(cur_dest, exist_ok=True)
    cur_path = os.path.join(cur_dest, seq_name+'.pkl')
    #  attention
    if os.path.exists(cur_path):
        layer_outputs = pickle.load(open(cur_path, 'rb'))
    else:
        if len(seq)>max_length-2:
            seq = seq[:max_length-2]
        features, attentions = model.extract_raw_feature(seq, return_all=True, output_attentions=True)
        layer_outputs = tuple([feat[1:-1, :].cpu() for feat in features])
        with open(cur_path, "wb") as f:
            pickle.dump(layer_outputs, f)
    return layer_outputs


def val_layer_head(model, dest, result_dir_name, val_path, layer, head, postprocess_wo_thresh=False, gt_pair_mode='all', max_length=514):
    val_data = read_seq_ss_data(val_path)
    cur_dest = os.path.join(dest, result_dir_name)
    os.makedirs(cur_dest, exist_ok=True)

    thresh_name_F1 = [{} for i in range(num_thresh)]
    for i, (name, seq, gt_connects, _) in enumerate(tqdm.tqdm(val_data)):
        if gt_pair_mode=='non-canonical':
            _, gt_connects = dispart_nc_pairs(seq, gt_connects)
        elif gt_pair_mode=='canonical':
            gt_connects, _ = dispart_nc_pairs(seq, gt_connects)
        attentions = get_attentions(model, dest, seq, name, max_length=max_length)
        matrix = attentions[layer][0, head]
        for thresh_idx, thresh in enumerate(thresh_values):
            f1 = cal_metric(matrix.cpu(), gt_connects, seq, thresh, postprocess_wo_thresh)['F1']
            thresh_name_F1[thresh_idx][name] = f1
            if postprocess_wo_thresh:
                break
   
    thresh_avgF1 = {i: np.mean(list(dic.values())) for i, dic in enumerate(thresh_name_F1)}
    best_idx = max(thresh_avgF1, key=lambda i: thresh_avgF1[i])
    best_th = thresh_values[best_idx]
    best_F1 = thresh_avgF1[best_idx]

    save_name = f'layer{layer:02d}_head{head:02d}_th{best_th:.3f}_F1score{best_F1:.4f}.json'
    with open(os.path.join(cur_dest, save_name), 'w') as fp:
        json.dump(thresh_name_F1, fp)


def val_layer_outputs(model, dest, result_dir_name, val_path, layer, postprocess_wo_thresh=False, gt_pair_mode='all', max_length=514):
    val_data = read_seq_ss_data(val_path)
    cur_dest = os.path.join(dest, result_dir_name)
    os.makedirs(cur_dest, exist_ok=True)

    thresh_name_F1 = [{} for i in range(num_thresh)]
    for i, (name, seq, gt_connects, _) in enumerate(tqdm.tqdm(val_data)):
        if gt_pair_mode=='non-canonical':
            _, gt_connects = dispart_nc_pairs(seq, gt_connects)
        elif gt_pair_mode=='canonical':
            gt_connects, _ = dispart_nc_pairs(seq, gt_connects)
        layer_outputs = get_layer_outputs(model, dest, seq, name, max_length=max_length)
        feat = layer_outputs[layer]
        matrix = feat @ feat.transpose(-1, -2)
        for thresh_idx, thresh in enumerate(thresh_values):
            f1 = cal_metric(matrix.cpu(), gt_connects, seq, thresh, postprocess_wo_thresh)['F1']
            thresh_name_F1[thresh_idx][name] = f1
            if postprocess_wo_thresh:
                break
   
    thresh_avgF1 = {i: np.mean(list(dic.values())) for i, dic in enumerate(thresh_name_F1)}
    best_idx = max(thresh_avgF1, key=lambda i: thresh_avgF1[i])
    best_th = thresh_values[best_idx]
    best_F1 = thresh_avgF1[best_idx]

    save_name = f'layer{layer:02d}_th{best_th:.3f}_F1score{best_F1:.4f}.json'
    with open(os.path.join(cur_dest, save_name), 'w') as fp:
        json.dump(thresh_name_F1, fp)


def cal_test_metric(model, dest, val_dir, test_path, postprocess_wo_thresh=False, max_length=514):
    test_data = read_seq_ss_data(test_path)
    test_name = os.path.basename(test_path)
    test_name = test_name[:test_name.rfind('.')]

    ## collecting val_results
    val_dir = os.path.join(dest, val_dir)
    layer_head_thresh = {}
    for f in os.listdir(val_dir):
        layer_s, head_s, th_s, F1_s = f[:f.rfind('.')].split('_')
        layer = int(layer_s[len('layer'):])
        head = int(head_s[len('head'):])
        th = float(th_s[len('th'):])
        layer_head_thresh[(layer, head)] = th

    df_data = []
    json_data = {}
    for i, (name, seq, gt_connects, _) in enumerate(test_data):
        if len(seq)>max_length-2 or len(seq)!=len(gt_connects): # discard len>max_length-2
            continue
        attentions = get_attentions(model, dest, seq, name)
        for layer in range(num_layers):
            for head in range(num_heads):
                matrix = attentions[layer][0, head]
                thresh = layer_head_thresh[(layer, head)]
                m_dic = cal_metric(matrix.cpu(), gt_connects, seq, thresh, postprocess_wo_thresh)
                f1 = m_dic['F1']
                mcc = m_dic['MCC']

                df_data.append({
                                'layer': layer,
                                'head': head,
                                'name': name,
                               'F1': f1, 
                               'MCC': mcc,
                               'thresh': thresh,
                          })
                if name not in json_data or json_data[name]['F1']<f1:
                    json_data[name] = {
                                       'F1': f1, 
                                       'MCC': mcc,
                                       'thresh': thresh,
                                      }
   
    pd.DataFrame(df_data).to_csv(os.path.join(dest, test_name+'.csv'), index=False)

    avg_F1 = np.mean([d['F1'] for d in json_data.values()])

    save_name = f'{test_name}_F1score{avg_F1:.4f}.json'
    with open(os.path.join(dest, save_name), 'w') as fp:
        json.dump(json_data, fp)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str, default='/public/share/heqinzhu_share/structRFM/structRFM_checkpoint')
    parser.add_argument('--model_name', type=str, default='structRFM')
    parser.add_argument('-g', '--gpu', type=str, default='0')
    parser.add_argument('-L', '--max_length', type=int, default=514)
    parser.add_argument('--num_attention_heads', type=int, default=12)
    parser.add_argument('--layer', type=int, default=12)
    parser.add_argument('--dim', type=int, default=768)
    parser.add_argument('--head', type=int, default=12)
    parser.add_argument('--layer_idx', type=int, default=0) # the i-th layer
    parser.add_argument('--head_idx', type=int, default=0)  # the j-th head
    args = parser.parse_args()
    return args


def run_zeroshot_ssp(from_pretrained, model_name=None, gpu='0', max_length=514, postprocess_wo_thresh=False, layer_idx=0, head_idx=0, **model_paras):

    gt_pair_mode = 'non-canonical' # 'all', 'non-canonical', 'canonical'
    # gt_pair_mode = 'non-canonical' # 'all', 'non-canonical', 'canonical'
    model_name = model_name or "structRFM" # model name, used for saving the results
    run_name = model_name or 'structRFM'
    if postprocess_wo_thresh:
        run_name = run_name + '_wo-thresh'

    
    val_file = 'VL_40_key_seq_contact_idx.pkl'
    val_file = 'dbn_PDB_test.txt'
    run_name = run_name + '_PDB116'


    prefix = '.'
    dest = os.path.join(prefix, f'task1_ss/attns/{run_name}')
    data_dir = os.path.join(prefix, 'task1_ss/data')

    val_path = os.path.join(data_dir, val_file)

    model = structRFM_infer(from_pretrained=from_pretrained, device=f'cuda:{gpu}', max_length=max_length, **model_paras)
    # model, alphabet = fm.pretrained.rna_fm_t12()

    ## cache all attn matrixs
    test_files = [
                  'casp14_key_seq_contact.pkl',
                  'casp15_key_seq_contact.pkl',
                  'TS_70_key_seq_contact_idx.pkl',
            ]
    files = [val_file]
    for f in files:
        data = read_seq_ss_data(os.path.join(data_dir, f))
        for i, (name, seq, gt_connects, _) in enumerate(tqdm.tqdm(data)):
            get_attentions(model, os.path.join(dest), seq, name)
            get_layer_outputs(model, os.path.join(dest), seq, name, max_length=max_length)

    existings = []

    gt_pair_flag = ''
    if gt_pair_mode == 'non-canonical':
        gt_pair_flag = '_nc'
    elif gt_pair_mode == 'canonical':
        gt_pair_flag = '_c'
    if layer_idx is not None and head_idx is not None:
        result_dir_name = 'val_result' + gt_pair_flag
        result_dir = os.path.join(dest, result_dir_name)
        if os.path.exists(result_dir):
            existings = ['_'.join(f.split('_')[:2]) for f in os.listdir(result_dir)]
        if f'layer{layer_idx:02d}_head{head_idx:02d}' not in existings:
            val_layer_head(model, dest, result_dir_name, val_path, layer_idx, head_idx, postprocess_wo_thresh, gt_pair_flag, max_length=max_length)
    elif layer_idx is not None and head_idx is None:
        result_dir_name = 'val_result_layer_outputs' + gt_pair_flag
        result_dir = os.path.join(dest, result_dir_name)
        if os.path.exists(result_dir):
            existings = ['_'.join(f.split('_')[0]) for f in os.listdir(result_dir)]
        if f'layer{layer_idx:02d}' not in existings:
            val_layer_outputs(model, dest, result_dir_name, val_path, layer_idx, postprocess_wo_thresh, gt_pair_flag, max_length=max_length)

    ## after running all val
    val_dir = 'val_result'
    if layer_idx is None and head_idx is None:
        for f in test_files:
            cal_test_metric(model, dest, val_dir, os.path.join(data_dir, f), postprocess_wo_thresh, max_length=max_length)


if __name__ == '__main__':
    args = parse_args()
    from_pretrained = os.getenv('structRFM_checkpoint', args.checkpoint_path)
    run_zeroshot_ssp(from_pretrained, model_name=args.model_name, gpu=args.gpu, max_length=args.max_length, postprocess_wo_thresh=False,  dim=args.dim, layer_idx=args.layer_idx, head_idx=args.head_idx, layer=args.layer, num_attention_heads=args.num_attention_heads)
