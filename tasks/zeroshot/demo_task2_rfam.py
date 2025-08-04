'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2025-06-20 16:03:00
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2025-06-25 18:32:36
FilePath: /coding/RNA_Zero_Shot/structRFM/demo.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
# pip install structRFM-0.0.1.tar.gz
# pip install transformers, datasets

from structRFM.infer import structRFM_infer
import pprint 
from task1_ss.evaluate_heatmap import mat2connects, connects2dbn, cal_metric_pairwise, attnmap_to_cont
import pickle
import torch 
import pandas as pd
import numpy as np 
import os 
from multiprocessing import Pool 
from extract_embs.tools import fftCal, emb_rfam_cos_sim_cal, metricCal 

num_cores = os.cpu_count()
print(f'当前可用 CPU 核心数: {num_cores}')


## single processing 

if __name__ == '__main__':
    from_pretrained = os.getenv('structRFM_checkpoint', '/public/share/heqinzhu_share/structRFM/structRFM_checkpoint')
    model = structRFM_infer(from_pretrained=from_pretrained, max_length=514, device='cuda:7')
    prefix = '.'

    model_name = "structRFM"  # model name, used for saving the results
    ## Step 1: load the data: (0, 'RF00239_AC055869.13/127592-127511', 'GCCUCUCUCUCCGUGUUCACAGCGGACCUUGAUUUAAAUGUCCAUACAAUUAAGGCACGCGGUGAAUGCCAAGAAUGGGGCU')
    ## RF00239：Rfam family ID 
    data = pickle.load(open(os.path.join(prefix, 'task2_rfam/data/random_sample_max10_n24607_rf4170.pkl'), 'rb')) # TODO, update path
    
    ## embedding the sequences, and then using FFN to get the final features, and saving the features to a file
    ## 'labels', 'keys', 'seqs', 'seq_repr', only change the 'seq_repr' 
    feature_path = os.path.join(prefix, f'task2_rfam/embs/{model_name}/embs.pkl') # TODO, update path

    if not os.path.exists(feature_path):
        os.makedirs(os.path.dirname(feature_path), exist_ok=True)
        all_embs = []
        all_labels = []
        all_keys = []
        all_seqs = []
        
        for i, (index, label, seq) in enumerate(data):
            # print(f'Processing {i+1}/{len(data)}: {label}')
            # if len(seq) > 513:
            #     print(f'Skipping {label} due to length {len(seq)} > 514')
            #     continue
            # print(seq)
            if len(seq)>512:
                seq = seq[:512]
            features = model.extract_feature(seq, return_all=False, output_attentions=False)
            # print(features.shape)
            out_features = fftCal(features.cpu().numpy())
            out_features = np.squeeze(out_features, axis=0)
            all_embs.append(out_features)  # Squeeze the second dimension
            # print(f'out_features shape: {out_features.shape}')
            
            all_keys.append(label)
            all_labels.append(index)
            all_seqs.append(seq)
        
        all_embs = np.array(all_embs)
        print(f'All features shape: {all_embs.shape}')
        
        all_features = {'labels': all_labels, 'keys': all_keys, 'seqs': all_seqs, 'seq_repr': all_embs}
        pickle.dump(all_features, open(feature_path, 'wb'))
        print(f'Features saved to {feature_path}')
 
    ## Step 2: 
    save_path = os.path.join(prefix, 'task2_rfam/cos_sim')
    pos_sim, neg_sim, delta_mean, delta_median, overlap_ratio = emb_rfam_cos_sim_cal(
        features_pkl=feature_path,
        save_path=save_path,
        model_name=model_name,
        n_samples=100000,      # 
        plot=True,             # 是否绘制 KDE 分布图
        xlim=[-0.5, 1.0],      # KDE 图x轴范围
        xticks=[0.0, 0.5, 1.0],
        gauss_kde=True         # 是否计算分布重叠度
    )

    print(f'Positive similarity: {pos_sim}, Negative similarity: {neg_sim}')
    print(f'Delta mean: {delta_mean}, Delta median: {delta_median}, Overlap ratio: {overlap_ratio}')
    print(f'Results saved to {save_path}')
    
    
    ## Step3: delta_mean,delta_median,F1,MCC,thresh,overlap_ratio
    all_delta_mean = []
    all_delta_median = []
    all_F1 = []
    all_MCC = []
    all_thresh = []
    all_overlap_ratio = []
    for ri in range(1, 4):
        idx = 0
        df = pd.DataFrame(data=None, columns=['model', 'delta_mean', 'delta_median', 'F1', 'MCC', 'thresh', 'overlap_ratio'])
        model = 'structRFM'  # model name, used for saving the results

        pos_sim, neg_sim, delta_mean, delta_median, overlap_ratio = emb_rfam_cos_sim_cal(
        features_pkl=feature_path,
        save_path=save_path,
        model_name=model,
        n_samples=100000,      # 
        plot=True,             # 是否绘制 KDE 分布图
        xlim=[-0.5, 1.0],      # KDE 图x轴范围
        xticks=[0.0, 0.5, 1.0],
        gauss_kde=True         # 是否计算分布重叠度
        )

        test_data = {
            'pred': np.hstack([pos_sim, neg_sim]),
            'true': np.hstack([np.ones_like(pos_sim), np.zeros_like(neg_sim)])
        }
        metric = metricCal(test_data, step=0.01)   
        
        all_delta_mean.append(round(delta_mean, 3))
        all_delta_median.append(round(delta_median, 3))
        all_F1.append(round(metric['F1'], 3))
        all_MCC.append(round(metric['MCC'], 3))
        all_thresh.append(round(metric['thresh'], 3)) 
        all_overlap_ratio.append(overlap_ratio)

        df.loc[idx] = (model, round(delta_mean, 3), round(delta_median, 3), 
                    round(metric['F1'], 3), round(metric['MCC'], 3), round(metric['thresh'], 2), 
                    overlap_ratio)
        idx += 1
        df.to_csv(f'or{ri}.csv', index=False)
        
    avg_delta_mean = round(np.mean(all_delta_mean), 3)
    avg_delta_median = round(np.mean(all_delta_median), 3)
    avg_F1 = round(np.mean(all_F1), 3)
    avg_MCC = round(np.mean(all_MCC), 3)
    avg_thresh = round(np.mean(all_thresh), 3)
    avg_overlap_ratio = round(np.mean(all_overlap_ratio), 3)
    
    std_delta_mean = round(np.std(all_delta_mean), 3)
    std_delta_median = round(np.std(all_delta_median), 3)
    std_F1 = round(np.std(all_F1), 3)
    std_MCC = round(np.std(all_MCC), 3)
    std_thresh = round(np.std(all_thresh), 3)
    std_overlap_ratio = round(np.std(all_overlap_ratio), 3)
    
    
    print(f'Average results for structRFM across 3 runs:')
    print(f'Average delta_mean: {avg_delta_mean}±{std_delta_mean}')
    print(f'Average delta_median: {avg_delta_median}±{std_delta_median}')
    print(f'Average F1: {avg_F1}±{std_F1}')
    print(f'Average MCC: {avg_MCC}±{std_MCC}')
    print(f'Average thresh: {avg_thresh}±{std_thresh}')
    print(f'Average overlap_ratio: {avg_overlap_ratio}±{std_overlap_ratio}')    
