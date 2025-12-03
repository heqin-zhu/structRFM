import os 
import argparse

from structRFM.infer import structRFM_infer 
from task1_ss.evaluate_heatmap import mat2connects, connects2dbn, cal_metric_pairwise, attnmap_to_cont
import numpy as np 
from multiprocessing import Pool 
from extract_embs.tools import fftCal, emb_archiveII_cos_sim_cal, metricCal
import pandas as pd 
import pickle
import torch 


num_cores = os.cpu_count()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str, default='/public/share/heqinzhu_share/structRFM/structRFM_checkpoint')
    parser.add_argument('--model_name', type=str, default='structRFM')
    parser.add_argument('-g', '--gpu', type=str, default='0')
    parser.add_argument('-L', '--max_length', type=int, default=514)
    parser.add_argument('--num_attention_heads', type=int, default=12)
    parser.add_argument('--dim', type=int, default=768)
    parser.add_argument('--layer', type=int, default=12)
    args = parser.parse_args()
    return args


def run_archiveII(from_pretrained, model_name=None, gpu='0', max_length=514, **model_paras):
    model = structRFM_infer(from_pretrained=from_pretrained, device=f'cuda:{gpu}', max_length=max_length, **model_paras)
    prefix = '.'

    # ## Step 1: load the data: '5s', '5s_Bacillus-cereus-1', 'UUUGGUGAUGAUGGCAGAGAGGUCACACCCGUUCCCAUACCGAACACGGAAGUUAAGCUCUCUAGCGCCGAUGGUAGUUGGGACCUUGUCCCUGUGAGAGUAGGACGUCGCCAAG')
    # ## RF00239：Rfam family ID 
    data = pickle.load(open(os.path.join(prefix, 'task3_archivell/data/archivell_n3864_dataset.pkl'), 'rb')) # TODO, update path
    
    # ## embedding the sequences, and then using FFN to get the final features, and saving the features to a file
    # ## 'labels': 5s, 'keys':'5s_Bacillus-cereus-1' , 'seqs', 'seq_repr', only change the 'seq_repr' 

    
    model_name = model_name or "structRFM" # model name, used for saving the results
    feature_path = os.path.join(prefix, f'task3_archivell/embs/{model_name}/embs.pkl') # TODO, update path
    # emb_data = pickle.load(open(feature_path, 'rb'))

    if not os.path.exists(feature_path):
        os.makedirs(os.path.dirname(feature_path), exist_ok=True)
        all_embs = []
        all_labels = []
        all_keys = []
        all_seqs = []
        
        for i, (label, key, seq) in enumerate(data):
            # print(f'Processing {i+1}/{len(data)}: {label}')
            if len(seq) > max_length-2:
                print(f'Skipping {label} due to length {len(seq)} > {max_length-2}')
                continue
            features = model.extract_raw_feature(seq, return_all=False, output_attentions=False)
            # print(features.shape)
            out_features = fftCal(features.cpu().numpy())
            out_features = np.squeeze(out_features, axis=0)
            all_embs.append(out_features)
            all_keys.append(key)
            all_labels.append(label)
            all_seqs.append(seq)
            
        all_embs = np.array(all_embs)
        print(f'All features shape: {all_embs.shape}')
        all_features = {'labels': all_labels, 'keys': all_keys, 'seqs': all_seqs, 'seq_repr': all_embs}
        pickle.dump(all_features, open(feature_path, 'wb'))
        print(f'Features saved to {feature_path}')

    ## Step 2: 
    save_path = os.path.join(prefix, 'task3_archivell/cos_sim')
    pos_sim, neg_sim, delta_mean, delta_median, overlap_ratio = emb_archiveII_cos_sim_cal(
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
    n_runs = 4
    for ri in range(n_runs):
        idx = 0
        df = pd.DataFrame(data=None, columns=['model', 'delta_mean', 'delta_median', 'F1', 'MCC', 'thresh', 'overlap_ratio'])
        model = 'structRFM'  # model name, used for saving the results

        pos_sim, neg_sim, delta_mean, delta_median, overlap_ratio = emb_archiveII_cos_sim_cal(
            features_pkl=feature_path,
            save_path=save_path,
            model_name=model,
            n_samples=100000,      # 
            plot=True,             # 是否绘制 KDE 分布图
            xlim=[0, 1.2],      # KDE 图x轴范围
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
        df.to_csv(f'task3_archivell/or{ri}.csv', index=False)
        
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

    metric_path = 'metric.csv'
    if not os.path.exists(metric_path):
        with open(metric_path, 'w') as fp:
            fp.write(f'model_name,dataset,avg_delta_mean,std_delta_mean,avg_F1,std_F1,avg_MCC,std_MCC,avg_thresh,std_thresh,avg_overlap_ratio,std_overlap_ratio,avg_delta_median,std_delta_median\n')
    with open(metric_path, 'a') as fp:
        fp.write(f'{model_name},archiveII,{avg_delta_mean},{std_delta_mean},{avg_F1},{std_F1},{avg_MCC},{std_MCC},{avg_thresh},{std_thresh},{avg_overlap_ratio},{std_overlap_ratio},{avg_delta_median},{std_delta_median}\n')
    
    print(f'Average results for structRFM across {n_runs} runs:')
    print(f'Average delta_mean: {avg_delta_mean}±{std_delta_mean}')
    print(f'Average delta_median: {avg_delta_median}±{std_delta_median}')
    print(f'Average F1: {avg_F1}±{std_F1}')
    print(f'Average MCC: {avg_MCC}±{std_MCC}')
    print(f'Average thresh: {avg_thresh}±{std_thresh}')
    print(f'Average overlap_ratio: {avg_overlap_ratio}±{std_overlap_ratio}')    

'''
Average results for structRFM across 4 runs:
Average delta_mean: 0.138±0.0
Average delta_median: 0.156±0.001
Average F1: 0.669±0.0
Average MCC: 0.214±0.001
Average thresh: 0.48±0.0
Average overlap_ratio: 0.48±0.001
'''

if __name__ == '__main__':
    args = parse_args()
    from_pretrained = os.getenv('structRFM_checkpoint', args.checkpoint_path)
    run_archiveII(from_pretrained, model_name=args.model_name, gpu=args.gpu, max_length=args.max_length, dim=args.dim, layer=args.layer, num_attention_heads=args.num_attention_heads)
