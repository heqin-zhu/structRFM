from structRFM.infer import structRFM_infer
import pprint 
from task1_ss.evaluate_heatmap import mat2connects, connects2dbn, cal_metric_pairwise, attnmap_to_cont
import pickle
import torch 
import numpy as np 
import os 
from multiprocessing import Pool 

num_cores = os.cpu_count()
print(f'CPU: {num_cores}')

def cal_metric(pred_matrix, gt_connects, thresh):
    #  thresh = 0.62 # TODO, vary it

    # postprocess: symmetry, ...
    ### pred_matrix = post_process_heatmap(seq, pred_matrix) # my postprocess, use their
    pred_matrix = attnmap_to_cont(pred_matrix)
    
    
    # apply threshold
    pred_matrix[pred_matrix<thresh] = 0
    pred_matrix[pred_matrix>thresh] = 1
    
   
    # cal metric  ## TODO, use their metric func
    pred_connects = mat2connects(pred_matrix) # convert matrix to connects (list of pairs)
    gt_connects = mat2connects(gt_connects)
    
    mcc, inf, f1, p, r = cal_metric_pairwise(pred_connects, gt_connects)
   
    print('f1', f1)
    assert f1 >= 0.0 and f1 <= 1.0, f"Unexpected F1: {f1}"

    # print dbn
    # print('gt  ', connects2dbn(gt_connects))
    # print('pred', connects2dbn(pred_connects))
    
    return f1 
    

def list_depth(lst):
    if not isinstance(lst, list):
        return 0
    elif not lst:
        return 1
    else:
        return 1 + max(list_depth(item) for item in lst)
    
    
## single processing 

# if __name__ == '__main__':
    # from_pretrained = '/public/share/heqinzhu_share/structRFM/structRFM_checkpoint'
#     model = structRFM_infer(from_pretrained=from_pretrained, max_length=514)

#     ## Step 1: load the data: valid data and test data, saving the gt_connects and seqs 
#     # the validation data has the 40 sequences, and the ground truth contacts, for each data, the first element is the name, the second element is the sequence, the third element is the ground truth contacts 
#     valid_data = pickle.load(open('/mnt/zj-gpfs/home/liruifeng/RNA_Zero_Shot/structRFM/task1_ss/data/VL_40_key_seq_contact_idx.pkl', 'rb'))
   
#     ## Step 2: 
#     num_heads = 12 
#     num_layers = 12 
#     num_shresh = 1000 # 0.001 step, from 0 to 1
#     shresh_values = np.linspace(0, 1, num_shresh)  
#     metrics = [[[[] for _ in range(num_shresh)] for _ in range(num_heads)] for _ in range(num_layers)]

    
#     for i, (name, seq, gt_connects, _ ) in enumerate(valid_data):
#         print(f'Processing {i+1}/{len(valid_data)}: {name}, seq len: {len(seq)}')
        
#         # extract feature and attention matrix
#         features, attentions = model.extract_raw_feature(seq, return_all=True, output_attentions=True)
#         attentions = tuple([atten[:, :, 1:-1, 1:-1] for atten in attentions])
        
#         gt_connects = torch.from_numpy(gt_connects)

#         for layer in range(len(attentions)):
#             layer_atten = attentions[layer]
#             batch, num_head, _,  _ = layer_atten.shape
#             for head in range(num_head):
#                 matrix = layer_atten[0, head]
#                 print(f'Layer {layer}, Head {head}, shape: {matrix.shape}')
                
#                 for shresh_idx in range(num_shresh):
#                     shresh = shresh_values[shresh_idx]
#                     f1 = cal_metric(matrix.cpu(), gt_connects.cpu(), shresh) 
#                     metrics[layer][head][shresh_idx].append(f1)  # ← 核心追加
#                     print('shresh', shresh,  'f1', f1)
    
#     ## Step 3: saving the results, including the attention matrix and the metric results 
#     best_score = -float("inf")
#     best_config = None
    

#     avg_metrics = [[[None for _ in range(num_shresh)] for _ in range(num_heads)] for _ in range(num_layers)]

#     for layer in range(num_layers):
#         for head in range(num_heads):
#             for s_idx in range(num_shresh):
#                 all_results = metrics[layer][head][s_idx]
#                 if all_results:
#                     avg_f1 = np.mean(all_results)
#                     avg_metrics[layer][head][s_idx] = avg_f1
#                     if avg_f1 > best_score:
#                         best_score = avg_f1
#                         best_config = (layer, head, shresh_values[s_idx])




#     print(f"Best configuration: Layer {best_config[0]}, Head {best_config[1]}, Threshold {best_config[2]:.3f}，平均 f1: {best_score:.4f}")


#     ## Step 4: saving the metrics and best config
#     # save metrics 和 avg_metrics
#     with open("/data/coding/RNA_Zero_Shot/structRFM/task1_ss/metrics_per_point.pkl", "wb") as f:
#         pickle.dump(metrics, f)


#     with open("/data/coding/RNA_Zero_Shot/structRFM/task1_ss/avg_metrics.pkl", "wb") as f:
#         pickle.dump(avg_metrics, f)
 
 
  
# multi processing 
# global parameters 
num_heads = 12 
num_layers = 12 
num_shresh = 1000  # 0.001 step, from 0 to 1
shresh_values = np.linspace(0, 1, num_shresh)  

def process_one_sequence(args):
    """
    single sequence
    """
    idx, name, seq, gt_connects, from_pretrained = args

    print(f'Processing {idx + 1}: {name}, seq len: {len(seq)}')

    # define the model
    model = structRFM_infer(from_pretrained=from_pretrained, max_length=514)

    #  attention
    features, attentions = model.extract_raw_feature(seq, return_all=True, output_attentions=True)
    attentions = tuple([atten[:, :, 1:-1, 1:-1] for atten in attentions])
    gt_connects = torch.from_numpy(gt_connects)
    
    #  metrics
    metrics_local = [[[[] for _ in range(num_shresh)] for _ in range(num_heads)] for _ in range(num_layers)]

    for layer in range(len(attentions)):
        layer_atten = attentions[layer]
        for head in range(num_heads):
            matrix = layer_atten[0, head]
            for shresh_idx, shresh in enumerate(shresh_values):
                f1 = cal_metric(matrix.cpu(), gt_connects.cpu(), shresh)
                metrics_local[layer][head][shresh_idx].append(f1)

    return metrics_local


if __name__ == '__main__':
    from_pretrained = os.getenv('structRFM_checkpoint', '/public/share/heqinzhu_share/structRFM/structRFM_checkpoint')
    prefix = '.'

    # CPU number 
    num_cores = os.cpu_count()
    print(f"Detected {num_cores} CPU cores, will use multiprocessing with {num_cores} processes.")

    # lond the validation data
    valid_data = pickle.load(open(os.path.join(prefix, 'task1_ss/data/VL_40_key_seq_contact_idx.pkl'), 'rb'))

    # preparing multiprocessing 
    args_list = [
        (i, name, seq, gt_connects, from_pretrained)
        for i, (name, seq, gt_connects, _) in enumerate(valid_data)
    ]

    # initialize metrics
    metrics = [[[[] for _ in range(num_shresh)] for _ in range(num_heads)] for _ in range(num_layers)]

    # 
    with Pool(processes=num_cores) as pool:
        results = pool.map(process_one_sequence, args_list)

    # 
    for metrics_local in results:
        for layer in range(num_layers):
            for head in range(num_heads):
                for shresh_idx in range(num_shresh):
                    metrics[layer][head][shresh_idx].extend(metrics_local[layer][head][shresh_idx])

    # Step 3: find the best results 
    best_score = -float("inf")
    best_config = None
    avg_metrics = [[[None for _ in range(num_shresh)] for _ in range(num_heads)] for _ in range(num_layers)]

    for layer in range(num_layers):
        for head in range(num_heads):
            for s_idx in range(num_shresh):
                all_results = metrics[layer][head][s_idx]
                if all_results:
                    avg_f1 = np.mean(all_results)
                    avg_metrics[layer][head][s_idx] = avg_f1
                    if avg_f1 > best_score:
                        best_score = avg_f1
                        best_config = (layer, head, shresh_values[s_idx])

    print(f"Best configuration: Layer {best_config[0]}, Head {best_config[1]}, Threshold {best_config[2]:.3f}，平均 f1: {best_score:.4f}")

    # Step 4: save metrics and avg_metrics
    os.makedirs(os.path.join(prefix, "task1_ss"), exist_ok=True)
    with open(os.path.join(prefix, "task1_ss/metrics_per_point.pkl"), "wb") as f:
        pickle.dump(metrics, f)

    with open(os.path.join(prefix, "task1_ss/avg_metrics.pkl"), "wb") as f:
        pickle.dump(avg_metrics, f)
