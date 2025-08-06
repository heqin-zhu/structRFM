import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import os
import numpy as np
import os
import sys
import time
import random
import argparse, json
import torch
import torch.optim as optim
import subprocess
from torch.utils.data import DataLoader
from tqdm import tqdm
from train_classification import train,val,test
from model.load_net import get_model 
from data.RNAGraph import RNADataset,collate_all
import torch.nn as nn
import matplotlib.pyplot as plt
import csv
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import sys
import dgl
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import roc_curve, auc, roc_auc_score,f1_score,accuracy_score,precision_score,recall_score,confusion_matrix
import pandas as pd
d = {'A': torch.tensor([[1., 0., 0., 0.]]),
     'G': torch.tensor([[0., 1., 0., 0.]]),
     'C': torch.tensor([[0., 0., 1., 0.]]),
     'T': torch.tensor([[0., 0., 0., 1.]])}

def graph2seq(g):
    torch.set_printoptions(profile="full")
    feat = g.ndata['feat'] # num x feat
    start, first_flag = 0, 0
    for batch_num in g.batch_num_nodes(): 
        if first_flag == 0:
            output = torch.transpose(feat[start:start + batch_num], 1, 0).unsqueeze(0)
            first_flag = 1
        else:
            output = torch.cat([output, torch.transpose(feat[start:start + batch_num], 1, 0).unsqueeze(0)], dim=0)
        start += batch_num
    output = torch.transpose(output, 1, 2)
    return output


def save_date(save_path,id,seq,scores,lables):
    
    save_dir = save_path
    if os.path.exists(save_dir) is False:
        os.makedirs(save_dir)
    if os.path.exists(save_dir) is False:
        os.makedirs(save_dir)
    df = pd.DataFrame({
    "ID": id,
    "Sequence": seq,
    "Label": lables
})
    scores_df = pd.DataFrame(scores, columns=['Score_0', 'Score_1'])
    final_df = pd.concat([df, scores_df], axis=1)    
    final_df.to_csv(save_dir + "/data.csv", index=False)
def save_record(file,path, print_str,save_name):
    m_dir = path
    if os.path.exists(m_dir) is False:
        os.makedirs(m_dir)
    
    with open(m_dir + save_name + '.txt', 'a') as f:
        f.write(file +':'+ '\n')
        f.write(print_str + '\n')

def select_free_gpu():
    try:
        result = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.free,memory.total", "--format=csv,nounits,noheader"],
            stderr=subprocess.STDOUT
        )
        # 解码并按行处理输出
        result = result.decode("utf-8").strip().splitlines()
        #print(result)
        #sys.exit()
        gpu_info = []
        for line in result:
            index, free_mem, total_mem = line.split(', ')
            free_mem = int(free_mem)
            total_mem = int(total_mem)
            gpu_info.append((index, free_mem, total_mem))
        print(gpu_info)
        gpu_info = sorted(gpu_info, key=lambda x: x[1], reverse=True)
        selected_gpu_id = gpu_info[0][0]
        free_memory = gpu_info[0][1]
        print(f"Selected GPU: {selected_gpu_id}, Free Memory: {free_memory} MiB")
        return selected_gpu_id

    except subprocess.CalledProcessError as e:
        print(f"Error executing nvidia-smi: {e.output.decode()}")
        return None  # 如果出错，返回 None，避免崩溃
def gpu_setup(use_gpu, gpu_id):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    if torch.cuda.is_available() and use_gpu:
        print('cuda available with GPU:' ,torch.cuda.get_device_name(0))
        device = torch.device("cuda")
    else:
        print('cuda not available')
        device = torch.device("cpu")
    return device
#
def load_model(MODEL_NAME, net_params,path):

    model = get_model(MODEL_NAME, net_params)
    PATH = path
    model.load_state_dict(torch.load(PATH))
    return model

def soft_voting(models,device, data_loader, use_fea,path = "",use_softmax=False):
    #让所有模型进入评估模式
    for model in models:
        model.eval()
    batch_id_array,batch_seq_array,batch_scores_array, batch_labels_array = np.array([]), np.array([]),np.array([]),np.array([])
    start_flag = 1
    with torch.no_grad():
        for iter, (batch_ids,batch_seqs,labels,ss_graphs,bpe_graphs,bpp_graphs) in enumerate(data_loader):
            bpe_batch_graphs = dgl.batch(bpe_graphs)
            bpe_batch_graphs = bpe_batch_graphs.to(device)
            bpe_batch_graphs.ndata['feat'] = bpe_batch_graphs.ndata['feat'].to(device)
            bpe_batch_graphs.edata['feat'] = bpe_batch_graphs.edata['feat'].to(device)
            
            batch_x = bpe_batch_graphs.ndata['feat'] # num x feat
            #获取真实标签
            batch_labels = labels.to(device)

            #开始训练

            batch_scores =torch.stack([model.forward(bpe_batch_graphs,batch_seqs) for model in models],0).mean(0)   

            #计算数值
            batch_id_numpy = batch_ids
            batch_seq_numpy = batch_seqs

            if use_softmax:
                batch_scores_numpy = F.softmax(batch_scores, dim=1).detach().cpu().numpy()
            else :
                batch_scores_numpy = batch_scores.squeeze().detach().cpu().numpy() 
            batch_labels_numpy = batch_labels.cpu().numpy()
            if start_flag:
                batch_id_array,batch_seq_array= batch_id_numpy,batch_seq_numpy
                batch_scores_array, batch_labels_array = batch_scores_numpy, batch_labels_numpy
                start_flag = 0
            else:
                batch_id_array = np.hstack((batch_id_array, batch_id_numpy))
                batch_seq_array = np.hstack((batch_seq_array, batch_seq_numpy))                
                batch_scores_array = np.vstack((batch_scores_array, batch_scores_numpy))
                batch_labels_array = np.hstack((batch_labels_array, batch_labels_numpy))

        if use_softmax:
            save_date(path,batch_id_array,batch_seq_array,batch_scores_array,batch_labels_array)
            pred_lable = np.argmax(batch_scores_array, axis=1)
            epoch_acc = accuracy_score(batch_labels_array,pred_lable)
            epoch_auc = roc_auc_score(batch_labels_array, batch_scores_array[:, 1])
            epoch_f1_score = f1_score(batch_labels_array,pred_lable)
            epoch_precision = precision_score(batch_labels_array,pred_lable)
            epoch_recall = recall_score(batch_labels_array,pred_lable)
            epoch_confusion_matrix = confusion_matrix(batch_labels_array,pred_lable)
    
    return epoch_acc, epoch_auc,epoch_f1_score,epoch_precision,epoch_recall,epoch_confusion_matrix

if __name__ == '__main__':
    gpu_id = select_free_gpu()
    if gpu_id is not None:
        gpu_use = True
        print(f"Selected GPU: {gpu_id}")
    else:
         gpu_use = False
         print("No suitable GPU found.")  
    device = gpu_setup(gpu_use,gpu_id)

    net_params = {
        "L": 2,
        "in_dim":4,
        "hidden_dim": 32,
        "out_dim": 32,
        "residual": False,
        "readout": "mean",
        "in_feat_dropout": 0.1,
        "dropout": 0.1,
        "batch_norm": True,
        "self_loop": False,
        "embedding_dim":0,
        "gpu_id": gpu_id,
        "type": "GCN",
        "n_classes": 2,
        "use_softmax":True
    }
    net_params['device'] = device
    net_params['fea_info'] = 4 

    #语言模型参数设置
    net_params['pretrain'] =  os.getenv('structRFM_checkpoint')
    net_params['training'] = True
    print("net_params:",net_params)

    load_dataset_path = os.getcwd() +"/RNAFinalData/undersampling/ensemble_1"
    testset =  RNADataset(load_dataset_path + "/test")
    test_loader = DataLoader(testset, batch_size= 1, shuffle=False, collate_fn=collate_all)
    #模型参数路径 path
    model_base_path = os.getcwd() +'/save_model/ensemble/'

    over_path = model_base_path + 'structRFM/over/model.pth'
    under1_path = model_base_path + 'structRFM/under1/model.pth'
    under2_path = model_base_path + 'structRFM/under2/model.pth'
    under3_path = model_base_path + 'structRFM/under3/model.pth'
    save_path = model_base_path + 'structRFM/'

    over_model =   load_model("IreeSeek_LM_model",net_params,over_path).to(device)
    under1_model = load_model("IreeSeek_LM_model",net_params,under1_path).to(device)
    under2_model = load_model("IreeSeek_LM_model",net_params,under2_path).to(device)
    under3_model = load_model("IreeSeek_LM_model",net_params,under3_path).to(device)
    

    models =[over_model,under1_model,under2_model,under3_model]

    test_acc,\
    test_auc,\
    test_f1,\
    precision,\
    test_recall,\
    test_confusion_matrix = soft_voting(models,
                                        device,
                                        test_loader,
                                        use_fea=net_params['fea_info'],
                                        path = save_path,
                                        use_softmax =net_params["use_softmax"])
 
    print_str = ' test_acc ' + str(test_acc) + \
                ' test_roc_auc ' + str(test_auc) + ' test_f1 '+str(test_f1)+\
                ' test_precision ' + str(precision) + ' test_recall ' + str(test_recall)
    print(print_str)
