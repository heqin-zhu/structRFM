"""
    Utility functions for training one epoch
    and evaluating one epoch
"""
import torch
import torch.nn as nn
import math
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import dgl
import torch.nn.functional as F
from sklearn.metrics import roc_curve, auc, roc_auc_score,f1_score,accuracy_score,precision_score,recall_score,confusion_matrix
from model.metrics import accuracy_binary,accuracy_softmax
from model.loss import *
import os
import pandas as pd
import sys
# 1. 创建词汇表
vocab = {'A': 1, 'T': 2, 'C': 3, 'G': 4}
def tokenize_rna(sequence):
    return [vocab[char] for char in sequence]

def save_date(epoch,save_path,id,seq,scores,probs,lables):
    m_path = save_path
    save_dir = "./"+m_path+"/test_data"
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
    prob_df = pd.DataFrame(probs,columns=['Prob_0', 'Prob_1'])
    final_df = pd.concat([df, scores_df,prob_df], axis=1)    
    final_df.to_csv(save_dir + "/epoch_"+str(epoch)+".csv", index=False)

def train(model, 
          optimizer, 
          criterion,
          device, 
          data_loader, 
          epoch,
          use_fea,
          use_softmax=False):
    model.train()
    epoch_loss = 0
    epoch_train_acc = 0
    nb_data = 0
    gpu_mem = 0
    parts = 1
    grh_prob =[]
    accuracy = accuracy_softmax if use_softmax else accuracy_binary
    for iter, (batch_ids,batch_seqs,labels,ss_graphs,bpe_graphs,bpp_graphs) in enumerate(data_loader):
        batch_size = len(labels)
        bpe_batch_graphs = dgl.batch(bpe_graphs)
        bpe_batch_graphs = bpe_batch_graphs.to(device)
        bpe_batch_graphs.ndata['feat'] = bpe_batch_graphs.ndata['feat'].to(device)
        bpe_batch_graphs.edata['feat'] = bpe_batch_graphs.edata['feat'].to(device)
        
        batch_x = bpe_batch_graphs.ndata['feat']        #num  x   [num = batch X len]
        #batch_seqs = np.char.replace(batch_seqs, "T", "U")

        #获取真实标签
        batch_labels = labels.to(device)
        batch_scores = model.forward(bpe_batch_graphs,batch_seqs)
   
        #计算数据
        if use_softmax :
            loss = criterion(batch_scores, batch_labels)
        else:
            loss = criterion(batch_scores.squeeze(), batch_labels.float())
        loss.backward(retain_graph=True)
        optimizer.step()
        optimizer.zero_grad()
        torch.cuda.empty_cache()
        epoch_loss += loss.detach().item()
        epoch_train_acc += accuracy(batch_scores, batch_labels)
        nb_data += batch_labels.size(0)
    epoch_loss /= (iter + 1)
    epoch_train_acc /= nb_data

    return epoch_loss, epoch_train_acc, optimizer
def val(model,
        criterion,
        device, 
        data_loader, 
        epoch,
        use_fea,
        use_softmax=False):
    model.eval()
    epoch_loss = 0
    epoch_acc = 0
    epoch_auc = 0
    nb_data = 0
    batch_scores_array, batch_labels_array = np.array([]), np.array([])
    start_flag = 1
    accuracy = accuracy_softmax if use_softmax else accuracy_binary
    with torch.no_grad():
        for iter, (batch_ids,batch_seqs,labels,ss_graphs,bpe_graphs,bpp_graphs) in enumerate(data_loader):
            bpe_batch_graphs = dgl.batch(bpe_graphs)
            bpe_batch_graphs = bpe_batch_graphs.to(device)
            bpe_batch_graphs.ndata['feat'] = bpe_batch_graphs.ndata['feat'].to(device)
            bpe_batch_graphs.edata['feat'] = bpe_batch_graphs.edata['feat'].to(device)
            
            batch_x = bpe_batch_graphs.ndata['feat'] # num x feat
            #获取真实标签
            batch_labels = labels.to(device)
            batch_scores = model.forward(bpe_batch_graphs,batch_seqs)
            #计算数值
            if use_softmax :
                loss = criterion(batch_scores, batch_labels)
            else:
                loss = criterion(batch_scores.squeeze(), batch_labels.float())
        
            epoch_loss += loss.detach().item()
            epoch_acc += accuracy(batch_scores, batch_labels)
            nb_data += batch_labels.size(0)
        epoch_loss /= (iter + 1)
        epoch_acc /= nb_data 
    return epoch_loss, epoch_acc


def test(model,
         device, 
         data_loader, 
         epoch,
         path,
         use_fea,
         use_softmax=False):
    model.eval()
    epoch_acc = 0
    epoch_auc = 0
    nb_data = 0
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
            batch_scores = model.forward(bpe_batch_graphs,batch_seqs)

            #计算数值
            batch_id_numpy = batch_ids
            batch_seq_numpy = batch_seqs
        
            if use_softmax:
                batch_prob_numpy = F.softmax(batch_scores, dim=1).detach().cpu().numpy()
                #print("batch_prob_numpy shape:",batch_prob_numpy.shape)
                batch_scores_numpy = batch_scores.squeeze().detach().cpu().numpy()
                #print("batch_scores_numpy shape:",batch_scores_numpy.shape)
            else :
                batch_scores_numpy = batch_scores.squeeze().detach().cpu().numpy() 
            batch_labels_numpy = batch_labels.cpu().numpy()
            if start_flag:
                batch_id_array,batch_seq_array= batch_id_numpy,batch_seq_numpy
                batch_scores_array = batch_scores_numpy
                batch_prob_array= batch_prob_numpy
                batch_labels_array = batch_labels_numpy
                start_flag = 0
            else:
                batch_id_array = np.hstack((batch_id_array, batch_id_numpy))
                batch_seq_array = np.hstack((batch_seq_array, batch_seq_numpy))
                batch_scores_array = np.vstack((batch_scores_array, batch_scores_numpy))                
                batch_prob_array = np.vstack((batch_prob_array, batch_prob_numpy))
                batch_labels_array = np.hstack((batch_labels_array, batch_labels_numpy))  
            nb_data += len(batch_labels)

        #print(batch_scores_array.shape)
        #sys.exit()
        #保存数据文件
        if use_softmax:
            save_date(epoch,path,batch_id_array,batch_seq_array,batch_scores_array,batch_prob_array,batch_labels_array)
            pred_lable = np.argmax(batch_prob_array, axis=1)
            epoch_acc = accuracy_score(batch_labels_array,pred_lable)
            epoch_auc = roc_auc_score(batch_labels_array, batch_prob_array[:, 1])
            epoch_f1_score = f1_score(batch_labels_array,pred_lable)
            epoch_precision = precision_score(batch_labels_array,pred_lable)
            epoch_recall = recall_score(batch_labels_array,pred_lable)
            epoch_confusion_matrix = confusion_matrix(batch_labels_array,pred_lable)

        '''
        else:
            save_date(epoch,path,batch_id_array,batch_seq_array,batch_scores_array,batch_labels_array)
            #获得预测标签
            pred_lable =  (batch_scores_array >= 0.5).astype(int)
            epoch_acc = accuracy_score(batch_labels_array,pred_lable)
            epoch_auc = roc_auc_score(batch_labels_array, batch_scores_array)
            epoch_f1_score = f1_score(batch_labels_array,pred_lable)
            epoch_precision = precision_score(batch_labels_array,pred_lable)
            epoch_recall = recall_score(batch_labels_array,pred_lable)
            epoch_confusion_matrix = confusion_matrix(batch_labels_array,pred_lable)
        '''

    return epoch_acc, epoch_auc,epoch_f1_score,epoch_precision,epoch_recall,epoch_confusion_matrix
