import torch
import torchtext
import numpy as np
from torch import nn,Tensor
from torch.utils import data
import math
import torch.nn.functional as F
from torch.nn import TransformerEncoder,TransformerEncoderLayer
import sys
import torch
import torch.nn as nn
import math
import time
import dgl
import scipy.sparse as sp
import timm
# utils.pos_embed import *
#from model.IreSeek_swin import get_swin_model
import timm
import os
from structRFM.infer import structRFM_infer
class MLP(nn.Module):
    def __init__(self, in_dim, out_dim=None, hidden_dim=None, dropout=0.1):
        super().__init__()
        out_dim = out_dim or in_dim
        hidden_dim = hidden_dim or 4*in_dim
        self.layers = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.Dropout(dropout)
          )
        
    def forward(self, x):
        return self.layers(x)

class RNA_CNN(nn.Module):
    def __init__(self, 
                 input_dim=4,
                 feature_dim=32,
                 seq_len=174,
                 head = 8,
                 T_num_layers =4):
        super(RNA_CNN, self).__init__()
        print(f"nhead is {head} T_num_layers {T_num_layers} RNN")
        # 输入形状: (b, 174, 4)
        
        # 第一层卷积 (保持长度不变)
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=64, 
                              kernel_size=5, padding=2, padding_mode='replicate')
        self.bn1 = nn.BatchNorm1d(64)
        
        # 第二层卷积 (保持长度不变)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=3, padding=1, padding_mode='replicate')
        self.bn2 = nn.BatchNorm1d(64)
        
        # 第三层卷积 (保持长度不变)
        self.conv3 = nn.Conv1d(64, feature_dim, kernel_size=3, padding=1, padding_mode='replicate')
        self.bn3 = nn.BatchNorm1d(feature_dim)

        encoder_layers = TransformerEncoderLayer(
            d_model=feature_dim, 
            nhead=head,
            batch_first=True
        )
        self.transformer = TransformerEncoder(
            encoder_layer= encoder_layers,
            num_layers =T_num_layers
        )
        
        # 输出形状: (b, 174, 32)
        
    def forward(self, x):
        # 输入x形状: (b, 174, 4)
        x = x.permute(0, 2, 1)  # 转为(b, 4, 174)以适应Conv1d
        
        # 第一层
        x = F.relu(self.bn1(self.conv1(x)))  # (b, 64, 174)
        
        # 第二层
        x = F.relu(self.bn2(self.conv2(x)))  # (b, 64, 174)
        
        # 第三层
        x = F.relu(self.bn3(self.conv3(x)))  # (b, 32, 174)
        # 转回(b, 174, 32)的输出形状
        x = x.permute(0, 2, 1)
        x =self.transformer(x)
        return x


class IreeSeek_LM_model(nn.Module):
    def  __init__(self,
                  net_params):
        super().__init__()
        seq_size = 174
        input_dim = net_params['in_dim']
        hidden_dim = net_params['hidden_dim']
        out_dim = net_params['out_dim']
        self.fea_info = net_params['fea_info']

        self.pretrain = net_params['pretrain']
        self.training =  net_params['training']


        class_out_dim = 2 if net_params["use_softmax"] else 1
        self.cnn_transformer = RNA_CNN(input_dim,hidden_dim,seq_size)
        self.RNALM = structRFM_infer(from_pretrained= self.pretrain, max_length=514)
        
        feat_in_dim = seq_size*out_dim + seq_size * 768

        self.MLP_layer = MLP(in_dim=feat_in_dim,out_dim=class_out_dim, hidden_dim=128)

    def lm_feature(self,seqs):
        feature_list = [] 
        for i, seq in enumerate(seqs):
            features = self.RNALM.extract_raw_feature(seq, is_training=self.training)
            #print("features:",features[1:-1, :].shape)
            features = features[1:-1, :]
            feature_list.append(features)
        
        
        batch_features = torch.stack(feature_list, dim=0)
        #print("batch_features:", batch_features.shape)
        #sys.exit()
        return batch_features

        


    def forward(self,bpe_g =None,seqs=None):
        assert bpe_g is not None , "Error: bpe_g cannot be None or empty!"
        
        
        seq = self._graph2seq(bpe_g) #[b,174,4]

        seq_f = self.cnn_transformer(seq)
        seq_f = torch.flatten(seq_f, start_dim=1)

        lm_f = self.lm_feature(seqs)
        lm_f = torch.flatten(lm_f, start_dim=1)
         
        final = torch.cat([seq_f,lm_f], dim=1)
        pred = self.MLP_layer(final)
        #print(final.shape,final[:10,:10])
        #print(pred[:10])
        return pred


    def _graph2seq(self, g):
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

