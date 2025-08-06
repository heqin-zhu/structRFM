import os
import pickle
from scipy.spatial.distance import cdist
import numpy as np
import multiprocessing as mp
import itertools
#import utils
#from utils.general_utils import Pool
#from utils.rna_utils import load_mat, load_seq
import sys
import dgl
import torch
import torch.utils.data
import time
import csv
from sklearn.model_selection import StratifiedShuffleSplit
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
class RNADataset(torch.utils.data.Dataset):
    def __init__(self, file_dir):
        self.file_dir = file_dir
        self.pkl_files = [f for f in os.listdir(file_dir) if f.endswith('.pkl')]

    def __len__(self):
        return len(self.pkl_files)

    def __getitem__(self, idx):
        pkl_dir = self.file_dir
        pkl_file = self.pkl_files[idx]
        pkl_path = os.path.join(pkl_dir,pkl_file)
        if os.path.exists(pkl_path):
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
            return data
        else:
            print("Don't find data,please cheack data")
            return None
    
    # form a mini batch from a given list of samples = [(graph, label) pairs]
  

class RNADataset_conv(torch.utils.data.Dataset):
    def __init__(self, file_dir, ss_dir, bpe_dir, bpp_dir):
        '''
            file_dir: # t
                seq
                ss
                bpe
                bpp
        '''
        self.file_dir = file_dir
        self.pkl_files = [f for f in os.listdir(file_dir) if f.endswith('.pkl')]
        self.names = [f[:f.rfind('.')].split('_')[0] for f in self.pkl_files]
        self.ss_files = [os.path.join(ss_dir, name+'.npy') for name in self.names]
        self.bpe_files = [os.path.join(bpe_dir, name+'.npy') for name in self.names]
        self.bpp_files = [os.path.join(bpp_dir, name+'.npy') for name in self.names]

        self.one_hot = {'A': torch.tensor([[1., 0., 0., 0.]]),
         'G': torch.tensor([[0., 1., 0., 0.]]),
         'C': torch.tensor([[0., 0., 1., 0.]]),
         'U': torch.tensor([[0., 0., 0., 1.]]),
         'T': torch.tensor([[0., 0., 0., 1.]])}
        self.vocab = {'A': 0, 'U': 1, 'T':1,'G': 2, 'C': 3}
    def __len__(self):
        return len(self.pkl_files)

    def __getitem__(self, idx):
        data_dic = {}
        pkl_dir = self.file_dir
        pkl_file = self.pkl_files[idx]
        pkl_path = os.path.join(pkl_dir,pkl_file)
        data = None
        
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
            data_dic['seq'] = data[1]
            #print(data_dic['seq'])
            #data_dic['seq'] = torch.cat([self.one_hot[base] for base in data_dic['seq']],dim=0)
            data_dic['seq'] = torch.tensor([self.vocab[base] for base in data_dic['seq']])
            data_dic['id'] = data[0]
            #print("pkl id:",data_dic['id'])
            data_dic['label'] = data[-1]
            #print(data_dic['label'], type(data_dic['label']))
        #print("ss id:",self.ss_files[idx])
        #print("bpe id:",self.bpe_files[idx])
        #print("bpp id:",self.bpp_files[idx])
        data_dic['ss'] = torch.FloatTensor(np.load(self.ss_files[idx]))
        data_dic['bpe'] = torch.FloatTensor(np.load(self.bpe_files[idx]))
        data_dic['bpp'] = torch.FloatTensor(np.load(self.bpp_files[idx]))
        for key in ['ss', 'bpe', 'bpp']:
            mat =  data_dic[key]
            if len(data_dic[key].shape) == 2:
                data_dic[key] = data_dic[key].unsqueeze(0)
            assert data_dic[key].shape == (1, 174, 174)
        return data_dic
        
def load_fasta_format(file):
    all_id = []
    all_seq = []
    seq = ''
    for row in file:
        if type(row) is bytes:
            row = row.decode('utf-8')
        row = row.rstrip()
        if row.startswith('>'):
            all_id.append(row.lstrip('>'))
            if seq != '':
                all_seq.append(seq)
                seq = ''
        else:
            seq += row
    all_seq.append(seq)
    return all_id, all_seq

def load_seq(filepath):
    if filepath.endswith('.fa'):
        file = open(filepath, 'r')
    else:
        file = gzip.open(filepath, 'rb')

    all_id, all_seq = load_fasta_format(file)
    for i in range(len(all_seq)):
        seq = all_seq[i]
        all_seq[i] = seq.upper().replace('T', 'U') 
    return all_id, all_seq

def collate_all(samples):
    ids,seqs,ss_graphs,eng_graphs,bpp_graphs,labels = zip(*samples)
    ids = np.array(ids)
    seqs =np.array(seqs)
    labels = torch.tensor(np.array(labels))
    return ids,seqs,labels,ss_graphs,eng_graphs,bpp_graphs
