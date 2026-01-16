import os

import numpy as np
import pandas as pd
from Bio import SeqIO

from torch.utils.data import Dataset


class SeqClsDataset(Dataset):
    def __init__(self, fasta_dir, prefix, tokenizer, seed=0, train=True, fast_test=False, max_seq_len=None):
        super(SeqClsDataset, self).__init__()

        self.fasta_dir = fasta_dir
        self.prefix = prefix
        self.tokenizer = tokenizer

        file_name = "train.fa" if train else "test.fa"
        fasta = os.path.join(fasta_dir, prefix, file_name)
        records = list(SeqIO.parse(fasta, "fasta"))
        self.data = [(str(x.seq), *x.description.split(" ")[0:2]) for x in records]
        if max_seq_len is not None:
            self.data = [x for x in self.data if len(x[0])<=max_seq_len]
        np_rng = np.random.RandomState(seed=seed)
        np_rng.shuffle(self.data)
        if fast_test:
            print('Fast testing...')
            if train:
                self.data = self.data[:500]
            else:
                self.data = self.data[:50]


    def __getitem__(self, idx):
        instance = self.data[idx]
        seq = instance[0]
        name = instance[1]
        label = instance[2]
        return {"seq": seq, "label": label, 'name': name}

    def __len__(self):
        return len(self.data)


class GenerateRRInterTrainTest:
    def __init__(self,
                 rr_dir,
                 dataset,
                 split=0.8,
                 seed=0):
        csv_path = os.path.join(rr_dir, dataset) + ".csv"
        self.data = pd.read_csv(csv_path, sep=",").values.tolist()

        self.split_index = int(len(self.data) * split)

        np_rng = np.random.RandomState(seed=seed)
        np_rng.shuffle(self.data)

    def get(self):
        return RRInterDataset(self.data[:self.split_index]), RRInterDataset(self.data[self.split_index:])


class RRInterDataset(Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data

    def __getitem__(self, idx):

        instance = self.data[idx]
        return {
            "a_name": instance[0],
            "a_seq": instance[1],
            "b_name": instance[2],
            "b_seq": instance[3],
            "label": instance[4],
        }

    def __len__(self):
        return len(self.data)


def bpseq2dotbracket(bpseq):
    dotbracket = []
    for i, x in enumerate(bpseq):
        if x == 0:
            dotbracket.append('.')
        elif x > i:
            dotbracket.append('(')
        else:
            dotbracket.append(')')
    return ''.join(dotbracket)


class BPseqDataset(Dataset):
    def __init__(self, data_dir):
        super().__init__()
        self.file_paths = []
        for pre, ds, fs in os.walk(data_dir):
            for f in fs:
                if any(f.endswith(suf) for suf in ['.ct', '.bpseq']):
                    self.file_paths.append(os.path.join(pre, f))

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        return self.load_bpseq(file_path)

    def load_bpseq(self, file_path):
        with open(file_path) as f:
            p = [0]
            s = ['']
            for line in f:
                line = line.rstrip('\n').split()
                idx, c, pair = line
                idx, pair = int(idx), int(pair)
                s.append(c)
                p.append(pair)
        seq = ''.join(s)
        return {"name": file_path, "seq": seq, "pairs": np.array(p)}
