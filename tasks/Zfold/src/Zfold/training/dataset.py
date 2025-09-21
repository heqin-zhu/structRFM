import pickle
from collections import defaultdict

from pathlib import Path

import copy, sys
import os.path
import zipfile
import zlib
import random
import warnings
import torch
import torch.nn as nn
import torch.distributed as distri
import numpy as np

from torch.utils.data import Dataset, DataLoader
from copy import deepcopy

base_dir = str(Path(os.path.dirname(os.path.abspath(__file__))).parent)
sys.path.insert(0, base_dir)
from network.config import obj, n_bins
from training.utils_training import subsample

IDX_BASE_DIC = {i: base for i, base in enumerate('AUCG')}
BASE_IDX_DIC = {base: i for i, base in IDX_BASE_DIC.items()}

bins_size = 1
dist_max = 40
dist_min = 3
dist_nbins = int((dist_max - dist_min) / bins_size + 1)
dist_bins = np.linspace(dist_min, dist_max, int((dist_max - dist_min) / bins_size + 1))
angle_bins = np.linspace(0.0, np.pi, 13)
dihe_bins = np.linspace(-np.pi, np.pi, 25)
angle1d_bins = np.linspace(70, 160, int((160 - 70) / 7.5 + 1)) * np.pi / 180

n_bins = {
    'C1\'': dist_nbins,
    'C4': dist_nbins,
    'P': dist_nbins,
    'N1': dist_nbins,
    'C3\'': dist_nbins,
    'CiNj': dist_nbins,
    'PiNj': dist_nbins,
    'Ci-1PiCi': 13,
    'Ci-1PiCiPi+1': 25,
    'PiCiPi+1': 13,
    'PiCiPi+1Ci+1': 25,
    'CjNiCiPi+1': 25,
    'CiNiCj': 13,
    'Pj+1CjNiCi': 25,
    'NiCjPj+1': 13,
    'NiCjPj+1Cj+1': 25,
    'contact': 1
}
bins = {
    'C1\'': dist_bins,
    'C4': dist_bins,
    'P': dist_bins,
    'N1': dist_bins,
    'C3\'': dist_bins,
    'CiNj': dist_bins,
    'PiNj': dist_bins,
    'Ci-1PiCi': angle1d_bins,
    'Ci-1PiCiPi+1': dihe_bins,
    'PiCiPi+1': angle1d_bins,
    'PiCiPi+1Ci+1': dihe_bins,
    'CjNiCiPi+1': dihe_bins,
    'CiNiCj': angle_bins,
    'Pj+1CjNiCi': dihe_bins,
    'NiCjPj+1': angle_bins,
    'NiCjPj+1Cj+1': dihe_bins,
}


class MSADataset(Dataset):
    def __init__(self,
                 lst,
                 npz_dir,
                 msa_cutoff=200,
                 lengthmax=300,
                 random_=True,
                 subsample_msa=True,
                 is_distill=False,
                 warning=False,
                 stru_feat_type='SS',
                 LM=None,
                 freeze_LM=False,
                 use_outer_product_mean=False,
                 ):
        super(MSADataset, self).__init__()

        self.lst = lst
        self.npz_dir = npz_dir
        self.msa_cutoff = msa_cutoff
        self.lengthmax = lengthmax
        self.random_ = random_
        self.subsample_msa = subsample_msa and random_
        self.is_distill = is_distill
        self.warning = warning

        self.use_outer_product_mean = use_outer_product_mean
        self.stru_feat_type = stru_feat_type
        if self.stru_feat_type!='SS':
            parent_dir = os.path.dirname(self.npz_dir)
            self.LM_feat_dir = os.path.join(parent_dir, 'LM_feat')
            os.makedirs(self.LM_feat_dir, exist_ok=True)
            self.LM = LM
        self.LM_feat_cache = {}
        self.freeze_LM = freeze_LM

    def __len__(self):
        return len(self.lst)

    def __getitem__(self, idx, return_npz=False):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        # file name
        pid = self.lst[idx]
        npz_file = f'{self.npz_dir}/{pid}.npz'
        if os.path.isfile(npz_file):
            raw = np.load(npz_file, allow_pickle=True)
        else:
            if self.warning:
                warn = f'miss {npz_file}!'
                warnings.warn(warn)
            return {}

        if np.isnan(raw["C3'"]).all():
            if self.warning:
                warn = f'{npz_file} only has P atom!'
                warnings.warn(warn)
            return {}

        sample = {}
        seq_arr, ss = self.get_SS_and_seq(raw, pid)
        sample['seq_arr'] = seq_arr
        sample['ss'] = ss

        aln = raw['aln']
        if len(aln.shape) == 1:
            return {}
        if len(aln.shape) == 3:
            if aln.shape[0] == 1:
                aln = aln[0]
            else:
                if self.warning:
                    warn = f'{npz_file} msa has shape {raw["aln"].shape}!'
                    warnings.warn(warn)
                return {}

        if self.subsample_msa:
            sample['aln'] = subsample(aln, limit=self.msa_cutoff)
        else:
            sample['aln'] = aln[:self.msa_cutoff]

        l = sample['aln'].shape[-1]
        if 'conf' in raw:
            conf_mask = (raw['conf'] <= .5)
            if (~conf_mask).sum() < 10: return {}
            conf_mask_2d = conf_mask[None, :] | conf_mask[:, None]
            sample['conf_mask'] = conf_mask.long()
            sample['conf_mask_2d'] = conf_mask_2d.long()
        else:
            sample['conf_mask'] = np.zeros(l)
            sample['conf_mask_2d'] = np.zeros((l, l))

        idx = np.arange(l)
        shape_er = False
        if l > self.lengthmax:
            try:
                if self.is_distill:
                    pair_mask = np.min(np.stack([raw['P'][..., 0], raw['N1'][..., 0], raw['C4'][..., 0],
                                                 raw["C3'"][..., 0], raw["C1'"][..., 0]]), axis=0) < 0.5
                else:
                    pair_mask = np.min(np.stack([raw['P'], raw['N1'], raw['PiNj'], raw['PiNj'].T]), axis=0) < 12
                try:
                    if self.random_:
                        point = random.choice(idx)
                    else:
                        point = 0
                    crop = idx[pair_mask[point]]
                    while len(crop) < self.lengthmax:
                        crop_new = idx[(pair_mask[crop]).max(0).astype(bool)]
                        if len(crop_new) == len(crop):
                            break
                        if len(crop_new) <= self.lengthmax:
                            crop = crop_new
                        else:
                            break
                except Exception:
                    crop = idx[:self.lengthmax]
            except ValueError:
                return {}
        else:
            crop = np.arange(l)

        try:
            sample['labels'] = self.parse_labels(raw, l, crop)
        except Exception as e:
            if 'shape' in str(e):
                if self.warning:
                    warn = f'shape error!'
                    warnings.warn(warn)
                return {}
            else:
                raise e

        sample['seq_arr'] = sample['seq_arr'][crop]
        seq = ''.join([IDX_BASE_DIC[i] for i in sample['seq_arr']])
        sample['aln'] = sample['aln'][:, crop]
        sample['ss'] = torch.tensor(sample['ss'][crop][:, crop]).unsqueeze(-1)
        if self.stru_feat_type != 'SS':
            # L x L x 1
            LM_feat = self.get_LM_feat(seq).unsqueeze(-1)
            if self.stru_feat_type == 'LM':
                sample['ss'] = LM_feat
            elif self.stru_feat_type == 'both':
                sample['ss'] = torch.cat([sample['ss'].to(LM_feat.device), LM_feat], dim=-1)
        sample['conf_mask'] = sample['conf_mask'][crop]
        sample['conf_mask_2d'] = sample['conf_mask_2d'][crop][:, crop]

        if not sample['ss'].shape[1] == sample['ss'].shape[0]:
            if self.warning:
                warn = f'inconsistent msa shape {sample["msa"].shape} and ss shape {sample["ss"].shape}!'
                warnings.warn(warn)
            return {}

        if shape_er:
            if self.warning:
                warn = f'shape err!'
                warnings.warn(warn)
            return {}

        sample['distance'] = {k: raw[k][crop][:, crop] for k in ['P', 'C3\'', 'C1\'', 'C4', 'N1']}
        sample['idx'] = crop

        return sample


    def get_SS_and_seq(self, raw, pid, use_SPOTRNA=True):
        if 'aln' not in raw:
            raise Exception(f'[Error] no aln: {pid}')
        aln = raw['aln']
        if len(aln.shape)==1:
            print(f'[Warning] {pid} aln has shape {aln.shape}')
            seq_arr = aln
        elif len(aln.shape)==2:
            seq_arr = aln[0]
        elif len(aln.shape)==3:
            print(f'[Warning] {pid} aln has shape {aln.shape}')
            seq_arr = aln[0][0]
        else:
            raise Exception(f'[Error] aln shape {raw["aln"].shape}')

        if use_SPOTRNA:
            if 'ss' in raw:
                return seq_arr, raw['ss'] + raw['ss'].T
            else:
                raise Exception(f'Not implemented: use SPOTRNA to pred SS for {pid}')
        else:
            raise Exception(f'Not implemented: use SPOTRNA instead: {pid}')


    def get_LM_feat(self, seq, use_cache=False):
        if self.freeze_LM and use_cache:
            if seq not in self.LM_feat_cache:
                feat = self.LM.extract_raw_feature(seq, return_all=False)[1:-1, :]
                feat2d = self.get_matrix_feature(feat)
                self.LM_feat_cache[seq] = feat2d
            feat2d = self.LM_feat_cache[seq]
        else:
            feat = self.LM.extract_raw_feature(seq, return_all=False, is_training=not self.freeze_LM)[1:-1, :]
            feat2d = self.get_matrix_feature(feat)
        return feat2d

    def get_matrix_feature(self, LM_embed):
        if self.use_outer_product_mean:
            shape = LM_embed.shape
            return (LM_embed.unsqueeze(-3).unsqueeze(-2) * LM_embed.unsqueeze(-2).unsqueeze(-1)).reshape(*shape[:-1], shape[-2], -1).mean(dim=-1)
        else:
            return LM_embed @ LM_embed.transpose(1, 0)

    def parse_labels(self, raw, l, crop=None):
        if crop is None: crop = np.arange(l)
        labels = defaultdict(dict)
        for k in obj:
            if self.is_distill and k == '1D': continue
            for kk in obj[k]:
                for a in obj[k][kk]:
                    if a in raw:
                        label = raw[a]
                    elif a == 'all atom':
                        labels['contact'] = raw['contact'][crop][:, crop]
                        continue
                    else:
                        continue

                    if not (l == label.shape[0]):
                        raise ValueError('shape err!')

                    if label.ndim == 1 or (self.is_distill and label.ndim == 2):
                        label = label[crop]
                    elif label.ndim == 2 or (self.is_distill and label.ndim == 3):
                        label = label[crop][:, crop]
                    else:
                        raise ValueError(f'label {a} has shape {label.shape}!')

                    if self.is_distill:
                        labels[kk][a] = label
                    else:
                        binned = np.digitize(label, bins[a])

                        if len(k) < 6:  # for distances
                            binned[(binned >= n_bins[a]) & (~np.isnan(label))] = 0
                            if k == 'P': nocontact_mask = deepcopy(binned == 0)
                        elif label.ndim == 2:  # for orientations
                            binned[nocontact_mask] = 0

                        onehot = (np.arange(n_bins[a]) == binned[..., None]).astype(np.uint8)
                        if k == '2D':
                            labels[kk][a] = onehot
                        else:
                            labels['torsion'][a] = onehot

        return labels
