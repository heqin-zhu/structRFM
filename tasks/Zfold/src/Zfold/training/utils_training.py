import sys

import os

from pathlib import Path

import random

import json
import numpy as np
import torch


def empty_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def subsample(msa, limit=30000):
    nr, nc = msa.shape
    if nr < 10: return msa
    n = min(limit, int(10 ** np.random.uniform(np.log10(nr)) - 10))
    if n <= 0: return msa[0:1]
    indices = sorted(random.sample(range(1, nr), n) + [0])
    return msa[indices]


def cont_loss(pred, real, conf_mask2d=None):
    pred = pred.squeeze()
    real = real.squeeze()
    assert pred.size() == real.size() and pred.ndim == 2
    nan_mask = torch.isnan(real)
    real = real[~nan_mask]
    pred = pred[~nan_mask]
    loss_mat = -((real * torch.log(pred + 1e-7)) + ((1 - real) * torch.log(1 - pred + 1e-7)))

    # n_pairs = (real >= 0).sum()
    # loss = torch.nansum(torch.where(real < 0, torch.zeros_like(loss_mat, device=pred.device), loss_mat)) / n_pairs
    if not (loss_mat > -1e-4).all():
        return 0
    if conf_mask2d is not None:
        conf_mask2d = conf_mask2d.squeeze()[~nan_mask]
        loss_mat = loss_mat * (1 - conf_mask2d)
        return (loss_mat.sum()) / (1 - conf_mask2d).sum()
    loss = loss_mat.mean()
    return loss


def cross_entropy(pred, real, conf_mask1d=None, conf_mask2d=None):
    pred = pred.squeeze()
    real = real.squeeze()
    ndim = pred.ndim
    if pred.size() != real.size():
        raise ValueError(f'pred has shape {pred.size()} but real has shape {real.size()}!')
    nanidx = torch.isnan(real[..., 0])
    real = real[~nanidx]
    pred = pred[~nanidx]
    loss_mat = -(real * torch.log(pred + 1e-7)).sum(dim=-1).squeeze()
    if ndim == 1 and conf_mask1d is not None:
        conf_mask1d = conf_mask1d.squeeze()[~nanidx]
        loss_mat = loss_mat * (1 - conf_mask1d)
        return (loss_mat.sum()) / (1 - conf_mask1d).sum()
    if ndim == 2 and conf_mask2d is not None:
        conf_mask2d = conf_mask2d.squeeze()[~nanidx]
        loss_mat = loss_mat * (1 - conf_mask2d)
        return (loss_mat.sum()) / (1 - conf_mask2d).sum()
    return loss_mat.mean()


def geometry_loss(pred_geom, native_geom, device, conf_mask1d=None, conf_mask2d=None):
    loss = 0
    for k in native_geom:
        if k == 'contact':
            loss += cont_loss(pred_geom[k].float(), native_geom[k].to(device), conf_mask2d)
        else:
            for kk in native_geom[k]:
                if conf_mask1d is not None and pred_geom[k][kk].squeeze().ndim <= 2:
                    continue
                loss += cross_entropy(pred_geom[k][kk].float(), native_geom[k][kk].to(device), conf_mask1d,
                                      conf_mask2d) / len(native_geom[k])
    return loss


def corr(ave, real):
    cov = np.nansum((ave - np.nanmean(ave)) * (real - np.nanmean(real)))
    vx = np.nansum((ave - np.nanmean(ave)) ** 2)
    vy = np.nansum((real - np.nanmean(real)) ** 2)
    corr_ = cov / (vx * vy) ** .5
    return corr_


def dist_corr(pred, dist, pcut=.05, bin_p=.01, sep=12):
    first_bin = 1
    first_value = 3.5
    bin_size = 1
    vmax = 40
    last_bin = int((vmax - 3) / bin_size)

    if pred.ndim == 4:
        pred = pred[0]
    if dist.ndim == 3:
        dist = dist[0]

    nres = int(pred.shape[0])
    i, j = np.meshgrid(np.arange(nres), np.arange(nres))
    sep_mask_1 = (i - j >= 1)
    sep_mask_12 = (i - j >= sep)

    n_bins = last_bin - first_bin + 1

    bin_values = first_value + bin_size * np.arange(n_bins)

    p_valid = np.where(pred > bin_p, pred, 0)[..., first_bin:last_bin + 1]
    pnorm = p_valid / p_valid.sum(axis=-1)[..., None]

    ave = (pnorm * bin_values[None, None, :]).sum(axis=-1)

    ave_12 = np.where((p_valid.sum(axis=-1) > pcut) & sep_mask_12, ave, np.nan)

    notnan = ~np.isnan(ave_12)
    ave_valid = ave_12[notnan]
    real_valid = dist[notnan]

    corr12 = corr(ave_valid, real_valid)
    return corr12


def save_to_json(obj, file):
    with open(file, "w") as f:
        jso = json.dumps(obj, cls=NpEncoder)
        f.write(jso)


def read_json(file):
    with open(file, 'r') as load_f:
        load_dict = json.load(load_f)
    return load_dict


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


class bcolors:
    RED = "\033[1;31m"
    BLUE = "\033[1;34m"
    CYAN = '\033[96m'
    GREEN = "\033[0;32m"
    RESET = "\033[0;0m"
    BOLD = "\033[;1m"
    REVERSE = "\033[;7m"
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    UNDERLINE = '\033[4m'
    HEADER = '\033[95m'
