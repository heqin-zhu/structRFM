from scipy.fftpack import fft
import numpy as np
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import random
import seaborn as sns
import os, sys
from scipy import stats
from sklearn.metrics import auc


plt.rcParams['font.family'] = 'Arial'

palette = sns.color_palette([
    '#3a7ca5', # blue
    '#4daf8c', # green
])


models = ['rsb25', 'structRFM', 'dnabert_k3', 'dnabert_k6', 'dnabert-2', 'nt', 'lucaone', 
          'utr_lm', 'splicebert', 'birna-bert_bpe', 'birna-bert_nuc', 
          'rna_fm', 'rinalmo_micro', 'rnaernie', 'ernie_rna', 'prot_rna', 'rinalmo_mega', 
          'rinalmo_giga', 'aido_rna_650m', 'rna_km', 'mp_rna', 'aido_rna_1600m']
model_names = ['RSB25', 'structRFM', 'DNABERT (k=3)', 'DNABERT (k=6)', 'DNABERT-2', 'NT', 'LucaOne', 
               'UTR_LM', 'SpliceBERT', 
               'BiRNA-BERT (BPE)', 'BiRNA-BERT (NUC)', 'RNA-FM', 'RiNALMo (micro)', 
               'RNAErnie', 'ERNIE-RNA', 'ProtRNA', 'RiNALMo (mega)', 'RiNALMo (giga)', 
               "AIDO.RNA (650M)", 'RNA-km', 'MP-RNA', "AIDO.RNA (1.6B)"] 

model_name_mapping = {k: v for k, v in zip(models, model_names)}

def find_max_min(df1, df2):
    max = np.maximum(df1, df2)
    min = np.minimum(df1, df2)
    return max, min


def cal_or_by_gaussian_kde(data1, data2, xlim, bins=100):
    x = np.linspace(xlim[0], xlim[1], bins, endpoint=True)
    kde = stats.gaussian_kde(data1)
    freq1 = kde(x)
    kde = stats.gaussian_kde(data2)
    freq2 = kde(x)    
    max, min = find_max_min(freq1, freq2)
    overlapping_ratio = sum(min) / sum(max)
    return overlapping_ratio


def fftCal(x: np.ndarray, scaler = 10, outdim=128):
    if len(x.shape) == 2:
        x = x.reshape(1, -1)
    elif len(x.shape) == 3:
        x = x.reshape(x.shape[0], -1)
    x = fft(scaler * x, outdim).real  # batch_size dont influent the FFT
    return x


def cosine_similarity(x1, x2):
    # x1: [b, d], x2: [b, d]

    assert x1.shape == x2.shape

    return np.sum(x1 * x2, axis=1) / (np.linalg.norm(x1, axis=1) * np.linalg.norm(x2, axis=1))


## calcualute the cosine similarity between two sets of embeddings 
def emb_archiveII_cos_sim_cal(features_pkl, 
                    save_path, model_name,
                    n_samples=100000, plot=True, xlim=None, xticks=None, gauss_kde=False, plot_title=False):
    with open(features_pkl, 'rb') as fr:
        dataset = pickle.load(fr)

    #labels_set = ['16s', '23s', '5s', 'RNaseP', 'grp1', 'srp', 'tRNA', 'telomerase', 'tmRNA']

    keys = dataset['keys']
    labels = dataset['labels']
    seq_repr = dataset['seq_repr']

    key_to_seq_repr = {keys[n]: seq_repr[n] for n in range(len(keys))}
    label_to_keys = {}
    for i, key in enumerate(keys):
        label_i = labels[i]
        if label_i in label_to_keys:
            label_to_keys[label_i].append(key)
        else:
            label_to_keys[label_i] = [key]
    unique_labels = list(label_to_keys.keys())
    print('Length of Data:', len(keys),'Unique labels:', unique_labels)

    pos_samples = np.zeros((n_samples, 2, seq_repr[0].size))
    neg_samples = np.zeros((n_samples, 2, seq_repr[0].size))

    idx = 0
    while True:
        sample_label = random.choice(unique_labels)
        if len(label_to_keys[sample_label]) > 2: # always True on archiveII
            key_a, key_b = random.sample(label_to_keys[sample_label], 2)
            pos_samples[idx, 0, :] = key_to_seq_repr[key_a]
            pos_samples[idx, 1, :] = key_to_seq_repr[key_b]

            label_a, label_b = random.sample(labels, 2)
            key_a, key_b = random.choice(label_to_keys[label_a]), random.choice(label_to_keys[label_b])
            neg_samples[idx, 0, :] = key_to_seq_repr[key_a]
            neg_samples[idx, 1, :] = key_to_seq_repr[key_b]

            idx += 1
        if idx == n_samples:
            break

    pos_cos_similarity = cosine_similarity(pos_samples[:, 0, :], pos_samples[:, 1, :])
    neg_cos_similarity = cosine_similarity(neg_samples[:, 0, :], neg_samples[:, 1, :])

    if plot:
        df = pd.DataFrame({'pos': pos_cos_similarity, 'neg': neg_cos_similarity})

        fig = plt.figure(figsize=(4, 2))
        ax = fig.add_subplot(111)
        fig.subplots_adjust(bottom=0.18)
        sns.kdeplot(data=df['pos'], color="grey", linestyle="-", legend=False, alpha=0.4,
                    bw_adjust=1, linewidth=1, fill=False)
        sns.kdeplot(data=df['pos'], ax=ax, color=palette[1], fill=True, legend=False, alpha=0.8,  
                    bw_adjust=1, 
                    palette=palette)   
            
        sns.kdeplot(data=df['neg'], color="grey", linestyle="-", legend=False, alpha=0.4,
                    bw_adjust=1, linewidth=1, fill=False)
        sns.kdeplot(data=df['neg'], ax=ax, color=palette[0], fill=True, legend=False, alpha=0.8,  
                    bw_adjust=1,
                    palette=palette)   

        #ax.set_xlabel('Cos. Sim.')
        ax.set_xlabel('')
        ax.set_ylabel('')
        if xlim is not None:
            ax.set_xlim(xlim)
        ax.set_yticks([])

        if xticks is not None:
            ax.set_xticks(xticks)
            ax.set_xticklabels(xticks, fontsize=20)

        overlap_ratio = -1
        if gauss_kde:
            if xlim is None:
                xlim = ax.get_xlim()
            x_data = plt.gca().get_lines()[0].get_xdata()
            overlap_ratio = cal_or_by_gaussian_kde(
                pos_cos_similarity, neg_cos_similarity, xlim, bins=len(x_data)
                )

        # TODO
        ax.text(
                0.03, 0.97, 
                # r"$\bf{OR=0.3}$" + "\n(95% CI: 0.2-0.5, p<0.001)", 
                f"OR={overlap_ratio:.2f}", 
                transform=ax.transAxes,
                fontsize=10,
                fontfamily='Arial',
                linespacing=1.4,
                verticalalignment='top',
                horizontalalignment='left',
                bbox=dict(
                    facecolor='white', 
                    alpha=0.85,
                    edgecolor='#5c5c5c',
                    linewidth=0.5,
                    boxstyle='round,pad=0.4'
                )
        )
        ax.spines['bottom'].set_linewidth(3)
        ax.spines['bottom'].set_color('grey')
        ax.spines['top'].set_linewidth(0)
        ax.spines['top'].set_color('grey')
        ax.spines['left'].set_linewidth(0)
        ax.spines['left'].set_color('grey')
        ax.spines['right'].set_linewidth(0)
        ax.spines['right'].set_color('grey')

        if plot_title:
            plt.title(model_name_mapping[model_name])
        plt.savefig(os.path.join(save_path, model_name + '_cos.png'), transparent=True, dpi=600, bbox_inches='tight')
        plt.savefig(os.path.join(save_path, model_name + '_cos.svg'), bbox_inches='tight')
        plt.savefig(os.path.join(save_path, model_name + '_cos.pdf'), bbox_inches='tight')
        plt.close()

    # mean_pos, mean_neg = round(np.mean(pos_cos_similarity), 3), round(np.mean(neg_cos_similarity), 3)
    # median_pos, median_neg = round(np.median(pos_cos_similarity), 3), round(np.median(neg_cos_similarity), 3)
    delta_mean = np.mean(pos_cos_similarity) - np.mean(neg_cos_similarity)
    delta_median = np.median(pos_cos_similarity) - np.median(neg_cos_similarity)
    
    return pos_cos_similarity, neg_cos_similarity, delta_mean, delta_median, round(overlap_ratio, 3)



def emb_rfam_cos_sim_cal(features_pkl, 
                         save_path, model_name,
                         n_samples=100000, plot=True, xlim=None, xticks=None, gauss_kde=False, plot_title=False):
    with open(features_pkl, 'rb') as fr:
        dataset = pickle.load(fr)

    keys = dataset['keys']
    seq_repr = dataset['seq_repr']

    key_to_seq_repr = {keys[n]: seq_repr[n] for n in range(len(keys))}
    rf_to_key = {}
    for key in keys:
        rf = key.split('_')[0]
        if rf in rf_to_key:
            rf_to_key[rf].append(key)
        else:
            rf_to_key[rf] = [key]
    rfs = list(rf_to_key.keys())

    pos_samples = np.zeros((n_samples, 2, seq_repr[0].size))
    neg_samples = np.zeros((n_samples, 2, seq_repr[0].size))

    idx = 0
    while True:
        sample_rf = random.choice(rfs)
        if len(rf_to_key[sample_rf]) > 2:
            key_a, key_b = random.sample(rf_to_key[sample_rf], 2)
            pos_samples[idx, 0, :] = key_to_seq_repr[key_a]
            pos_samples[idx, 1, :] = key_to_seq_repr[key_b]

            rf_a, rf_b = random.sample(rfs, 2)
            key_a, key_b = random.choice(rf_to_key[rf_a]), random.choice(rf_to_key[rf_b])
            neg_samples[idx, 0, :] = key_to_seq_repr[key_a]
            neg_samples[idx, 1, :] = key_to_seq_repr[key_b]
            
            idx += 1
        if idx == n_samples:
            break

    pos_cos_similarity = cosine_similarity(pos_samples[:, 0, :], pos_samples[:, 1, :])
    neg_cos_similarity = cosine_similarity(neg_samples[:, 0, :], neg_samples[:, 1, :])

    if plot:
        df = pd.DataFrame({'neg': neg_cos_similarity, 'pos': pos_cos_similarity})

        fig = plt.figure(figsize=(4, 2))
        ax = fig.add_subplot(111)
        bw_adjust = 1
        sns.kdeplot(data=df['pos'], color="grey", linestyle="-", legend=False, alpha=0.4,
                    bw_adjust=bw_adjust, linewidth=1, fill=False)
        sns.kdeplot(data=df['pos'], ax=ax, color=palette[1], fill=True, legend=False, alpha=0.8,  
                    bw_adjust=bw_adjust,
                    palette=palette)   

        sns.kdeplot(data=df['neg'], color="grey", linestyle="-", legend=False, alpha=0.4,
                    bw_adjust=bw_adjust, linewidth=1, fill=False)
        sns.kdeplot(data=df['neg'], ax=ax, color=palette[0], legend=False, alpha=0.8,  
                    bw_adjust=bw_adjust, fill=True,
                    palette=palette)  

        #ax.set_xlabel('Cos. Sim.')
        ax.set_xlabel('')
        ax.set_ylabel('')
        if xlim is not None:
            ax.set_xlim(xlim)
        ax.set_yticks([])

        if xticks is not None:
            ax.set_xticks(xticks)
            ax.set_xticklabels(xticks, fontsize=20)

        overlap_ratio = -1
        if gauss_kde:
            if xlim is None:
                xlim = ax.get_xlim()
            x_data = plt.gca().get_lines()[0].get_xdata()
            overlap_ratio = cal_or_by_gaussian_kde(
                pos_cos_similarity, neg_cos_similarity, xlim, bins=len(x_data)
                )
        # TODO
        ax.text(
                0.03, 0.97, 
                # r"$\bf{OR=0.3}$" + "\n(95% CI: 0.2-0.5, p<0.001)", 
                f"OR={overlap_ratio:.2f}", 
                transform=ax.transAxes,
                fontsize=10,
                fontfamily='Arial',
                linespacing=1.4,
                verticalalignment='top',
                horizontalalignment='left',
                bbox=dict(
                    facecolor='white', 
                    alpha=0.85,
                    edgecolor='#5c5c5c',
                    linewidth=0.5,
                    boxstyle='round,pad=0.4'
                )
        )

        ax.spines['bottom'].set_linewidth(3)
        ax.spines['bottom'].set_color('grey')
        ax.spines['top'].set_linewidth(0)
        ax.spines['top'].set_color('grey')
        ax.spines['left'].set_linewidth(0)
        ax.spines['left'].set_color('grey')
        ax.spines['right'].set_linewidth(0)
        ax.spines['right'].set_color('grey')
        fig.subplots_adjust(bottom=0.2, top=0.95, left=0.05, right=0.95)
        if plot_title:
            plt.title(model_name_mapping[model_name])
        plt.savefig(os.path.join(save_path, model_name + '_cos.png'), transparent=True, dpi=600, bbox_inches='tight')
        plt.savefig(os.path.join(save_path, model_name + '_cos.svg'), bbox_inches='tight')
        plt.savefig(os.path.join(save_path, model_name + '_cos.pdf'), bbox_inches='tight')
        plt.close()

    # mean_pos, mean_neg = round(np.mean(pos_cos_similarity), 3), round(np.mean(neg_cos_similarity), 3)
    # median_pos, median_neg = round(np.median(pos_cos_similarity), 3), round(np.median(neg_cos_similarity), 3)
    delta_mean = np.mean(pos_cos_similarity) - np.mean(neg_cos_similarity)
    delta_median = np.median(pos_cos_similarity) - np.median(neg_cos_similarity)

    return pos_cos_similarity, neg_cos_similarity, delta_mean, delta_median, round(overlap_ratio, 3)

def metricCal(
    test_data,
    step: float = 0.01,
    eps: float = 1e-11
):
    T = np.arange(0, 1 + step, step)[None, :]  # threshold value [1,1001]

    predictions = test_data["pred"]
    targets = test_data["true"]

    # Check sizes
    if predictions.size != targets.size:
        raise ValueError(
            f"Size mismatch. Received predictions of size {predictions.size}, "
            f"targets of size {targets.size}"
        )

    targets = targets.reshape(-1, 1)    #[n,1]
    predictions = predictions.reshape(-1, 1)   #[n,1]

    outputs_T = np.greater_equal(predictions, T)

    # should .astype(float)
    tp = np.sum(np.logical_and(outputs_T, targets).astype(float), axis=0)
    tn = np.sum(np.logical_and(np.logical_not(outputs_T), np.logical_not(targets)).astype(float), axis=0)
    fp = np.sum(np.logical_and(outputs_T, np.logical_not(targets)).astype(float), axis=0)
    fn = np.sum(np.logical_and(np.logical_not(outputs_T), targets).astype(float), axis=0)

    prec = tp / (tp + fp + eps)  # precision
    recall = tp / (tp + fn + eps)  # recall
    sens = tp / (tp + fn + eps) # senstivity
    spec = tn / (tn + fp + eps)  # spec
    TPR = tp / (tp + fn + eps)
    FPR = fp / (tn + fp + eps)
    prec[np.isnan(prec)] = 0
    F1 = 2 * ((prec * sens) / (prec + sens + eps)) # or (2 * tp + eps) / (2 * tp + fp + fn + eps)
    MCC = ((tp * tn) - (fp * fn)) / (np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) + eps)

    max_f1_pos = np.nanargmax(F1)
    threshold = T[0, max_f1_pos]
    Recall = recall[max_f1_pos]
    F1 = np.nanmax(F1)  # F1 Score
    MCC = MCC[max_f1_pos]
    PR1 = auc(recall, prec)
    PR = np.trapz(y=recall, x=prec)  # average precision-recall value
    AUC = np.trapz(y=sens, x=spec)

    return {"F1":F1, "MCC":MCC, "PR1":PR1, "PR":PR,"AUC":AUC,"Recall":Recall, "thresh":threshold}
