import os
from collections import Counter

from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from umap import UMAP
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt

from src.RNAStruBert.infer import RNAStruBert_infer
from src.RNAStruBert.data import preprocess_and_load_dataset, get_mlm_tokenizer


def get_palette(name='Set1'):
    ''' 
        qualitative: 区分
        sequential: 渐变
        diverging: 
    '''
    if name in {'Paired', 'Set2', 'Set3', 'Accent', 'Dark2', 'Pastel1', 'Pastel2', 'hls'}:
        return sns.color_palette(name)
    elif name == 'deep_blue_yellow_other':
        colors = ['#80b1d3', '#ffffb3', '#8dd3c7', '#f2d1ac', '#bebada'] # blue, yellow, green, orange, purple
        return sns.color_palette(colors)
    elif name == 'deep_blue_brown_other':
        colors = ['#80b1d3', '#e4d3bb', '#8dd3c7', '#ebeba4', '#bebada', '#fdb462'] # blue, yellow, green, orange, purple
        return sns.color_palette(colors)
    elif name == 'blue_brown_other':
        colors = ['#b3d0e5', '#efe5d6', '#bbe5dd', '#ffffb3', '#bebada'] # blue, yellow, green, orange, purple
        return sns.color_palette(colors)
    elif name == 'blue_green':
        colors = ['#1e90c1', '#3fb2c4', '#81cbbf', '#c0e1bd'] # blue, yellow, green, orange, purple
        return sns.color_palette(colors)
    elif name == 'tSNE':
        colors = ['#62c3a2', '#a2a783', '#e5946d', '#db9085', '#a99eb2', '#9e9ec2', '#c097c1', '#e890bc', '#cab58e', '#afd55b', '#c3d744', '#ecd632', '#f7d442', '#ebcd77', '#e0c298', '#c5bcab', '#b3b4b2', '#ccccff', '#ab8fff']
        return sns.color_palette(colors)
    else:
        raise Exception(f'Unknown palette: {name}')


def load_df_dataset(data_path, max_length=514, phase='test'):
    '''
        data_path: data_dir for `load_dataset`
    '''
    dest = os.path.join(os.path.dirname(data_path), f'RNAcentral_{phase}.csv')
    if not os.path.exists(dest):
        tag = 'mlm'
        mlm_structure=True
        seed = 42
        tokenizer = get_mlm_tokenizer(max_length=max_length)
        dataset = preprocess_and_load_dataset(data_path, tokenizer, tag, with_structure=mlm_structure)
        split_dataset = dataset
        if 'test' not in split_dataset:
            if 'validate' in dataset:
                split_dataset['test'] = split_dataset['validate']
            else:
                split_dataset = dataset['train'].train_test_split(test_size=0.05, seed=seed)
        print(split_dataset)
        select_dataset = split_dataset[phase].select_columns(['name', 'seq', 'connects'])
        select_dataset.to_csv(dest)
    return pd.read_csv(dest, index_col=False)


def visualize(dest, data, labels, label_idx=None, title=None, decomp='tsne', random_state=42, alpha=0.8, max_class_num=15, max_vis_num=20000, palette_name='tSNE', ncol=None, select_labels=None, perplexity=30):
    '''
        data: shape: N x M, number
        labels: shape: N
    '''
    def get_label_mapping(labels):
        label_mapping = {}
        for lbl in labels:
            if lbl not in label_mapping:
                label_mapping[lbl] = len(label_mapping)+1
        return label_mapping

    ## select labels
    data = np.array(data)
    labels = np.array(labels)

    if select_labels is not None:
        idx_df = pd.DataFrame({'idx': range(len(data)), 'label': labels})
        print('select labels', len(labels))
        print(Counter(labels.tolist()))
        print(select_labels)
        idx_df = idx_df.groupby('label', group_keys=False).apply(lambda x: x.sample(n=select_labels[x.name], random_state=42))
        data = np.array([data[idx] for idx in idx_df['idx']])
        labels = np.array([label for label in idx_df['label']])
    print('input data', data.shape)

    ## label map: str -> int
    origin_labels = labels[:]
    label_mapping = get_label_mapping(labels)
    labels = np.array([label_mapping[lbl] for lbl in origin_labels])
    
    # remove noisy points
    not_noisy = labels != -1
    data = data[not_noisy]
    labels = labels[not_noisy]

    # reduce points
    if len(labels)>max_vis_num:
        chosen = np.random.choice(len(labels), max_vis_num, replace=False, p=None)
        data = data[chosen]
        labels = labels[chosen]

    classes = np.unique(labels).tolist()
    classes = sorted(classes, key=lambda cls: np.sum(labels==cls), reverse=True)
    num_classes = len(classes)

    # reduce class
    if num_classes > max_class_num:
        classes = sorted(classes, key=lambda cls: np.sum(labels==cls), reverse=True)[:max_class_num]
        class_set = set(classes)
        num_classes = len(classes)
        new_data = []
        new_label = []
        for d, lab in zip(data, labels):
            if lab in class_set:
                new_data.append(d)
                new_label.append(lab)
        data = np.array(new_data)
        labels = np.array(new_label)

    # color
    palette = np.array(get_palette(palette_name))
    if label_idx is None:
        label_idx = {cls: i for i, cls in enumerate(sorted(classes))}
    idx_label = {v: k for k, v in label_idx.items()}
    colors = np.array([palette[label_idx[lab]] for lab in labels])

    print('before tsne data', data.shape)
    DECOMP = None
    if decomp.lower()=='tsne':
        DECOMP = TSNE(n_components=2, init='pca', random_state=random_state, perplexity=perplexity)
    elif decomp.lower()=='pca':
        DECOMP = PCA(random_state=random_state)
    elif decomp.lower()=='umap':
        DECOMP = UMAP(n_components=2, random_state=random_state)
    else:
        raise NotImplementedError

    Xt = DECOMP.fit_transform(data)

    ### matplotlib, seaborn, config
    # plt.style.use("dark_background")
    # cmap = plt.cm.get_cmap('rainbow', 10)
    fontsize = 8
    PAGE_WIDTH = 18.0 # cm
    CM_PER_INCH = 2.54
    plt.rcParams['font.family'] = 'Arial'
    figsize = (PAGE_WIDTH/2/CM_PER_INCH*2, PAGE_WIDTH/2/CM_PER_INCH*2)

    fig, ax = plt.subplots(figsize=figsize)
    sns.set_theme(context='paper', style='white')

    handles = []
    hdl_labels = []
    idx_label = {v: k for k, v in label_mapping.items()}
    for cls in classes:
        indexes = labels == cls
        hdl = ax.scatter(Xt[indexes, 0], Xt[indexes, 1], c=colors[indexes], alpha=alpha, label=str(cls), edgecolors='none', s=15)
        handles.append(hdl)
        hdl_labels.append(idx_label[cls])
    # hdl = plt.scatter(Xt[:, 0], Xt[:, 1], c=colors, alpha=alpha)
    # plt.legend(handles=hdl.legend_elements()[0], labels=origin_labels, ncol=num_classes//3, loc='best')
    if ncol is None:
        ncol = num_classes//5
    # bbox_to_anchor: 1.05 reprs right (0 reprs left), 1 reprs top (0 reprs bottom) # bbox_to_anchor=(1.05, 0), 
    # bbox_to_anchor=(0, -0.2): left, the below of bottom,  loc relative to bbox, upper left
    # plt.legend(handles=handles, labels=hdl_labels, ncol=1, loc='upper left', bbox_to_anchor=(1.2, 1), fontsize=fontsize, frameon=False) # TODO
    plt.legend(handles=handles, labels=hdl_labels, ncol=ncol, loc='best', fontsize=fontsize, frameon=False)
    for i in range(num_classes):
        xtext, ytext = np.median(Xt[labels == idx_label[i], :], axis=0)
        txt = ax.text(xtext, ytext, idx_label[i], fontsize=fontsize)
    # cbar = plt.colorbar() # ticks=range(10)
    # cbar.set_label(label='value', fontdict=font)
    plt.xticks([])
    plt.yticks([])
    for spine in ['right', 'left', 'top', 'bottom']:
        ax.spines[spine].set_visible(False)
    plt.tight_layout()

    plt.savefig(dest, bbox_inches='tight', pad_inches=0.1, dpi=600, format='pdf')
    plt.savefig(dest[:-4]+'.png', bbox_inches='tight', pad_inches=0.1, dpi=600, format='png')
    plt.show()
    plt.close()


def get_npy_data(npy_path, model, seq, device):
    if not (os.path.exists(npy_path) and os.path.getsize(npy_path)):
        feat = model.extract_feature(seq=seq, return_all=False)
        data = feat.detach().cpu().numpy()
        np.save(npy_path, data)
    else:
        data = np.load(npy_path)
    return data


def get_data_and_labels(df, dest_npy, cls_flag, model_path, device, max_length=514, cache_flag='', rerun=False):
    '''
        cls_flag: cls, all
    '''
    if cache_flag and cache_flag[0]!='_':
        cache_flag = '_'+cache_flag
    npy_dir = os.path.join(dest_npy, os.path.basename(model_path))
    os.makedirs(npy_dir, exist_ok=True)
    model = RNAStruBert_infer(from_pretrained=model_path, max_length=max_length, device=device)
    cache_pre = os.path.join(dest_npy, f'{os.path.basename(npy_dir)}')
    cache_data_path = cache_pre + f'_data_{cls_flag}{cache_flag}.npy'
    cache_label_path = cache_pre + f'_labels{cache_flag}.npy'
    if not os.path.exists(cache_data_path) or rerun:
        data, labels = [], []
        for seq_name, seq in tqdm(zip(df['name'], df['seq'])):
            # labels
            vals = df.loc[df['name'] == seq_name, 'family'].values
            if len(vals)<1:
                print('No family found, ignored:', seq_name)
                continue
            labels.append(vals[0])

            # data
            npy_path = os.path.join(npy_dir, seq_name+'.npy')
            arr = get_npy_data(npy_path, model, seq, device)

            # data pad
            arr = np.pad(arr, ((0, max_length-arr.shape[0]), (0, 0)), constant_values=0)
            data.append(arr)
        np.save(cache_label_path, np.array(labels))

        data = np.array(data)
        N = data.shape[0]
        np.save(cache_pre + f'_data_cls{cache_flag}.npy', data[:,0,:].reshape(N, -1))
        np.save(cache_pre + f'_data_all{cache_flag}.npy', data.reshape(N, -1))
    data = np.load(cache_data_path)
    labels = np.load(cache_label_path)
    return data, labels


def select_samples(dic, num_sample=50000):
    if sum(dic.values())<num_sample:
        return dic
    inc_num = 100
    num_class = len(dic)
    base_num = ((2*num_sample)/num_class-(num_class-1)*100)//2
    print(dic)

    max_key = max(dic, key=lambda k: dic[k])
    new_dic = {}
    ## stage 1
    for i, k in enumerate(sorted(dic, key=lambda k:dic[k])):
        v = dic[k]
        new_dic[k] = int(min(base_num + i*inc_num, v))
    ## stage 2
    cur_num = sum(new_dic.values())
    new_dic[max_key] += num_sample - sum(new_dic.values())
    print(new_dic)
    return new_dic


if __name__ == '__main__':
    phase = 'train'
    model_paths = [
                   'checkpoint-2516835',
                   'nostru_checkpoint-1174460',
                   '/public/share/heqinzhu_share/RNAStruBert/checkpoint-1174460',
                    '/public/share/heqinzhu_share/RNAStruBert/checkpoint-1761690',
                  ]
    data_path = '../RNAcentral/RNAcentral_BPfold_SS_disk'
    device = 'cuda:2' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    seq_cls_csv = '../RNAcentral/rnacentral_active.csv'

    dest_npy = f'RNAcentral_{phase}_npy'
    dest_fig = 'tSNE_results'
    os.makedirs(dest_npy, exist_ok=True)
    os.makedirs(dest_fig, exist_ok=True)

    # cls_flags = ['cls', 'all']
    cls_flags = ['cls']
    # decomp_methods = ['tSNE', 'PCA', 'UMAP']
    decomp_methods = ['tSNE', 'UMAP']

    MAX_CLASS_NUM = 19
    MAX_VIS_NUM = 30000 # 500, 600, ..., 2500

    select_fam_num = {
                      'lncRNA': 478,
                      'rRNA': 6017,
                      'tRNA': 1591,
                      'RNase_P_RNA': 52,
                      'pre_miRNA': 224,
                      'sRNA': 144,
                      'snRNA': 109,
                      'SRP_RNA': 73,
                      'snoRNA': 127,
                      'piRNA': 104,
                      'hammerhead_ribozyme': 55,
                      'miRNA': 48,
                      'tmRNA': 84,
                    }

    ## testset
    select_fam_num = {
        'rRNA': 648564,
        'tRNA': 163187,
        'lncRNA': 52632,
        'pre_miRNA': 22155,
        # 'sRNA': 15473,
        'snRNA': 14123,
        'snoRNA': 13473,
        'piRNA': 10960,
        'tmRNA': 7687,
        'SRP_RNA': 6696,
        'hammerhead_ribozyme': 6675,
        'miRNA': 5812,
        'RNase_P_RNA': 5174,
        'siRNA': 2196,
        'ribozyme': 1981,
        'antisense_RNA': 1639,
        'Y_RNA': 1030,
        'precursor_RNA': 346,
        # 'RNase_MRP_RNA': 119,
    }


    ## mannual
    select_fam_num = {
        'rRNA': 2000,
        'tRNA': 2000,
        'lncRNA': 1500,
        'pre_miRNA': 1200,
        # 'sRNA': 1200,
        'snRNA': 1000,
        'snoRNA': 1000,
        'piRNA': 1000,
        'tmRNA': 1000,
        'SRP_RNA': 1000,
        'hammerhead_ribozyme': 1000,
        'miRNA': 1000,
        'RNase_P_RNA': 1000,
        'siRNA': 1000,
        'ribozyme': 1000,
        'antisense_RNA': 1000,
        'Y_RNA': 1000,
        'precursor_RNA': 346,
        # 'RNase_MRP_RNA': 119,
    }

    select_fam_num = {
        'rRNA': 12320837,
        'tRNA': 3102776,
        'lncRNA': 1003112,
        'pre_miRNA': 422695,
        # 'sRNA': 293843,
        'snRNA': 271153,
        'snoRNA': 253329,
        'piRNA': 208689,
        'tmRNA': 147258,
        'hammerhead_ribozyme': 126753,
        'SRP_RNA': 124310,
        'miRNA': 110759,
        'RNase_P_RNA': 96689,
        'siRNA': 41175,
        'antisense_RNA': 30643,
        'Y_RNA': 19410,
        # 'precursor_RNA': 6511,
        'RNase_MRP_RNA': 2283,
        'vault_RNA': 1572,
        'guide_RNA': 1474,
        'telomerase_RNA': 859,
     }


    select_fam_num = select_samples(select_fam_num, num_sample=MAX_VIS_NUM)
    
    palette_name = 'tSNE'

    max_length = 514
    df = load_df_dataset(data_path, max_length, phase=phase)
    rerun = True

    # sample seqs
    # df = df.sample(n=MAX_VIS_NUM, random_state=42)

    ## sample level-wisely
    seq_cls_df = pd.read_csv(seq_cls_csv, index_col=False)
    seq_cls_df = seq_cls_df.rename(columns={'id': 'name'})
    df = pd.merge(df, seq_cls_df, on='name', how='left') # add fam col
    df = df[df['family'].isin(select_fam_num)] # filter vis fam
    # must filter before groupby
    df = df.groupby('family', group_keys=False).apply(lambda x: x.sample(n=select_fam_num[x.name], random_state=42)) # sample through fam
    print('sampled df', len(df))

    cache_flag = 'select'
    for model_path in model_paths:
        print('model_path')
        for cls_flag in cls_flags:
            print('cls_flag', cls_flag)
            for decomp in decomp_methods:
                print('decomp', decomp)
                data, labels = get_data_and_labels(df, dest_npy, cls_flag, model_path, device, max_length=514, cache_flag=cache_flag, rerun=rerun)
                for perplexity in [30, 50]:
                    fig_path = os.path.join(dest_fig, f'select_{decomp}_{os.path.basename(model_path)}_{cls_flag}.pdf')
                    if decomp == 'tSNE':
                        fig_path = fig_path[:-4] + f'_perp{perplexity}' + '.pdf'
                    visualize(fig_path, data, labels, label_idx=None, decomp=decomp, random_state=42, alpha=0.8, max_class_num=MAX_CLASS_NUM, max_vis_num=MAX_VIS_NUM, ncol=5, select_labels=None, perplexity=perplexity, palette_name=palette_name)
                    if decomp != 'tSNE':
                        break
