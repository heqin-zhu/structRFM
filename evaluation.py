import os
import argparse

import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from transformers import AutoTokenizer
from tqdm import tqdm
import pandas as pd
from scipy.stats import spearmanr

from src.NucleoLM.infer import RNALM_MLM, save_seqs_to_csv


@torch.no_grad()
def cal_likelihood(seqs, model_mlm):
    '''
        higher, better
    '''
    lls = []
    CE_loss = CrossEntropyLoss(reduction="mean")
    for seq in tqdm(seqs):
        outputs, inputs = model_mlm.model_forward(seq, return_inputs=True)
        logits = outputs['logits']
        # logits: B x (1+len+1) x vocab_len; input_ids: B x (1+len+1)
        ce = CE_loss(logits[0][:-1, ...], inputs['input_ids'][0][1:]).item()
        lls.append(-ce)
    return lls


@torch.no_grad()
def cal_perplexity(seqs, model_mlm):
    '''
        lower, better
    '''
    ppls = []
    for seq in tqdm(seqs):
        # outputs = model(**inputs, labels=input_ids)
        outputs, inputs = model_mlm.model_forward(seq, return_inputs=True, is_cal_loss=True)
        loss = outputs.loss.item()
        ppls.append(torch.exp(torch.tensor(loss)).item())
    return ppls


def evaluate_all(model_paths, data_dir, dest_dir='.'):
    '''
        data_dir:
            - xxx.csv  # cols: name, seq
            - yyy.csv  # cols: name, seq

    '''
    os.makedirs(dest_dir, exist_ok=True)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    for f in os.listdir(data_dir):
        path = os.path.join(data_dir, f)
        df = pd.read_csv(path)
        print(path)
        print("Data Num", len(df))
        seqs = df['seq'].tolist()
        names = df['name'].tolist() if 'name' in df.columns else list(range(1, 1+len(seqs)))
        data = {'name': names}
        for model_path in model_paths:
            print(model_path, end=' ')
            model_mlm = RNALM_MLM(from_pretrained=model_path, max_length=514, device=device)
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model_name = os.path.basename(model_path)
            # lls = cal_likelihood(seqs, model_mlm)
            ppls = cal_perplexity(seqs, model_mlm)
            mean_ppl = np.mean(ppls)
            print('perplexity', mean_ppl)
            data[model_name] = ppls

            # df = df[~df['label'].isna()]
            # labels = df['label'].tolist()
            # corr, _ = spearmanr(lls, labels)
            # print(f"correlation: {corr:.4f}")
        pd.DataFrame(data).to_csv(os.path.join(dest_dir, os.path.basename(path).replace(".csv", f"_perplexity.csv")), index=False)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--dest_dir", type=str)
    parser.add_argument("--model_path", type=str)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # TODO args.model_path
    model_paths = []
    step_interval = 16_778
    for epoch in [1, 2, 3, 4, 5] + np.arange(5.5, 10.5, 0.5).tolist():
        step = round(step_interval * 10 * epoch)
        model_path = f'/heqinzhu/runs/mlm_768x12_lr0.0001/checkpoint-{step}'
        model_paths.append(model_path)

    model_dir = '/heqinzhu/runs/mlm_768x12_lr0.0001_stru'
    model_paths = [os.path.join(model_dir, f) for f in os.listdir(model_dir) if f.startswith('checkpoint')]

    evaluate_all(model_paths, args.data_dir, args.dest_dir)
