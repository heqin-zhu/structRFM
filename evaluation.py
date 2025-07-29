import os
import argparse

import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from transformers import AutoTokenizer
from tqdm import tqdm
import pandas as pd
from scipy.stats import spearmanr

from src.SgRFM.infer import SgRFM_infer, save_seqs_to_csv


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
    data_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir)] if os.path.isdir(data_dir) else [data_dir]
    for data_path in data_paths:
        df = pd.read_csv(data_path)
        print(data_path)
        print("Data Num", len(df))
        seqs = df['seq'].tolist()
        names = df['name'].tolist() if 'name' in df.columns else list(range(1, 1+len(seqs)))
        data = {'name': names}
        for model_path in model_paths:
            model_mlm = SgRFM_infer(from_pretrained=model_path, max_length=514, device=device)
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model_name = os.path.basename(model_path)
            # lls = cal_likelihood(seqs, model_mlm)
            ppls = cal_perplexity(seqs, model_mlm)
            data[model_name] = ppls

            # df = df[~df['label'].isna()]
            # labels = df['label'].tolist()
            # corr, _ = spearmanr(lls, labels)
            # print(f"correlation: {corr:.4f}")
        df = pd.DataFrame(data)
        df.to_csv(os.path.join(dest_dir, os.path.basename(data_path).replace(".csv", f"_perplexity.csv")), index=False)

        averages = df[[col for col in df.columns if col.startswith('checkpoint')]].mean()
        for column, avg in averages.items():
            print(f"mean_ppl={avg:.4f}, {column}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--dest_dir", type=str)
    parser.add_argument("--model_path", type=str)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    model_paths = []
    step_interval = 16_778
    model_dir1 = '/heqinzhu/runs/mlm_768x12_lr0.0001'
    model_dir2 = '/heqinzhu/runs/mlm_768x12_lr0.0001_stru'
    # for epoch in [1, 2, 3, 4, 5] + np.arange(5.5, 15.5, 0.5).tolist():
    for epoch in range(5, 16):
        model_dir = model_dir1 if epoch <= 10 else model_dir2
        step = round(step_interval * 10 * epoch)
        model_path = os.path.join(model_dir, f'checkpoint-{step}')
        if os.path.exists(model_path):
            model_paths.append(model_path)
    print(model_paths)
    evaluate_all(model_paths, args.data_dir, args.dest_dir)
