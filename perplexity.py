import argparse

import torch
from torch.nn import CrossEntropyLoss
from transformers import AutoTokenizer
from tqdm import tqdm
import pandas as pd
from scipy.stats import spearmanr

from src.NucleoLM.infer import RNALM_MLM, save_seqs_to_csv


@torch.no_grad()
def cal_score(seqs, model_mlm, device="cuda"):
    scores = []
    cal_loss = CrossEntropyLoss(reduction="mean")
    for seq in seqs:
        outputs, inputs = model_mlm.model_forward(seq, return_inputs=True)
        logits = outputs['logits']
        # logits: B x (1+len+1) x vocab_len; input_ids: B x (1+len+1)
        score = -cal_loss(logits[0][:-1, ...], inputs['input_ids'][0][1:]).item()
        scores.append(score)
    return scores


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--seq_col", type=str, default="seq")
    parser.add_argument("--label_col", type=str, default="label")
    parser.add_argument("--input_file", type=str, default="seq.csv")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # model init
    from_pretrained = '/heqinzhu/runs/mlm_768x12_lr0.0001/checkpoint-1124126'
    model_path = args.model_path or from_pretrained
    model_mlm = RNALM_MLM(from_pretrained=model_path, max_length=514)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    df = pd.read_csv(args.input_file)
    df = df[~df[args.label_col].isna()]
    print("Valid data:", len(df))
    seqs = df[args.seq_col].tolist()
    labels = df[args.label_col].tolist()
    scores = cal_score(seqs, model_mlm, device=device)
    model_name = model_path.split("/")[-1]
    df[f"{model_name}"] = scores
    corr, _ = spearmanr(scores, labels)
    print(f"Spearman correlation: {corr:.4f}")
    df.to_csv(args.input_file.replace(".csv", f"_{model_name}_score.csv"), index=False)
