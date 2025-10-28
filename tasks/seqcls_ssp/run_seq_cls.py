import os
import argparse

import torch
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from RNAdata import SeqClsDataset
from utils import get_config, get_and_set_device
from utils import str2bool, str2list
from collators import SeqClsCollator
from losses import SeqClsLoss
from metrics import SeqClsMetrics
from trainers import SeqClsTrainer
from tokenizer import RNATokenizer
from seq_cls import RNABertForSeqCls, RNAFmForSeqCls, RNAMsmForSeqCls


import sys
sys.path.append('../..')
from src.structRFM.model import get_structRFM_for_cls, get_model_scale
from src.structRFM.data import get_mlm_tokenizer

# ========== Define constants
MODELS = ["RNABERT", "RNAMSM", "RNAFM", 'structRFM']
MAX_SEQ_LEN = {"RNABERT": 440,
               "RNAMSM": 512,
               "RNAFM": 512,
               "structRFM": 512,
              }

TASKS = ["nRC", "lncRNA_H", "lncRNA_M", 'IRES', 'lncRNA_H_uni']
LABEL2ID = {
    "nRC": {
        "5S_rRNA": 0,
        "5_8S_rRNA": 1,
        "tRNA": 2,
        "ribozyme": 3,
        "CD-box": 4,
        "Intron_gpI": 5,
        "Intron_gpII": 6,
        "riboswitch": 7,
        "IRES": 8,
        "HACA-box": 9,
        "scaRNA": 10,
        "leader": 11,
        "miRNA": 12
    },
    "lncRNA_H": {
        "lnc": 0,
        "pc": 1
    },
    "lncRNA_H_uni": {
        "lnc": 0,
        "pc": 1
    },
    "lncRNA_M": {
        "lnc": 0,
        "pc": 1
    },
    "IRES": {
        "negative": 0,
        "positive": 1
    },
}


def get_args():
    # ========== Configuration
    parser = argparse.ArgumentParser(
        'Implementation of RNA sequence classification.')
    # model args
    parser.add_argument('--model_name', type=str, default="structRFM", choices=MODELS)
    parser.add_argument('--model_scale', type=str, choices=['base', 'large'], default='base')
    parser.add_argument('--vocab_path', type=str, default="./vocabs/")
    parser.add_argument('--LM_path', type=str)
    parser.add_argument('--config_path', type=str,default="./configs/")
    parser.add_argument('--dataset_dir', type=str, default="./data/seq_cls")
    parser.add_argument('--dataset', type=str, default="nRC", choices=TASKS)
    parser.add_argument('--replace_T', type=bool, default=True)
    parser.add_argument('--replace_U', type=bool, default=False)

    parser.add_argument('--device', type=str, help='cpu, cuda:0')
    parser.add_argument('--max_seq_len', type=int, default=0)
    parser.add_argument('--dataloader_num_workers', type=int, default=0)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--train', type=str2bool, default=True)
    parser.add_argument('--disable_tqdm', type=str2bool,
                        default=False, help='Disable tqdm display if true.')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='The number of samples used per step & per device.')
    parser.add_argument('--freeze_base', action='store_true', help='freeze base language model')
    parser.add_argument('--use_mean_feature', action='store_true')
    parser.add_argument('--use_automodelforseqcls', action='store_true')
    parser.add_argument('--num_train_epochs', type=int, default=60,
                        help='The number of epoch for training.')
    parser.add_argument('--metrics', type=str2list,
                        default="F1s,Precision,Recall,Accuracy,Mcc",)

    parser.add_argument('--logging_steps', type=int, default=1000,
                        help='Update visualdl logs every logging_steps.')
    # save checkpoint
    parser.add_argument('--output_dir', type=str)
    # parser.add_argument('--save_strategy', type=str, default='epoch')
    # parser.add_argument('--evaluation_strategy', type=str, default='epoch')
    # parser.add_argument('--load_best_model_at_end', type=str2bool, default=True)
    # parser.add_argument('--metric_for_best_model', type=str, default='F1s')
    # parser.add_argument('--greater_is_better', type=str2bool, default=True)
    args = parser.parse_args()
    return args


def freeze(model):
    for name, para in model.named_parameters():
        para.requires_grad = False


if __name__ == "__main__":
    args = get_args()
    # ========== post process
    if args.max_seq_len == 0:
        args.max_seq_len = MAX_SEQ_LEN[args.model_name]
        if args.dataset == 'lncRNA_H':
            args.max_seq_len = 3000
        elif args.dataset == 'lncRNA_M':
            args.max_seq_len = 6000

    # ========== args check
    assert args.replace_T ^ args.replace_U, "Only replace T or U."

    # ========== Build tokenizer, model, criterion

    if args.model_name == "RNABERT":
        tokenizer = RNATokenizer(args.vocab_path + "{}.txt".format(args.model_name))
        from RNABERT.rnabert import BertModel
        model_config = get_config(
            args.config_path + "{}.json".format(args.model_name))
        model = BertModel(model_config)
        if args.freeze_base:
            freeze(model)
        model = RNABertForSeqCls(model)
        model._load_pretrained_bert(args.LM_path)
    elif args.model_name == "RNAMSM":
        tokenizer = RNATokenizer(args.vocab_path + "{}.txt".format(args.model_name))
        from RNAMSM.model import MSATransformer
        model_config = get_config(
            args.config_path + "{}.json".format(args.model_name))
        model = MSATransformer(**model_config)
        if args.freeze_base:
            freeze(model)
        model = RNAMsmForSeqCls(model)
        model._load_pretrained_bert(args.LM_path)
    elif args.model_name == "RNAFM":
        tokenizer = RNATokenizer(args.vocab_path + "{}.txt".format(args.model_name))
        import RNAFM.fm as fm
        model, alphabet = fm.pretrained.rna_fm_t12()
        if args.freeze_base:
            freeze(model)
        model = RNAFmForSeqCls(model)
    elif args.model_name == "structRFM":
        from_pretrained = args.LM_path
        model_paras = get_model_scale(args.model_scale)
        if args.max_seq_len+2>514:
            tokenizer = get_mlm_tokenizer(max_length=args.max_seq_len+2)
            model = get_structRFM_for_cls(num_class=len(LABEL2ID[args.dataset]), from_pretrained=from_pretrained, tokenizer=tokenizer, pretrained_length=514, freeze_base=args.freeze_base, use_mean_feature=args.use_mean_feature, **model_paras)
        else:
            if args.use_automodelforseqcls:
                model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=from_pretrained, num_labels=len(LABEL2ID[args.dataset]))
                if args.freeze_base:
                    freeze(model.bert.encoder)
            else:
                model = get_structRFM_for_cls(num_class=len(LABEL2ID[args.dataset]), from_pretrained=from_pretrained, freeze_base=args.freeze_base, use_mean_feature=args.use_mean_feature, **model_paras)
            tokenizer = AutoTokenizer.from_pretrained(from_pretrained)
    else:
        raise ValueError("Unknown model name: {}".format(args.model_name))

    args.device = torch.device(args.device)
    if torch.cuda.device_count()>1:
        model = torch.nn.DataParallel(model)
    model.to(args.device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print(f"train/total parameter: {trainable_params}/{total_params}")

    _loss_fn = SeqClsLoss().to(args.device)
    if args.output_dir is None:
        if args.model_name=='structRFM' and args.LM_path and os.path.exists(args.LM_path):
            run_name = os.path.basename(args.LM_path)
            run_name = run_name[:run_name.rfind('.')]
        else:
            run_name = args.model_name
        args.output_dir = os.path.join('outputs', run_name)
    os.makedirs(args.output_dir, exist_ok=True)
        

    # ========== Prepare data
    dataset_train = SeqClsDataset(fasta_dir=args.dataset_dir, prefix=args.dataset, tokenizer=tokenizer)
    dataset_eval = SeqClsDataset(fasta_dir=args.dataset_dir, prefix=args.dataset, tokenizer=tokenizer, train=False)
    # 6230 : 2600
    print('dataset train:test', len(dataset_train), len(dataset_eval)) 

    # ========== Create the data collator
    _collate_fn = SeqClsCollator(
        max_seq_len=args.max_seq_len, tokenizer=tokenizer,
        label2id=LABEL2ID[args.dataset], replace_T=args.replace_T, replace_U=args.replace_U, is_structRFM=args.model_name.startswith('structRFM'))

    # ========== Create the learning_rate scheduler (if need) and optimizer
    optimizer = AdamW(params=[para for para in model.parameters() if para.requires_grad], lr=args.learning_rate)

    # ========== Create the metrics
    _metric = SeqClsMetrics(metrics=args.metrics)

    # ========== Create the trainer
    seq_cls_trainer = SeqClsTrainer(
        args=args,
        tokenizer=tokenizer,
        model=model,
        train_dataset=dataset_train,
        eval_dataset=dataset_eval,
        data_collator=_collate_fn,
        loss_fn=_loss_fn,
        optimizer=optimizer,
        compute_metrics=_metric,
    )
    if args.train:
        for i_epoch in range(args.num_train_epochs):
            print("Epoch: {}".format(i_epoch))
            seq_cls_trainer.train(i_epoch)
            seq_cls_trainer.eval(i_epoch)
