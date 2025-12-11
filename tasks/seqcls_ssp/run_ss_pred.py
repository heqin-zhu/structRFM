import os
import os.path as osp
import argparse

import torch
from torch.optim import Adam, AdamW
from transformers import AutoTokenizer

from utils import get_config, get_and_set_device
from utils import str2bool, str2list
from tokenizer import RNATokenizer
import param_turner2004
from ss_pred import (
    RNABertForSsp,
    MixedFold,
    RNAFmForSsp,
    RNAMsmForSsp,
    structRFMForSsp,
)
from metrics import SspMetrics
from collators import SspCollator
from losses import StructuredLoss
from RNAdata import BPseqDataset
from trainers import SspTrainer

import sys
from structRFM.model import get_structRFM, get_model_scale
from structRFM.data import get_mlm_tokenizer

# ========== Define constants
MODELS = ["RNABERT", "RNAMSM", "RNAFM", "structRFM"]
TASKS = ["RNAStrAlign", "bpRNA1m"]
MAX_SEQ_LEN = {"RNABERT": 440,
               "RNAMSM": 1024,
               "RNAFM": 1024,
               "structRFM": 514}
EMBED_DIMS = {"RNABERT": 120,
              "RNAMSM": 768,
              "RNAFM": 640,
              "structRFM": 768}
DATASETS = {
    "RNAStrAlign": ("RNAStrAlign600.lst", "archiveII600.lst"),
    "bpRNA1m": ("TR0.lst", "TS0.lst"),
}

def get_args():
    # ========== Configuration
    parser = argparse.ArgumentParser(
        description='RNA secondary structure prediction using deep learning with thermodynamic integrations', add_help=True)

    # model args
    parser.add_argument('--LM_path', type=str)
    parser.add_argument('--checkpoint_path', type=str)
    parser.add_argument('--model_name', type=str, default="structRFM", choices=MODELS)
    parser.add_argument('--model_scale', type=str, choices=['base', 'large'], default='base')
    parser.add_argument('--vocab_path', type=str, default="./vocabs/")
    parser.add_argument('--config_path', type=str, default="./configs/")

    parser.add_argument('--train', type=str2bool, default=True)
    parser.add_argument('--max_seq_len', type=int, default=0)
    # data args
    parser.add_argument('--task_name', type=str, default="RNAStrAlign",
                        choices=TASKS, help='Task name of training data.')
    parser.add_argument('--dataloader_num_workers', type=int,
                        default=8, help='The number of threads used by dataloader.')
    parser.add_argument('--dataloader_drop_last', type=str2bool,
                        default=True, help='Whether drop the last batch sample.')
    parser.add_argument('--dataset_dir', type=str,
                        default="./data/ssp", help='Local path for dataset.')
    parser.add_argument('--replace_T', type=bool, default=True)
    parser.add_argument('--replace_U', type=bool, default=False)
    # training args
    parser.add_argument('--disable_tqdm', type=str2bool,
                        default=False, help='Disable tqdm display if true.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument('--device', type=str, help='cpu, cuda:0')

    parser.add_argument('--num_train_epochs', type=int, default=60,
                        help='The number of epoch for training.')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='The number of samples used per step, must be 1.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='the learning rate for optimizer (default: 0.001)')
    parser.add_argument('--metrics',
                        type=str2list,
                        default="F1s,Accuracy,Precision,Recall",
                        help='Use which metrics to evaluate model, could be concatenate by ",".')

    # logging args
    parser.add_argument('--output_dir', type=str, help='Logging directory.')
    parser.add_argument('--logging_steps', type=int, default=100,
                        help='Update visualdl logs every logging_steps.')
    parser.add_argument('--freeze_base', action='store_true', help='freeze base language model')

    args = parser.parse_args()
    return args


def freeze(model):
    for name, para in model.named_parameters():
        para.requires_grad = False


def parse_epoch_from_str(s):
    # examples: ep5, epoch5
    begin = 0
    while s[begin].isalpha():
        begin += 1
    return int(s[begin:])

if __name__ == "__main__":
    args = get_args()

    if args.output_dir is None:
        if args.model_name=='structRFM' and args.LM_path and os.path.exists(args.LM_path):
            run_name = os.path.basename(args.LM_path)
            run_name = run_name[:run_name.rfind('.')]
        else:
            run_name = args.model_name
        args.output_dir = os.path.join('outputs', run_name)
    os.makedirs(args.output_dir, exist_ok=True)

    # ========== post process
    if args.max_seq_len == 0:
        args.max_seq_len = MAX_SEQ_LEN[args.model_name]
    args.dataset_train = osp.join(args.dataset_dir, DATASETS[args.task_name][0])
    args.dataset_test = osp.join(args.dataset_dir, DATASETS[args.task_name][1])

    # ========== args check
    assert args.replace_T ^ args.replace_U, "Only replace T or U."

    args.device = get_and_set_device(args.device)
    # ========== Build tokenizer, model, criterion

    if args.model_name == "RNABERT":
        tokenizer = RNATokenizer(args.vocab_path + "{}.txt".format(args.model_name))
        from RNABERT.rnabert import BertModel
        model_config = get_config(
            args.config_path + "{}.json".format(args.model_name))
        pretrained_model = BertModel(model_config)
        if args.freeze_base:
            freeze(pretrained_model)
        pretrained_model = RNABertForSsp(pretrained_model)
        pretrained_model._load_pretrained_bert(args.LM_path)
    elif args.model_name == "RNAMSM":
        tokenizer = RNATokenizer(args.vocab_path + "{}.txt".format(args.model_name))
        from RNAMSM.model import MSATransformer
        model_config = get_config(
            args.config_path + "{}.json".format(args.model_name))
        pretrained_model = MSATransformer(**model_config)
        if args.freeze_base:
            freeze(pretrained_model)
        pretrained_model = RNAMsmForSsp(pretrained_model)
        pretrained_model._load_pretrained_bert(args.LM_path)
    elif args.model_name == "RNAFM":
        tokenizer = RNATokenizer(args.vocab_path + "{}.txt".format(args.model_name))
        from RNAFM import fm
        pretrained_model, alphabet = fm.pretrained.rna_fm_t12()
        if args.freeze_base:
            freeze(pretrained_model)
        pretrained_model = RNAFmForSsp(pretrained_model)
        # pretrained_model._load_pretrained_bert(args.LM_path)
    elif args.model_name == 'structRFM':
        tokenizer = get_mlm_tokenizer(max_length=args.max_seq_len)
        model_paras = get_model_scale(args.model_scale)
        model = get_structRFM(from_pretrained=args.LM_path, output_hidden_states=True, tokenizer=tokenizer, pretrained_length=514, **model_paras)
        if args.freeze_base:
            freeze(model)
        pretrained_model = structRFMForSsp(model)
        # tokenizer = AutoTokenizer.from_pretrained(args.LM_path)
    else:
        raise ValueError("Unknown model name: {}".format(args.model_name))


    # load model
    config = {
        'max_helix_length': 30,
        'embed_size': 64,
        'num_filters': (64, 64, 64, 64, 64, 64, 64, 64),
        'filter_size': (5, 3, 5, 3, 5, 3, 5, 3),
        'pool_size': (1, ),
        'dilation': 0,
        'num_lstm_layers': 2,
        'num_lstm_units': 32,
        'num_transformer_layers': 0,
        'num_hidden_units': (32, ),
        'num_paired_filters': (64, 64, 64, 64, 64, 64, 64, 64),
        'paired_filter_size': (5, 3, 5, 3, 5, 3, 5, 3),
        'dropout_rate': 0.5,
        'fc_dropout_rate': 0.5,
        'num_att': 8,
        'pair_join': 'cat',
        'no_split_lr': False,
        'n_out_paired_layers': 3,
        'n_out_unpaired_layers': 0,
        'exclude_diag': True,
        'embed_dim': EMBED_DIMS[args.model_name]
    }
    model = MixedFold(init_param=param_turner2004, **config)

    ## loading checkpoints
    if not args.checkpoint_path:
        paths = [f for f in os.listdir(args.output_dir) if f.startswith('model') and f[f.rfind('.'):] in ['.pt', '.pth']]
        if paths:
            args.checkpoint_path = os.path.join(args.output_dir, sorted(paths, key=lambda name: parse_epoch_from_str(name.split('_')[1]))[-1])
    begin_epoch = 0
    print(f'checkpoint_path: {args.checkpoint_path}')
    if args.checkpoint_path and os.path.exists(args.checkpoint_path):
        data = torch.load(args.checkpoint_path)
        if 'model' in data:
            model.load_state_dict(data['model'])
            if 'pretrained_model' in data:
                pretrained_model.load_state_dict(data['pretrained_model'])
        else:
            model.load_state_dict(data)
        print(f'Loading {args.checkpoint_path}')
        epoch_str = os.path.basename(args.checkpoint_path).split('_')[1]
        begin_epoch = int(''.join([ch for ch in epoch_str if ch.isdigit()]))+1
    print(f'Training begin_epoch: {begin_epoch}')
    model.to(args.device)
    pretrained_model = pretrained_model.to(args.device)

    # load loss function
    _loss_fn = StructuredLoss(loss_pos_paired=0.5, loss_neg_paired=0.005,
                              loss_pos_unpaired=0., loss_neg_unpaired=0., l1_weight=0., l2_weight=0.)
    _loss_fn = _loss_fn.to(args.device)

    # ========== Prepare data
    train_dataset = BPseqDataset(args.dataset_dir, args.dataset_train)
    test_dataset = BPseqDataset(args.dataset_dir, args.dataset_test)

    # ========== Create the data collator
    _collate_fn = SspCollator(max_seq_len=args.max_seq_len,
                              tokenizer=tokenizer, replace_T=args.replace_T, replace_U=args.replace_U, is_structRFM=args.model_name=='structRFM')

    # ========== Create the learning_rate scheduler (if need) and optimizer
    _optimizer = AdamW([
                       dict(params=[para for para in model.parameters() if para.requires_grad], lr=args.lr),
                       dict(params=[para for para in pretrained_model.parameters() if para.requires_grad], lr=args.lr/2),
                      ])

    # ========== Create the metrics
    _metric = SspMetrics(metrics=args.metrics)

    print('dataset train:test', len(train_dataset), len(test_dataset)) 
    # ========== Training
    ssp_trainer = SspTrainer(
        args=args,
        tokenizer=tokenizer,
        model=model,
        pretrained_model=pretrained_model,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=_collate_fn,
        loss_fn=_loss_fn,
        optimizer=_optimizer,
        compute_metrics=_metric,
    )
    if args.train:
        for i_epoch in range(begin_epoch, begin_epoch+args.num_train_epochs):
            if not ssp_trainer.get_status():
                print("Epoch: {}".format(i_epoch))
                ssp_trainer.train(i_epoch)
                ssp_trainer.eval(i_epoch)
