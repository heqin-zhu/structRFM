import os
import math
import argparse
from collections import defaultdict

import pandas as pd
import torch
from tqdm import tqdm

from structRFM.model import get_structRFM, get_model_scale
from structRFM.data import get_mlm_tokenizer
from BPfold.util.RNA_kit import read_fasta, connects2mat, connects2dbn, write_SS
from BPfold.util.misc import get_file_name

file_dir = os.path.dirname(__file__)
ssp_dir = os.path.join(os.path.dirname(file_dir), 'tasks', 'seqcls_ssp')
import sys
sys.path.append(ssp_dir)
from utils import Stack
import param_turner2004
from ss_pred import structRFMForSsp, MixedFold


class SSP_data(torch.utils.data.Dataset):
    def __init__(self, fasta_path):
        super().__init__()
        self.name_seqs = list(read_fasta(fasta_path))

    def __len__(self):
        return len(self.name_seqs)

    def __getitem__(self, idx):
        name, seq = self.name_seqs[idx]
        return {'name': name, 'seq': seq}


class SSP_collator:
    def __init__(self, max_seq_len, tokenizer, replace_T=True, replace_U=False, model_name='structRFM'):
        self.stack_fn = Stack()
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        self.replace_T = replace_T
        self.replace_U = replace_U
        self.model_name = model_name

    def process_input(self, seq):
        if self.model_name == 'structRFM':
            return '[CLS]'+seq
        elif self.model_name.lower().startswith('rinalmo'):
            return seq
        else:
            kmer_text = seq2kmer(seq)
            return "[CLS] " + kmer_text

    def __call__(self, raw_data_b):
        raw_data = raw_data_b[0]
        name_stack = [raw_data["name"] if "name" in raw_data else None]
        seq_stack = [raw_data["seq"]]
        seq_stack = [x[:self.max_seq_len-1] for x in seq_stack]

        input_seqs = raw_data["seq"].upper()
        input_seqs = input_seqs.replace("T", "U") if self.replace_T else input_seqs.replace("U", "T")
        kmer_text = self.process_input(input_seqs)
        input_ids_stack = self.tokenizer(kmer_text)["input_ids"]
        input_ids_stack = input_ids_stack[:self.max_seq_len]
        if None in input_ids_stack:
            # replace all None with 0
            input_ids_stack = [0 if x is None else x for x in input_ids_stack]
        input_ids = self.stack_fn(input_ids_stack)
        return {
            "names": name_stack,
            "seqs": seq_stack,
            "input_ids": input_ids,
        }


def prepare_model(checkpoint_path):
    LM_path = None
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    tokenizer = get_mlm_tokenizer(max_length=514)
    model_paras = get_model_scale('base')
    LM = get_structRFM(from_pretrained=LM_path, output_hidden_states=True, tokenizer=tokenizer, **model_paras)
    pretrained_model = structRFMForSsp(LM)

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
        'embed_dim': 768,
    }
    model = MixedFold(init_param=param_turner2004, **config)

    # load checkpoint
    data = torch.load(checkpoint_path)
    if 'model' in data:
        model.load_state_dict(data['model'])
        if 'pretrained_model' in data:
            pretrained_model.load_state_dict(data['pretrained_model'])
    else:
        raise Exception('Error checkpoint')
    print(f'Loading {checkpoint_path}')
    pretrained_model = pretrained_model.to(device)
    model.to(device)
    return model, tokenizer, pretrained_model


def predict_SS(model, tokenizer, pretrained_model, fasta_path):
    test_dataset = SSP_data(fasta_path)
    _collate_fn = SSP_collator(max_seq_len=512, tokenizer=tokenizer, replace_T=True, replace_U=False)
    test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=1,
            collate_fn=_collate_fn,
            num_workers=1,
            shuffle=False,
        )
    model.eval()
    n_dataset = len(test_dataloader.dataset)
    res = defaultdict(list)
    device = next(model.parameters()).device
    ret = []
    for instance in tqdm(test_dataloader):
        headers = instance["names"]
        seqs = (instance["seqs"][0], )
        input_ids = (instance["input_ids"], )
        seqs = (seqs[0], )
        input_tensor = torch.from_numpy(input_ids[0]).unsqueeze(0).to(device)
        with torch.no_grad():
            embeddings = pretrained_model(input_tensor, names=headers)
            scs, preds, bps = model(seqs, embeddings)
            for header, seq, sc, pred, bp in zip(headers, seqs, scs, preds, bps):
                name = os.path.basename(header)
                if any(name.endswith(suf) for suf in ['.bpseq', '.ct', '.dbn']):
                    name = name[:name.rfind('.')]
                ret.append({'seq_name': name, 'seq': seq, 'connects':  bp[1:]})
    return ret


def save_pred_results(pred_results, save_dir='.', save_name=None, out_type='csv', hide_dbn=False):
    '''
    Save pred_results predicted by `BPfold_predict.predict` in `save_dir` in format of `out_type`.

    Parameters
    ----------
    pred_results: Iterable(dict)
        Each item is a dict which contains keys `seq_name`, `seq`, `connects`(optional: connects_nc).
    save_dir: str
        save_dir, if not exist, will mkdir.
    save_name: str
        if not available, use basename of save_dir
    out_type: str
        csv, bpseq, ct, dbn
    '''
    def print_result(save_name, idx, seq, dbn, hide_dbn=False, num_digit=7):
        print(f"[{str(idx).rjust(num_digit)}] saved in \"{save_name}\"")
        if not hide_dbn:
            print(f'{seq}\n{dbn}')

    os.makedirs(save_dir, exist_ok=True)
    if out_type=='csv':
        if save_name is None:
            save_name = os.path.basename(os.path.abspath(save_dir))
        df = pd.DataFrame(pred_results)
        df['dbn'] = df['connects'].apply(connects2dbn)
        csv_path = os.path.join(save_dir, f'{save_name}.csv')
        num_digit = math.ceil(math.log(len(df), 10))
        for idx, row in enumerate(df.itertuples()):
            print_result(f'{csv_path}:line{idx+2}:{row.seq_name}', idx+1, row.seq, row.dbn, hide_dbn=hide_dbn, num_digit=num_digit)
        df.to_csv(csv_path, index=False)
        print(f"Predicted structures in format of dot-bracket are saved in \"{csv_path}\".")

    else:
        num_digit = 7
        for ct, res_dic in enumerate(pred_results):
            seq_name = res_dic['seq_name']
            seq = res_dic['seq']
            connects = res_dic['connects']
            path = os.path.join(save_dir, seq_name+f'.{out_type}')
            write_SS(path, seq, connects)
            print_result(path, ct+1, seq, connects2dbn(connects), hide_dbn=hide_dbn, num_digit=num_digit)


def get_args():
    parser = argparse.ArgumentParser(description='RNA secondary structure prediction using fine-tuned structRFM', add_help=True)
    # model args
    parser.add_argument('--checkpoint_path', type=str, default='checkpoints/seqcls_ssp/ssp_bpRNA1m.pth')
    parser.add_argument('--input_fasta', type=str, default='Rfam14.10-15.0/Rfam14.10-15.0.fasta')
    parser.add_argument('--output_dir', type=str, default='structRFM_SSP_results')
    parser.add_argument('--output_format', default='bpseq', choices=['bpseq', 'ct', 'dbn', 'csv'], help='Saved file type.')
    parser.add_argument('--gpu', type=str, default='0')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    model, tokenizer, pretrained_model = prepare_model(args.checkpoint_path)
    pred_results = predict_SS(model, tokenizer, pretrained_model, args.input_fasta)
    save_pred_results(pred_results, save_dir=args.output_dir, save_name=get_file_name(args.input_fasta), out_type=args.output_format, hide_dbn=False)
