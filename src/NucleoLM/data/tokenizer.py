import os

from tokenizers import Tokenizer
from tokenizers.implementations import BaseTokenizer
from transformers import PreTrainedTokenizerFast

SRC_DIR = os.path.abspath(os.path.dirname(__file__))


def get_mlm_tokenizer(tokenizer_file=None, max_length=514, *args, **kargs):
    if tokenizer_file is None:
        tokenizer_file = os.path.join(SRC_DIR, f'tokenizer_mlm.json')
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=tokenizer_file,
        padding_side='right', # delt by collator
        truncation_side='right',
        cls_token='[CLS]',
        bos_token='[CLS]',
        sep_token='[SEP]',
        eos_token='[SEP]',
        unk_token='[UNK]',
        mask_token='[MASK]',
        pad_token='[PAD]',
        model_max_length=max_length,
        *args,
        **kargs,
    )
    return tokenizer


def get_ar_tokenizer(tokenizer_file=None, max_length=514, *args, **kargs):
    if tokenizer_file is None:
        tokenizer_file = os.path.join(SRC_DIR, f'tokenizer_ar.json')
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=tokenizer_file,
        padding_side='right', # delt by collator
        truncation_side='right',
        cls_token='<BOS>',
        bos_token='<BOS>',
        sep_token='<EOS>',
        eos_token='<EOS>',
        unk_token='<UNK>',
        mask_token='<MASK>',
        pad_token='<PAD>',
        model_max_length=max_length,
        *args,
        **kargs,
    )
    return tokenizer


if __name__ == '__main__':
    seq = 'ATGCUUNK'
    tokenizer = get_mlm_tokenizer()
    text = f"[CLS]{seq.replace('T', 'U')}[SEP]"
    print(text, tokenizer(text))

    dbn = '(...)...'
    tokenizer = get_ar_tokenizer()
    text = f"<BOS>{seq.replace('T', 'U')}<SS>{dbn}<EOS>"
    print(text, tokenizer(text))
