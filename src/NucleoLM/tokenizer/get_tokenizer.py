import os

from tokenizers import Tokenizer
from tokenizers.implementations import BaseTokenizer
from transformers import PreTrainedTokenizerFast


from BPfold.util.RNA_kit import connects2dbn


SRC_DIR = os.path.abspath(os.path.dirname(__file__))


def get_tokenizer(tokenizer_file, max_length):
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=tokenizer_file,
        padding_side='right',
        truncation_side='right',
        cls_token='[CLS]',
        bos_token='[CLS]',
        sep_token='[SEP]',
        eos_token='[SEP]',
        unk_token='[UNK]',
        mask_token='[MASK]',
        pad_token='[PAD]',
        model_max_length=max_length
    )
    return tokenizer


def get_tokenizer_by_tag(tag='MLM', *args, **kargs):
    tag = tag.lower()
    assert tag in ['mlm', 'ar'], f'Unknown tag: {tag}, should be in ["mlm", "ar"]'
    path = os.path.join(SRC_DIR, f'tokenizer_{tag}.json')
    tokenizer = get_tokenizer(tokenizer_file=path, *args, **kwargs)
    return tokenizer


def preprocess_mlm(samples, tokenizer):
    processed_samples = {
        "input_ids": [],
        "attention_mask": []
    }
    for seq in samples['seq']:
        # "[CLS]AUGCNX[SEP]"
        seq = seq.upper().replace('T', 'U')
        text = f"[CLS]{seq}[SEP]" # head/rear special tokens will be removed and readded.
        tokenized_input = tokenizer(text, padding="longest", truncation=True, max_length=tokenizer.model_max_length)
        processed_samples["input_ids"].append(tokenized_input["input_ids"])
        processed_samples["attention_mask"].append(tokenized_input["attention_mask"])
        # labels = tokenizer(sample['...'], max_length=tokenizer.model_max_length, truncation=True)
        # processed_samples['labels'].append(labels['input_ids'])
    return processed_samples


def preprocess_ar(samples, tokenizer):
    processed_samples = {
        "input_ids": [],
        "attention_mask": []
    }
    dbn_vocab = set('.()[]{}')
    for seq, connects in zip(samples['seq'], samples['connects']):
        # "[CLS]AUGCNX[SEP]"
        seq = seq.upper().replace('T', 'U')
        dbn = connects2dbn(connects)
        dbn = ''.join([i if i in dbn_vocab else '?' for i in dbn])
        text = f"[CLS]{seq}[SEP]{dbn}[SEP]" # head/rear special tokens will be removed and readded.
        tokenized_input = tokenizer(text, padding="longest", truncation=True, max_length=tokenizer.model_max_length)
        processed_samples["input_ids"].append(tokenized_input["input_ids"])
        processed_samples["attention_mask"].append(tokenized_input["attention_mask"])
        # labels = tokenizer(sample['...'], max_length=tokenizer.model_max_length, truncation=True)
        # processed_samples['labels'].append(labels['input_ids'])
    return processed_samples


if __name__ == '__main__':
    tokenizer = get_tokenizer('tokenizer_mlm.json', max_length=512)

    seq_list = [
            'ATGCUUNK',
               ]
    for seq in seq_list:
        text = f"[CLS]{seq.replace('T', 'U')}[SEP]"
        print(seq, tokenizer(seq))
        print(text, tokenizer(text))
