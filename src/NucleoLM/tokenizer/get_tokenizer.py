import os
import json

from tokenizers import Tokenizer
from tokenizers.implementations import BaseTokenizer
from transformers import PreTrainedTokenizerFast


from BPfold.util.RNA_kit import connects2dbn


SRC_DIR = os.path.abspath(os.path.dirname(__file__))


__all__ = ['get_tokenizer_by_tag',  'preprocess']


def get_tokenizer(tokenizer_file, max_length, bos_token, eos_token):
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=tokenizer_file,
        padding_side='right',
        truncation_side='right',
        cls_token=bos_token,
        bos_token=bos_token,
        sep_token=eos_token,
        eos_token=eos_token,
        unk_token='[UNK]',
        mask_token='[MASK]',
        pad_token='[PAD]',
        model_max_length=max_length
    )
    return tokenizer


def get_tokenizer_by_tag(tag='mlm', *args, **kargs):
    tag = tag.lower()
    assert tag in ['mlm', 'ar'], f'Unknown tag: {tag}, should be in ["mlm", "ar"]'
    path = os.path.join(SRC_DIR, f'tokenizer_{tag}.json')
    bos_token = '[CLS]' if tag == 'mlm' else '<BOS>'
    eos_token = '[SEP]' if tag == 'mlm' else '<EOS>'
    tokenizer = get_tokenizer(tokenizer_file=path, bos_token=bos_token, eos_token=eos_token, *args, **kargs)
    return tokenizer


def preprocess(samples, tokenizer, tag='mlm'):
    processed_samples = {
        "input_ids": [],
        "attention_mask": []
    }
    dbn_vocab = set('.()[]{}')
    bos_token = tokenizer.bos_token
    eos_token = tokenizer.eos_token
    for seq, connects in zip(samples['seq'], samples['connects']):
        text = None
        if tag=='mlm':
            # "[CLS]AUGCNX[SEP]"
            # TODO  connects
            seq = seq.upper().replace('T', 'U')
            text = f"{bos_token}{seq}{eos_token}" # head/rear special tokens will be removed and readded.
        elif tag=='ar':
            # "<BOS>AUGCNX<SS>DBN<EOS>"
            seq = seq.upper().replace('T', 'U')
            dbn = connects2dbn(json.loads(connects))
            dbn = ''.join([i if i in dbn_vocab else '?' for i in dbn])
            text = f"{bos_token}{seq}<SS>{dbn}{eos_token}" # head/rear special tokens will be removed and readded.
        else:
            raise Exception(f'tag={tag} should be "mlm" or "ar"')
        tokenized_input = tokenizer(text, padding="longest", truncation=True, max_length=tokenizer.model_max_length)
        processed_samples["input_ids"].append(tokenized_input["input_ids"])
        processed_samples["attention_mask"].append(tokenized_input["attention_mask"])
        # labels = tokenizer(sample['...'], max_length=tokenizer.model_max_length, truncation=True)
        # processed_samples['labels'].append(labels['input_ids'])
    return processed_samples


if __name__ == '__main__':
    seq = 'ATGCUUNK'
    tokenizer = get_tokenizer_by_tag('mlm', max_length=512)
    text = f"[CLS]{seq.replace('T', 'U')}[SEP]"
    print(text, tokenizer(text))

    dbn = '(...)...'
    tokenizer = get_tokenizer_by_tag('ar', max_length=512)
    text = f"<BOS>{seq.replace('T', 'U')}<SS>{dbn}<EOS>"
    print(text, tokenizer(text))
