import os
import json
from functools import partial

from datasets import load_dataset, load_from_disk

from ..util.RNA_kit import connects2dbn


def process_mlm_input_seq(seq, bos_token='[CLS]', eos_token='[SEP]'):
    # "[CLS]AUGCNX[SEP]"
    seq = seq.upper().replace('T', 'U')
    text = f"{bos_token}{seq}{eos_token}" # head/rear special tokens will be removed and readded.
    return text


def process_ar_input_seq_and_connects(seq, connects_str, bos_token='<BOS>', eos_token='<EOS>'):
    # "<BOS>AUGCNX<SS>DBN<EOS>"
    seq = seq.upper().replace('T', 'U')
    dbn = connects2dbn(json.loads(connects_str))
    dbn = ''.join([i if i in dbn_vocab else '?' for i in dbn])
    text = f"{bos_token}{seq}<SS>{dbn}{eos_token}" # head/rear special tokens will be removed and readded.
    return text


def preprocess_mlm_pretrain(samples, tokenizer):
    ''' columns in samples: seq, connects '''
    processed_samples = {
        "input_ids": [],
        "attention_mask": [],
        "connects": [],
    }
    dbn_vocab = set('.()[]{}')
    bos_token = tokenizer.bos_token
    eos_token = tokenizer.eos_token
    for seq, connects_str in zip(samples['seq'], samples['connects']):
        text = process_mlm_input_seq(seq, bos_token, eos_token)
        connects = [0] + json.loads(connects_str) + [0]
        processed_samples["connects"].append(connects) # for mlm_structure
        tokenized_input = tokenizer(text, padding="longest", truncation=True, max_length=tokenizer.model_max_length)
        processed_samples["input_ids"].append(tokenized_input["input_ids"])
        processed_samples["attention_mask"].append(tokenized_input["attention_mask"])
    return processed_samples


def preprocess_mlm_finetune(samples, tokenizer):
    ''' columns in samples: seq '''
    processed_samples = {
        "input_ids": [],
        "attention_mask": [],
    }
    dbn_vocab = set('.()[]{}')
    bos_token = tokenizer.bos_token
    eos_token = tokenizer.eos_token
    for seq in samples['seq']:
        text = process_mlm_input_seq(seq, bos_token, eos_token)
        tokenized_input = tokenizer(text, padding="longest", truncation=True, max_length=tokenizer.model_max_length)
        processed_samples["input_ids"].append(tokenized_input["input_ids"])
        processed_samples["attention_mask"].append(tokenized_input["attention_mask"])
    return processed_samples


def preprocess_ar(samples, tokenizer):
    ''' columns in samples: seq, connects '''
    processed_samples = {
        "input_ids": [],
        "attention_mask": []
    }
    dbn_vocab = set('.()[]{}')
    bos_token = tokenizer.bos_token
    eos_token = tokenizer.eos_token
    for seq, connects_str in zip(samples['seq'], samples['connects']):
        text = process_ar_input_seq_and_connects(seq, connects_str, bos_token, eos_token)
        tokenized_input = tokenizer(text, padding="longest", truncation=True, max_length=tokenizer.model_max_length)
        processed_samples["input_ids"].append(tokenized_input["input_ids"])
        processed_samples["attention_mask"].append(tokenized_input["attention_mask"])
        # labels = tokenizer(sample['...'], max_length=tokenizer.model_max_length, truncation=True)
        # processed_samples['labels'].append(labels['input_ids'])
    return processed_samples


def preprocess_dataset(data_path, tokenizer, tag, preprocess, save_to_disk=True):
    '''
        save_to_disk, or .csv file (cols: name, seq, [connects])
    '''
    ## load_dataset
    ## path options:  json, csv, text, panda, imagefolder
    # dataset = load_dataset('csv', data_files={'train':['my_train_file_1.csv','my_train_file_2.csv'],'test': 'my_test_file.csv'})
    # train_dataset = load_dataset('csv', data_files=args.data_path, split='train[:90%]', verification_mode='no_checks')

    if data_path.endswith('_disk'):
        return load_from_disk(data_path)

    dataset_name = os.path.basename(data_path)
    p = dataset_name.rfind('.')
    if p!=-1:
        dataset_name = dataset_name[:p]
    disk_dir = os.path.join(os.path.dirname(data_path), f'{dataset_name}_for_{tag}_disk')
    if os.path.exists(disk_dir):
        return load_from_disk(disk_dir)
    else:
        data_files = data_path if os.path.isfile(data_path) else [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.csv')]
        dataset = load_dataset("csv", data_files=data_files)
        pre_func = partial(preprocess, tokenizer=tokenizer)
        dataset = dataset.map(pre_func, batched=True, num_proc=8)
        if save_to_disk:
            dataset.save_to_disk(disk_dir)
        return dataset


def get_pretrain_dataset(data_path, tokenizer, tag, save_to_disk=True):
    preprocess = preprocess_mlm_pretrain if tag == 'mlm' else preprocess_ar
    return preprocess_dataset(data_path, tokenizer, tag, preprocess=preprocess, save_to_disk=save_to_disk)


def get_finetune_dataset(data_path, tokenizer, tag, save_to_disk=True):
    preprocess = preprocess_mlm_finetune if tag == 'mlm' else preprocess_ar
    return preprocess_dataset(data_path, tokenizer, tag, preprocess=preprocess, save_to_disk=save_to_disk)
