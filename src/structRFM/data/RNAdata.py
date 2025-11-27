import os
from ast import literal_eval
from functools import partial

from datasets import load_dataset, load_from_disk
from BPfold.util.RNA_kit import connects2dbn


def process_mlm_input_seq(seq, bos_token='[CLS]', eos_token='[SEP]'):
    # "[CLS]AUGCNX[SEP]"
    seq = seq.upper().replace('T', 'U')
    text = f"{bos_token}{seq}{eos_token}" # head/rear special tokens will be removed and readded.
    return text


def process_ar_input_seq_and_connects(seq, connects_str, bos_token='<BOS>', eos_token='<EOS>'):
    # "<BOS>AUGCNX<SS>DBN<EOS>"
    seq = seq.upper().replace('T', 'U')
    # use literal_eval instread of json.loads, since json.loads is strict: such as 1. 'str' will be wrong, 2. "null", not "None"
    dbn = connects2dbn(literal_eval(connects_str))
    dbn = ''.join([i if i in dbn_vocab else '?' for i in dbn])
    text = f"{bos_token}{dbn}<SS>{seq}{eos_token}" # head/rear special tokens will be removed and readded.
    return text


def preprocess_mlm_with_structure(samples, tokenizer):
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
        connects = [0] + literal_eval(connects_str) + [0]
        processed_samples["connects"].append(connects) # for mlm_structure
        tokenized_input = tokenizer(text, padding="longest", truncation=True, max_length=tokenizer.model_max_length)
        processed_samples["input_ids"].append(tokenized_input["input_ids"])
        processed_samples["attention_mask"].append(tokenized_input["attention_mask"])
    return processed_samples


def preprocess_mlm_without_structure(samples, tokenizer):
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


def preprocess_and_load_dataset(data_path, tokenizer, tag, with_structure=True, save_to_disk=True):
    '''
        save_to_disk, or .csv file (cols: name, seq, [connects])
    '''
    ## transformer.load_dataset
    ## path options:  json, csv, text, panda, imagefolder
    # dataset = load_dataset('csv', data_files={'train':['my_train_file_1.csv','my_train_file_2.csv'],'test': 'my_test_file.csv'})
    # train_dataset = load_dataset('csv', data_files=args.data_path, split='train[:90%]', verification_mode='no_checks')

    dataset_name = os.path.basename(data_path)
    p = dataset_name.rfind('.')
    if p!=-1:
        dataset_name = dataset_name[:p]
    disk_dir = os.path.join(os.path.dirname(data_path), f'{dataset_name}_for_{tag}_disk')

    preprocess_func = None
    if tag == 'ar': # must with structure
        preprocess_func = preprocess_ar
    elif tag == 'mlm':
        if with_structure: # pretrain with structure
            preprocess_func = preprocess_mlm_with_structure
        else:   # pretrain without structure, or finetinue 
            preprocess_func = preprocess_mlm_without_structure
    else:
        raise Exception(f'Unknown tag: {tag}')

    dataset = None
    if os.path.exists(disk_dir):
        print(f'Loading disk data: {disk_dir}')
        dataset = load_from_disk(disk_dir)
    elif data_path.endswith('_disk') and os.path.exists(data_path):
        print(f'Loading disk data: {data_path}')
        dataset = load_from_disk(data_path)
    else:
        data_files = data_path if os.path.isfile(data_path) else [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.csv')]
        dataset = load_dataset("csv", data_files=data_files)
        pre_func = partial(preprocess_func, tokenizer=tokenizer)
        dataset = dataset.map(pre_func, batched=True, num_proc=8)
        if save_to_disk:
            dataset.save_to_disk(disk_dir)
    columns = ["name", "seq", "input_ids", "attention_mask", "connects"]
    if tag == 'mlm' and not with_structure:
        columns = ["name", "seq", "input_ids", "attention_mask"]
    return dataset.select_columns(columns)
