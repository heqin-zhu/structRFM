import os
from functools import partial

from datasets import load_dataset, load_from_disk

from .tokenizer_and_preprocess import preprocess


def get_dataset(data_path, tokenizer, tag, save_to_disk=True):
    '''
        save_to_disk, or .csv file (cols: name, seq, connects)
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
        dataset = load_dataset("csv", data_files=data_files) # columns: name, seq, connects
        pre_func = partial(preprocess, tokenizer=tokenizer, tag=tag)
        dataset = dataset.map(pre_func, batched=True, num_proc=8)
        if save_to_disk:
            dataset.save_to_disk(disk_dir)
        return dataset
