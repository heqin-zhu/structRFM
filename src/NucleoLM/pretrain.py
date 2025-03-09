import os
import random
import argparse
from functools import partial

import torch
import tensorboard
from datasets import load_dataset, load_from_disk
from transformers import TrainingArguments, Trainer
from transformers import DataCollatorForLanguageModeling
from transformers import LlamaForCausalLM, LlamaModel

from .data import get_tokenizer_by_tag, preprocess, PretrainDataCollatorWithStructure
from .model import get_bert_mlm_stru_pretraining, get_llama_model, get_llama_causal_model


def set_seed(seed, deterministic=True):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic


def parse_args():
    parser = argparse.ArgumentParser(description='RNA LM')
    parser.add_argument('--run_name', type=str, required=True)
    # Data args
    parser.add_argument('--data_path', type=str, default='../RNAcentral/RNAcentral_BPfold_SS')
    parser.add_argument('--tag', type=str, choices=['mlm', 'ar'], default='mlm')
    parser.add_argument('--max_length', type=int, default=512, help='Max length of tokens')
    parser.add_argument('--seed', type=int, default=42)

    # Model args
    parser.add_argument('--dim', type=int, default=768, help='hidden dim')
    parser.add_argument('--layer', type=int, default=12)
    parser.add_argument('--from_pretrained', type=str, help='for model')
    parser.add_argument('--resume_from_checkpoint', action='store_true', help='for trainer')

    # Training args
    parser.add_argument('--lr', type=float, default=0.0003, help='learning rate')
    parser.add_argument('--epoch', type=int, default=10, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--mlm_structure', action='store_true')
    args = parser.parse_args()
    return args


def get_dataset(data_path, tokenizer, tag):
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
        pre_func = partial(preprocess, tokenizer=tokenizer, tag=tag)
        dataset = dataset.map(pre_func, batched=True, num_proc=8)
        dataset.save_to_disk(disk_dir)
        return dataset


def pretrain(args, tag):
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    set_seed(args.seed)
    tokenizer = get_tokenizer_by_tag(tag=tag, max_length=args.max_length)

    model = None
    model_name = ''
    if tag=='mlm':
        model = get_bert_mlm_stru_pretraining(args.dim, args.layer, args.from_pretrained, tokenizer)
        model_name = 'bert'
    else:
        model = get_llama_causal_model(args.dim, args.layer, args.from_pretrained, tokenizer)
        model_name = 'llama'
    model_param_size = sum(t.numel() for t in model.parameters())
    print(model)
    print(f"{model_name} model paras: {model_param_size/1e6:.1f}M")

    dataset = get_dataset(args.data_path, tokenizer, tag)
    split_dataset = dataset
    if 'test' not in split_dataset:
        if 'validate' in dataset:
            split_dataset['test'] = split_dataset['validate']
        else:
            split_dataset = dataset['train'].train_test_split(test_size=0.05, seed=args.seed)
    print(split_dataset)

    # DataCollatorWithPadding, DataCollatorForSeq2Seq, ForWholeWordMask
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm= tag=='mlm')
    if tag == 'mlm' and args.mlm_structure:
        # structure-directed masking
        data_collator = PretrainDataCollatorWithStructure(tokenizer=tokenizer, mlm=True)

    model_size = f'{args.dim}_{args.layer}'
    total_steps = 21477078//args.batch_size 
    step_interval = total_steps//10
    training_args = TrainingArguments(
        output_dir=args.run_name,
        learning_rate=args.lr,
        num_train_epochs=args.epoch,
        max_steps=total_steps*args.epoch,
        weight_decay=0., # TODO
        gradient_accumulation_steps=1, # 显存不够大 bs， 增加此参数
        per_device_train_batch_size=args.batch_size,
        # warmup_steps=10_000,
        logging_strategy="steps",
        logging_steps=step_interval//10,
        evaluation_strategy="steps",
        eval_steps=step_interval,
        save_strategy="steps", 
        save_steps=step_interval, 
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss', # TODO
        greater_is_better=False,
        fp16=True,
        report_to = "tensorboard",
    )
    my_callbacks = []
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=split_dataset["train"], # torch.utils.data.Dataset or torch.utils.data.IterableDataset: if torch.utils.data.Dataset, 则会自动删除模型的 forward() 方法不接受的列。 这也太坑了, data_collator 要用到的时候，被删除了， 找了半天的bug
        eval_dataset=split_dataset["test"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=my_callbacks,
    )

    if args.resume_from_checkpoint and any(f.startswith('checkpoint') for f in os.listdir(args.run_name)):
        print(f'Resume_from_checkpoint...')
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.save_model(os.path.join(args.run_name, f"trainer_ep{args.epoch}"))
    tokenizer.save_pretrained(os.path.join(args.run_name, f"tokenizer_ep{args.epoch}"))


def run_pretrain():
    args = parse_args()
    assert args.tag in {'mlm', 'ar'}, f'tag={args.tag} should be "mlm" or "ar"'
    pretrain(args, args.tag)


if __name__ == '__main__':
    '''
    BPfold 7M
    LLaMA:
    (dim, layer), para
    (256, 4), 3M
    (384，6), 13M
    (512, 8), 31M
    (768, 12), 106M
    (4096, 32), 1B
    '''
    run_pretrain()
