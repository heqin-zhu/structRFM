import os
import random
import argparse
from functools import partial

import torch
import tensorboard
from datasets import load_dataset
from transformers import TrainingArguments, Trainer
from transformers import DataCollatorForLanguageModeling
from transformers import LlamaForCausalLM, LlamaModel

from .tokenizer import get_tokenizer_by_tag, preprocess
from .model import get_llama_model, get_bert_model


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
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--tag', type=str, choices=['mlm', 'ar'], default='mlm')
    parser.add_argument('--max_length', type=int, default=512, help='Max length of tokens')
    parser.add_argument('--dim', type=int, default=768, help='hidden dim')
    parser.add_argument('--layer', type=int, default=12)
    parser.add_argument('--from_pretrained', type=str, help='for model')
    parser.add_argument('--resume_from_checkpoint', type=str, help='for trainer')
    args = parser.parse_args()
    return args


def pretrain(args, tag):
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    set_seed(42)
    tokenizer = get_tokenizer_by_tag(tag=tag, max_length=args.max_length)

    model = None
    model_name = ''
    if tag=='mlm':
        model = get_bert_model(args.dim, args.layer, args.from_pretrained, tokenizer)
        model_name = 'bert'
    else:
        model = get_llama_causal_model(args.dim, args.layer, args.from_pretrained, tokenizer)
        model_name = 'llama'
    model_param_size = sum(t.numel() for t in model.parameters())
    print(model)
    print(f"model paras: {model_param_size/1e6:.1f}M parameters")

    ## load_dataset
    ## path options:  json, csv, text, panda, imagefolder
    # dataset = load_dataset('csv', data_files={'train':['my_train_file_1.csv','my_train_file_2.csv'],'test': 'my_test_file.csv'})

    # dataset = load_dataset('csv', data_files=args.data_path, split='train[:90%]+train[90%:100%]', verification_mode='no_checks')
    dataset = load_dataset('csv', data_files=args.data_path)
    pre_func = partial(preprocess, tokenizer=tokenizer, tag=tag)
    dataset = dataset.map(pre_func, batched=True, num_proc=8)

    print(dataset['train'][0])
    print(dataset.keys(), dataset['train'], len(dataset['train'])) # TODO
    # exit()

    # TODO, structure-aware masking
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm= tag.lower()=='mlm')
    # DataCollatorWithPadding, DataCollatorForSeq2Seq, ForWholeWordMask

    epoch = 50
    batch_size = 32
    model_size = f'{args.dim}_{args.layer}'
    training_args = TrainingArguments(
        output_dir=f"./{model_name}_{model_size}_results",
        evaluation_strategy="epoch",
        learning_rate=3e-4,
        weight_decay=0.1,
        gradient_accumulation_steps=1,
        per_device_train_batch_size=batch_size,
        # warmup_steps=10_000,
        # max_steps=100_000, # only a demo
        # logging_steps=1000,
        # eval_steps=5000,
        num_train_epochs=epoch,
        logging_strategy="epoch",
        save_strategy="epoch",
        fp16=True,
        report_to = "tensorboard",
    )
    my_callbacks = []
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["train"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=my_callbacks,
    )


    if args.resume_from_checkpoint:
        print(f'Loading checkpoint: {args.resume_from_checkpoint}')
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    else:
        trainer.train()

    trainer.save_model(f"./{model_name}_{model_size}")
    tokenizer.save_pretrained(f"./{model_name}_{model_size}")


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
