# Modified from https://github.com/terry-r123/RNABenchmark
import os
import csv
import copy
import json
import logging
import pdb
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence, Tuple, List

import random
from transformers import Trainer, TrainingArguments, BertTokenizer,EsmTokenizer, EsmModel, AutoConfig, AutoModel, EarlyStoppingCallback, AutoTokenizer, AutoModelForSequenceClassification

import torch
import transformers
import sklearn
import scipy
import numpy as np
import re
from torch.utils.data import Dataset

import sys

current_path = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_path)
sys.path.append(parent_dir)


# from model.rnalm.modeling_rnalm import RnaLmForNucleotideLevel
# from model.rnalm.rnalm_config import RnaLmConfig
# from model.rnafm.modeling_rnafm import RnaFmForNucleotideLevel
# from model.rnabert.modeling_rnabert import RnaBertForNucleotideLevel
# from model.rnamsm.modeling_rnamsm import RnaMsmForNucleotideLevel
# from model.splicebert.modeling_splicebert import SpliceBertForNucleotideLevel
# from model.utrbert.modeling_utrbert import UtrBertForNucleotideLevel
# from model.utrlm.modeling_utrlm import UtrLmForNucleotideLevel
# from tokenizer.tokenization_opensource import OpenRnaLMTokenizer
from model.structRFM.modeling_structRFM import structRFMForNucleotideLevel
early_stopping = EarlyStoppingCallback(early_stopping_patience=20)
@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="")
    use_lora: bool = field(default=False, metadata={"help": "whether to use LoRA"})
    use_alibi: bool = field(default=True, metadata={"help": "whether to use alibi"})
    use_features: bool = field(default=True, metadata={"help": "whether to use alibi"})
    lora_r: int = field(default=8, metadata={"help": "hidden dimension for LoRA"})
    lora_alpha: int = field(default=32, metadata={"help": "alpha for LoRA"})
    lora_dropout: float = field(default=0.05, metadata={"help": "dropout rate for LoRA"})
    lora_target_modules: str = field(default="query,value", metadata={"help": "where to perform LoRA"})
    tokenizer_name_or_path: Optional[str] = field(default="")

@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    kmer: int = field(default=-1, metadata={"help": "k-mer for input sequence. -1 means not using k-mer."})
    data_train_path: str = field(default=None, metadata={"help": "Path to the training data."})
    data_val_path: str = field(default=None, metadata={"help": "Path to the training data."})
    data_test_path: str = field(default=None, metadata={"help": "Path to the test data. is list"})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    freeze_base: bool = field(default=False)
    cache_dir: Optional[str] = field(default=None)
    run_name: str = field(default="run")
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(default=512, metadata={"help": "Maximum sequence length."})
    gradient_accumulation_steps: int = field(default=1)
    per_device_train_batch_size: int = field(default=1)
    per_device_eval_batch_size: int = field(default=1)
    num_train_epochs: int = field(default=1)
    fp16: bool = field(default=False)
    logging_steps: int = field(default=100)
    save_steps: int = field(default=100)
    eval_steps: int = field(default=100)
    evaluation_strategy: str = field(default="steps"),
    warmup_steps: int = field(default=50)
    weight_decay: float = field(default=0.01)
    learning_rate: float = field(default=1e-4)
    save_total_limit: int = field(default=1)
    load_best_model_at_end: bool = field(default=True)
    output_dir: str = field(default="output")
    find_unused_parameters: bool = field(default=False)
    checkpointing: bool = field(default=False)
    dataloader_pin_memory: bool = field(default=False)
    eval_and_save_results: bool = field(default=True)
    save_model: bool = field(default=True)
    seed: int = field(default=42)
    report_to: str = field(default="tensorboard")
    metric_for_best_model : str = field(default="avg topk acc")
    stage: str = field(default='0')
    model_type: str = field(default='rna')
    token_type: str = field(default='6mer')
    train_from_scratch: bool = field(default=False)
    log_dir: str = field(default="output")
    attn_implementation: str = field(default="eager")
    dataloader_num_workers: int = field(default=4)
    dataloader_prefetch_factor: int = field(default=2)
def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    if torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(args.seed)
    print(f"seed is fixed ,seed = {args.seed}")

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


"""
Transform a rna sequence to k-mer string
"""
def generate_kmer_str(sequence: str, k: int) -> str:
    """Generate k-mer string from rna sequence."""
    return " ".join([sequence[i:i+k] for i in range(len(sequence) - k + 1)])


"""
Load or generate k-mer string for each sequence. The generated k-mer string will be saved to the same directory as the original data with the same name but with a suffix of "_{k}mer".
"""
def load_or_generate_kmer(data_path: str, texts: List[str], k: int) -> List[str]:
    """Load or generate k-mer string for each sequence."""
    kmer_path = data_path.replace(".csv", f"_{k}mer.json")
    if os.path.exists(kmer_path):
        logging.warning(f"Loading k-mer from {kmer_path}...")
        with open(kmer_path, "r") as f:
            kmer = json.load(f)
    else:        
        logging.warning(f"Generating k-mer...")
        kmer = [generate_kmer_str(text, k) for text in texts]
        with open(kmer_path, "w") as f:
            logging.warning(f"Saving k-mer to {kmer_path}...")
            json.dump(kmer, f)
        
    return kmer

def bpe_position(texts,attn_mask, tokenizer):
    position_id = torch.zeros(attn_mask.shape)
    for i,text in enumerate(texts):   
        text = tokenizer.tokenize(text)
        position_id[:, 0] = 1 #[cls]
        index = 0
        for j, token in enumerate(text):
            index = j+1
            position_id[i,index] = len(token) #start after [cls]   
        position_id[i, index+1] = 1 #[sep]
        
    print(position_id[0,:])
    print('position_id.shape',position_id.shape)
    return position_id
class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, 
                 data_path: str, args,
                 tokenizer: transformers.PreTrainedTokenizer, 
                 kmer: int = -1,
                 replace_T=True,
                ):

        super(SupervisedDataset, self).__init__()

        # load data from the disk
        with open(data_path, "r") as f:
            data = list(csv.reader(f))[1:]
        
        
        if len(data[0]) == 2:
            # data is in the format of [text, label]
            if replace_T:
                texts = [d[0].upper().replace("T", "U") for d in data]          
            else:
                texts = [d[0].upper().replace("U", "T") for d in data]          
            labels = np.array([list(map(float, d[1])) for d in data]).astype(np.float32)          
        else:
            print(len(data[0]))
            raise ValueError("Data format not supported.")
        seq_length = len(texts[0])
        if kmer != -1:
            # only write file on the first process
            if torch.distributed.get_rank() not in [0, -1]:
                torch.distributed.barrier()

            logging.warning(f"Using {kmer}-mer as input...")
            texts = load_or_generate_kmer(data_path, texts, kmer)
            if torch.distributed.get_rank() == 0:
                torch.distributed.barrier()
        # ensure tokenizer
        print(type(texts[0]))
        print(texts[0])
        test_example = tokenizer.tokenize(texts[0])
        print(test_example)
        print(len(test_example))
        print(tokenizer(texts[0]))
        output = tokenizer(
            texts,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )

        self.input_ids = output["input_ids"]
        self.attention_mask = output["attention_mask"]

        self.labels = labels
        self.weight_mask = torch.ones((self.input_ids.shape[0],seq_length)) # TODO
        # self.weight_mask = torch.ones((self.input_ids.shape[0],seq_length+2))
        if 'mer' in args.token_type:
            for i in range(1,kmer-1):
                self.weight_mask[:,i+1]=self.weight_mask[:,-i-2]=1/(i+1) 
            self.weight_mask[:, kmer:-kmer] = 1/kmer
        self.post_token_length = torch.zeros(self.attention_mask.shape)
        if args.token_type == 'bpe' or args.token_type == 'non-overlap':
            self.post_token_length = bpe_position(texts,self.attention_mask,tokenizer)
        self.num_labels = 3

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor,]:
        targets = torch.tensor(self.labels[i, :], dtype=torch.float32)
        return dict(input_ids=self.input_ids[i],labels=targets, attention_mask=self.attention_mask[i],
            weight_mask=self.weight_mask[i],post_token_length=self.post_token_length[i])

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, args):
        self.tokenizer = tokenizer
        self.args = args

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, attention_mask, weight_mask, post_token_length  = tuple([instance[key] for instance in instances] for key in ("input_ids" ,"labels", "attention_mask","weight_mask","post_token_length"))
        input_ids = torch.stack(input_ids)
        labels = torch.stack(labels)
        attention_mask = torch.stack(attention_mask)
        weight_mask = torch.stack(weight_mask)
        post_token_length = torch.stack(post_token_length)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
            weight_mask=weight_mask,
            post_token_length=post_token_length
        )
def top_k_accuracy_multidimensional(scores, true_labels, class_index):
    """
    Calculate top-k accuracy for a specific class across all sequences and samples,
    given multidimensional arrays for scores and true labels.
    
    Parameters:
    - scores: Numpy array of shape (num_samples, seq_length, num_classes) with predicted scores.
    - true_labels: Numpy array of shape (num_samples, seq_length) with class labels.
    - class_index: The index of the class for which to calculate top-k accuracy.
    
    Returns:
    - top_k_accuracy: The top-k accuracy for the specified class.
    """
    
    # Select scores and labels for the specified class and flatten the arrays

    class_scores = scores[:, :, class_index].flatten()
    # turn true_labels into one-hot
    true_labels = true_labels.flatten()
    one_hot_labels = np.zeros((true_labels.size, 3))
    print('in top k', true_labels.size, true_labels.shape)
    print('in top k', type(true_labels))
    print('in top k', scores.size, scores.shape)
    print('in top k', type(scores))
    one_hot_labels[np.arange(true_labels.size), true_labels.astype(int)] = 1
    class_true_labels = one_hot_labels[:, class_index].flatten()
    
    # Compute k as the total number of positive instances for the class
    k = int(np.sum(class_true_labels))
    
    # Ensure k is a valid number
    if k == 0:
        raise ValueError("There are no positive instances in the true labels for the specified class.")
    
    # Proceed with top-k accuracy calculation
    top_k_indices = np.argsort(class_scores)[::-1][:k]  # Get indices of top-k scores
    true_positives = np.sum(class_true_labels[top_k_indices])  # Count true positives among top-k
    
    top_k_accuracy = true_positives / k  # Calculate top-k accuracy
    
    return top_k_accuracy

"""
Manually calculate the mse and r^2.
"""
def calculate_metric_with_sklearn(logits: np.ndarray, labels: np.ndarray):
    labels = labels.squeeze()
    logits = logits.squeeze()
    metrics = [top_k_accuracy_multidimensional(logits, labels, i) for i in range(logits.shape[-1])]
    return {
        "acceptor topk acc" : metrics[1],
        "donor topk acc": metrics[2],
        "avg topk acc": np.mean(metrics[1:])
    }

"""
Compute metrics used for huggingface trainer.
"""
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    print(logits.shape, labels.shape)
    return calculate_metric_with_sklearn(logits, labels)



def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    set_seed(training_args)
    # load tokenizer
    if training_args.model_type == 'rnalm':
        tokenizer = EsmTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=True,
            trust_remote_code=True,
        )
    elif training_args.model_type in ['rna-fm','rnabert','rnamsm','splicebert-human510','splicebert-ms510','splicebert-ms1024','utrbert-3mer','utrbert-4mer','utrbert-5mer','utrbert-6mer','utr-lm-mrl','utr-lm-te-el']:
        tokenizer = OpenRnaLMTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=True,
            trust_remote_code=True,
        )
    elif training_args.model_type == 'structRFM':
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=True,
            trust_remote_code=True,
        )

    if "InstaDeepAI" in model_args.model_name_or_path:
        tokenizer.eos_token = tokenizer.pad_token
    if 'mer' in training_args.token_type:
        data_args.kmer=int(training_args.token_type[0])
    # define datasets and data collator
    train_dataset = SupervisedDataset(tokenizer=tokenizer, args=training_args,
                                     data_path=os.path.join(data_args.data_path, data_args.data_train_path), 
                                      kmer=data_args.kmer)
    val_dataset = SupervisedDataset(tokenizer=tokenizer, args=training_args,
                                     data_path=os.path.join(data_args.data_path, data_args.data_val_path), 
                                     kmer=data_args.kmer)
    test_dataset = SupervisedDataset(tokenizer=tokenizer, args=training_args,
                                     data_path=os.path.join(data_args.data_path, data_args.data_test_path), 
                                     kmer=data_args.kmer)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer,args=training_args)
    print(f'# train: {len(train_dataset)},val:{len(val_dataset)},test:{len(test_dataset)}')

    # load model
    if training_args.model_type == 'rnalm':
        if training_args.train_from_scratch:
            print('Train from scratch')
            config = RnaLmConfig.from_pretrained(model_args.model_name_or_path,
                num_labels=train_dataset.num_labels,
                problem_type="single_label_classification",
                token_type=training_args.token_type,
                attn_implementation=training_args.attn_implementation,
                )
            print(config)
            model =  RnaLmForNucleotideLevel(
                config,
                tokenizer=tokenizer,
                )
        else:
            print('Loading rnalm model')
            print(train_dataset.num_labels)
            model =  RnaLmForNucleotideLevel.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                num_labels=train_dataset.num_labels,
                trust_remote_code=True,
                problem_type="single_label_classification",
                token_type=training_args.token_type,
                attn_implementation=training_args.attn_implementation,
                )
    elif training_args.model_type == 'rna-fm':      
        print(training_args.model_type)
        print(f'Loading {training_args.model_type} model')
        model = RnaFmForNucleotideLevel.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            num_labels=train_dataset.num_labels,
            trust_remote_code=True,
            problem_type="single_label_classification",
            token_type=training_args.token_type,
            tokenizer=tokenizer,
        )     
    elif training_args.model_type == 'rnabert':      
        print(training_args.model_type)
        print(f'Loading {training_args.model_type} model')
        model = RnaBertForNucleotideLevel.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            num_labels=train_dataset.num_labels,
            trust_remote_code=True,
            problem_type="single_label_classification",
            token_type=training_args.token_type,
            tokenizer=tokenizer,
        )     
    elif training_args.model_type == 'rnamsm':
        print(training_args.model_type)
        print(f'Loading {training_args.model_type} model')
        model = RnaMsmForNucleotideLevel.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            num_labels=train_dataset.num_labels,
            trust_remote_code=True,
            problem_type="single_label_classification",
            token_type=training_args.token_type,
            tokenizer=tokenizer,
        )        
    elif 'splicebert' in training_args.model_type:
        print(training_args.model_type)
        print(f'Loading {training_args.model_type} model')
        model = SpliceBertForNucleotideLevel.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            num_labels=train_dataset.num_labels,
            trust_remote_code=True,
            problem_type="single_label_classification",
            token_type=training_args.token_type,
            tokenizer=tokenizer,
        )       
    elif 'utrbert' in training_args.model_type:
        print(training_args.model_type)
        print(f'Loading {training_args.model_type} model')
        model = UtrBertForNucleotideLevel.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            num_labels=train_dataset.num_labels,
            trust_remote_code=True,
            problem_type="single_label_classification",
            token_type=training_args.token_type,
            tokenizer=tokenizer,
        )  
    elif 'utr-lm' in training_args.model_type:
        print(training_args.model_type)
        print(f'Loading {training_args.model_type} model')
        model = UtrLmForNucleotideLevel.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            num_labels=train_dataset.num_labels,
            trust_remote_code=True,
            problem_type="single_label_classification",
            token_type=training_args.token_type,
            tokenizer=tokenizer,
        )     
    elif 'structRFM' in training_args.model_type:
        print(training_args.model_type)
        print(f'Loading {training_args.model_type} model')
        model = structRFMForNucleotideLevel.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            num_labels=train_dataset.num_labels,
            trust_remote_code=True,
            problem_type="single_label_classification",
            # token_type=training_args.token_type, # TODO
            tokenizer=tokenizer,
        )
    if training_args.freeze_base:
        # only freeze encoder
        for para in model.bert.encoder.parameters():
            para.requires_grad = False
    print(f'{training_args.model_type}')
    print(model)
    os.makedirs(training_args.output_dir, exist_ok=True)
    with open(os.path.join(training_args.output_dir, f'{training_args.model_type}_network.txt'), 'w') as fp:
        print(model, file=fp)
    total_para = sum([p.numel() for p in model.parameters()])
    train_para = sum([p.numel() for p in model.parameters() if p.requires_grad])
    print(f'freeze_base: {training_args.freeze_base}, train/total={train_para}/{total_para}')

    # define trainer
    trainer = transformers.Trainer(model=model,
                                   tokenizer=tokenizer,
                                   args=training_args,
                                   compute_metrics=compute_metrics,
                                   train_dataset=train_dataset,
                                   eval_dataset=val_dataset,
                                   data_collator=data_collator,
                                   callbacks=[early_stopping],
                                   )
    trainer.train()

    if training_args.save_model:
        trainer.save_state()
        #safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)

    # get the evaluation results from trainer
    # if training_args.eval_and_save_results:
        # test_results = trainer.evaluate(eval_dataset=test_dataset)

    results_path = os.path.join(training_args.output_dir, "results", training_args.run_name)
    
    os.makedirs(results_path, exist_ok=True)
    
    # predictions
    test_results = trainer.predict(test_dataset=test_dataset)
    with open(os.path.join(results_path, "test_results.json"), "w") as f:
        json.dump(test_results.metrics, f, indent=4)
    np.savez(
             os.path.join(results_path, "pred_data.npz"),
             pred=test_results.predictions,
             label=test_results.label_ids,
            )
        

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    train()
