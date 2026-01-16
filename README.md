<div align="center">

  <h1 align="center">structRFM: Structure-guided RNA Foundation Model</h1>

  <p align="center">
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/) [![PyTorch](https://img.shields.io/badge/PyTorch-2.0.1-ee4c2c.svg)](https://pytorch.org/)
  </p>

  <p align="center">
    <a href="https://www.biorxiv.org/content/early/2025/08/07/2025.08.06.668731">bioRxiv</a> |
    <a href="https://www.biorxiv.org/content/early/2025/08/07/2025.08.06.668731.full.pdf">PDF</a> |
    <a href="https://github.com/heqin-zhu/structRFM">GitHub</a> |
    <a href="https://pypi.org/project/structRFM">PyPI</a>
  </p>
</div>

<!-- vim-markdown-toc GFM -->

* [Overview](#overview)
* [Key Features](#key-features)
* [Quick Start](#quick-start)
    * [AutoModel and AutoTokenizer](#automodel-and-autotokenizer)
    * [Mannually Usage](#mannually-usage)
        * [Extracting Features](#extracting-features)
        * [Building Model and Tokenizer](#building-model-and-tokenizer)
    * [Pre-training and Fine-tuning](#pre-training-and-fine-tuning)
        * [Download sequence-structure dataset](#download-sequence-structure-dataset)
        * [Run Pre-training](#run-pre-training)
    * [Downstream Tasks Fine-tuning](#downstream-tasks-fine-tuning)
* [Acknowledgement](#acknowledgement)
* [LICENSE](#license)
* [Citation](#citation)

<!-- vim-markdown-toc -->

## Overview
structRFM is a **fully open-source** structure-guided RNA foundation model that integrates sequence and structural knowledge through innovative pre-training strategies. By leveraging 21 million sequence-structure pairs and a novel Structure-guided Masked Language Modeling (SgMLM) approach, structRFM achieves state-of-the-art performance across a broad spectrum of RNA structural and functional inference tasks, setting new benchmarks for reliability and generalizability.

<div align="center">
<img src="images/Fig1.png", width="800">
<sub>Figure: Overview of architecture and downstream applications</sub>
</div>

## Key Features
- **Structure-Guided Pre-Training**: SgMLM strategy dynamically balances sequence-level and structure-level masking, capturing base-pair interactions without task-specific biases.
- **Multi-Source Structure Ensemble**: MUSES (Multi-source ensemble of secondary structures) integrates thermodynamics-based, probability-based, and deep learning-based predictors to mitigate annotation biases.
- **Versatile Feature Output**: Generates classification-level, sequence-level, and pairwise matrix features to support sequence-wise, nucleotide-wise, and structure-wise tasks.
- **State-of-the-Art Performance**: Archieves state-of-the-art performances on zero-shot, secondary structure prediction, tertiary structure prediction, function prediction tasks.
- **Zero-Shot Capability**: Ranks top 4 in zero-shot homology classification across Rfam and ArchiveII datasets, with strong secondary structure prediction without labeled data.
- **Long RNA Handling**: Overlapping sliding window strategy enables high-accuracy classification of long non-coding RNAs (lncRNAs) up to 3,000 nt.
- **Fully Open Resources**: 21M sequence-structure dataset, pre-trained models, and fine-tuned checkpoints are publicly available for the research community.

## Quick Start
### AutoModel and AutoTokenizer
Install package: `pip install transformers`

```python
import os

from transformers import AutoModel, AutoTokenizer

model_path = 'heqin-zhu/structRFM'
# model_path = os.getenv('structRFM_checkpoint')

model = AutoModel.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# single sequence
seq = 'GUCCCAACUCUUGCGGGGAGGGAU'
inputs = tokenizer(seq, return_tensors="pt")
outputs = model(**inputs)
print('>>> single seq, length:', len(seq))
for k, v in outputs.items():
    print(k, v.shape)
print(outputs.last_hidden_state.shape)

# batch mode
seqs = ["GUCCCAA", 'AGUGUUG', 'AUGUAGUTCUN']
inputs = tokenizer(
             seqs,
             add_special_tokens=True,
             max_length=512,
             padding='max_length',
             truncation=True,
             return_tensors='pt'
        )
outputs = model(**inputs) # note that the output sequential features are padded to max-length
print('>>> batch seqs, batch:', len(seqs))
for k, v in outputs.items():
    print(k, v.shape)

'''
>>> single seq, length: 24
last_hidden_state torch.Size([1, 24, 768])
pooler_output torch.Size([1, 768])
torch.Size([1, 24, 768])
>>> batch seqs, batch: 3
last_hidden_state torch.Size([3, 512, 768])
pooler_output torch.Size([3, 768])
'''
```

### Mannually Usage
1. Install packages
```shell
pip install transformers structRFM BPfold
```
2. Download and decompress pretrained structRFM (~300 MB).
```shell
wget https://github.com/heqin-zhu/structRFM/releases/latest/download/structRFM_checkpoint.tar.gz
tar -xzf structRFM_checkpoint.tar.gz
```
3. Set environment varible `structRFM_checkpoint`.
```shell
export structRFM_checkpoint=PATH_TO_CHECKPOINT # modify ~/.bashrc for permanent setting
```

#### Extracting Features
```python
import os

from structRFM.infer import structRFM_infer

from_pretrained = os.getenv('structRFM_checkpoint')
model_paras = dict(max_length=514, dim=768, layer=12, num_attention_heads=12)
model = structRFM_infer(from_pretrained=from_pretrained, **model_paras)

seq = 'AGUACGUAGUA'

print('seq len:', len(seq))
feat_dic = model.extract_feature(seq)
for k, v in feat_dic.items():
    print(k, v.shape)

'''
seq len: 11
cls_feat torch.Size([768])
seq_feat torch.Size([11, 768])
mat_feat torch.Size([11, 11])
'''
```

#### Building Model and Tokenizer
```python3
import os

from structRFM.model import get_structRFM
from structRFM.data import preprocess_and_load_dataset, get_mlm_tokenizer

from_pretrained = os.getenv('structRFM_checkpoint')

tokenizer = get_mlm_tokenizer(max_length=514)
model = get_structRFM(dim=768, layer=12, num_attention_heads=12, from_pretrained=from_pretrained, pretrained_length=None, max_length=514, tokenizer=tokenizer)
```

### Pre-training and Fine-tuning
**Requirements**
- python3.8+
- anaconda

**Installation**
0. Clone GitHub repo.
```shell
git clone https://github.com/heqin-zhu/structRFM.git
cd structRFM
```
1. Create and activate conda environment.
```shell
conda env create -f structRFM_environment.yaml
conda activate structRFM
```
2. Download and decompress pretrained structRFM (~300 MB).
```shell
wget https://github.com/heqin-zhu/structRFM/releases/latest/download/structRFM_checkpoint.tar.gz
tar -xzf structRFM_checkpoint.tar.gz
```
3. Set environment varible `structRFM_checkpoint`.
```shell
export structRFM_checkpoint=PATH_TO_CHECKPOINT # modify ~/.bashrc for permanent setting
```

#### Download sequence-structure dataset
The pretrianing sequence-structure dataset is constructed using RNAcentral and BPfold. We filter sequences with a length limited to 512, resulting about 21 millions sequence-structure paired data. It can be downloaded at [Zenodo](https://doi.org/10.5281/zenodo.16754363) (4.5 GB).

Or use huggingface to load datasets:
```python
# pip install datasets
from datasets import load_dataset
dataset = load_dataset("heqin-zhu/structRFM-dataset")
```

#### Run Pre-training
- Modify variables `USER_DIR` and `PROGRAM_DIR` in `scripts/run.sh`,
- Specify `DATA_PATH` and `run_name` in the following command,

Then run:
```bash
bash scripts/run.sh --batch_size 96 --epoch 100 --lr 0.0001 --tag mlm --mlm_structure --max_length 514 --model_scale base --data_path DATA_PATH --run_name structRFM_512
```

For more information, run `python3 main.py -h`.

### Downstream Tasks Fine-tuning
Download all data (3.7 GB) and checkpoints (2.2 GB) from [Zenodo](https://doi.org/10.5281/zenodo.16754363), and then place them into corresponding folder of each task.

- Zero-shot inference
    - [Zero-shot homology classfication](tasks/zeroshot)
    - [Zero-shot secondary structure prediction](tasks/zeroshot)
- Structure prediction
    - [Secondary structure prediction](tasks/seqcls_ssp)
    - [Tertiary structure prediction](tasks/Zfold)
- Function prediction
    - [ncRNA classification](tasks/seqcls_ssp)
    - [Splice site prediction](tasks/splice_site_prediction)
    - [IRES identification](IRES)

## Acknowledgement
We appreciate the following open-source projects for their valuable contributions:
- [RNAcentral](https://rnacentral.org)
- [BPfold](https://github.com/heqin-zhu/BPfold)
- [RNAErnie](https://github.com/CatIIIIIIII/RNAErnie)
- [trRosettaRNA](https://yanglab.qd.sdu.edu.cn/trRosettaRNA)
- [BEACON](https://github.com/terry-r123/RNABenchmark)
- [MXfold2](https://github.com/mxfold/mxfold2)

## LICENSE
[MIT LICENSE](LICENSE)

## Citation
If you find our work helpful, please cite our paper:

```bibtex
@article {structRFM,
    author = {Zhu, Heqin and Li, Ruifeng and Zhang, Feng and Tang, Fenghe and Ye, Tong and Li, Xin and Gu, Yujie and Xiong, Peng and Zhou, S Kevin},
    title = {A fully-open structure-guided RNA foundation model for robust structural and functional inference},
    elocation-id = {2025.08.06.668731},
    year = {2025},
    doi = {10.1101/2025.08.06.668731},
    publisher = {Cold Spring Harbor Laboratory},
    URL = {https://www.biorxiv.org/content/early/2025/08/07/2025.08.06.668731},
    journal = {bioRxiv}
}
```
