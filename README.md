<p align="center">

  <h1 align="center">A fully open structure-guided RNA foundation model for robust structural and functional inference</h1>
  <p align="center">
    <a href="https://heqin-zhu.github.io/"><strong>Heqin Zhu</strong></a>
    ·
    <strong>Ruifeng Li</strong>
    ·
    <!-- <a href="https://zaixizhang.github.io/"><strong>Zaixi Zhang</strong></a> · -->
    <strong>Feng Zhang</strong>
    ·
    <a href="https://fenghetan9.github.io/"><strong>Fenghe Tang</strong></a><br>
    <strong>Tong Ye</strong>
    ·
    <strong>Xin Li</strong>
    ·
    <strong>Yunjie Gu</strong>
    ·
    <a href="https://bme.ustc.edu.cn/2023/0322/c28131a596069/page.htm"><strong>Peng Xiong*</strong></a>
    ·
    <a href="https://scholar.google.com/citations?user=8eNm2GMAAAAJ"><strong>S. Kevin Zhou*</strong></a>
  </p>
  <h2 align="center">Submitted</h2>
  <div align="center">
    <img src="images/Fig1.png", width="800">
  </div>
  <p align="center">
    <a href="https://www.biorxiv.org/content/early/2025/08/07/2025.08.06.668731">bioRxiv</a> |
    <a href="https://www.biorxiv.org/content/early/2025/08/07/2025.08.06.668731.full.pdf">PDF</a> |
    <a href="https://github.com/heqin-zhu/structRFM">GitHub</a> |
    <a href="https://pypi.org/project/structRFM">PyPI</a>
  </p>
</p>


<!-- vim-markdown-toc GFM -->

* [Abstract](#abstract)
* [Installation](#installation)
    * [Requirements](#requirements)
    * [Instructions](#instructions)
* [Pretraining](#pretraining)
    * [Download sequence-structure dataset](#download-sequence-structure-dataset)
    * [Run pretraining](#run-pretraining)
    * [Extract RNA sequence features](#extract-rna-sequence-features)
* [Downstream Tasks](#downstream-tasks)
* [Acknowledgement](#acknowledgement)
* [LICENSE](#license)
* [Citation](#citation)

<!-- vim-markdown-toc -->

## Abstract
RNA language models have achieved strong performance across diverse downstream tasks by leveraging large-scale sequence data. However, RNA function is fundamentally shaped by its hierarchical structure, making the integration of structural information into pretraining essential. Existing methods often depend on noisy structural annotations or introduce task-specific biases, limiting model generalizability. Here, we introduce structRFM, a structure-guided RNA foundation model that is pretrained by implicitly incorporating large-scale base pairing interactions and sequence data via a dynamic masking ratio to balance nucleotide-level and structure-level masking. structRFM learns joint knowledge of sequential and structural data, producing versatile representations-including classification-level, sequence-level, and pairwise matrix features-that support broad downstream adaptations. structRFM ranks top models in zero-shot homology classification across fifteen biological language models, and sets new benchmarks for secondary structure prediction, achieving F1 scores of 0.873 on ArchiveII and 0.641 on bpRNA-TS0 dataset. structRFM further enables robust and reliable tertiary structure prediction, with consistent improvements in both 3D accuracy and extracted 2D structures. In functional tasks such as internal ribosome entry site identification, structRFM achieves a 49\% performance gain. These results demonstrate the effectiveness of structure-guided pretraining and highlight a promising direction for developing multi-modal RNA language models in computational biology.


## Installation
### Requirements
- python3.8+
- anaconda

### Instructions
0. Clone this repo.
```shell
git clone git@github.com:heqin-zhu/structRFM.git
cd structRFM
```
1. Create and activate conda environment.
```shell
conda env create -f environment.yaml
conda activate structRFM
```
2. Install structRFM.
```shell
pip3 install structRFM
```
3. Download and decompress pretrained structRFM (305 M).
```shell
wget https://github.com/heqin-zhu/structRFM/releases/latest/download/structRFM_checkpoint.tar.gz
tar -xzf model_predict.tar.gz
```
4. Set environment varible `structRFM_checkpoint`.
```shell
export structRFM_checkpoint=PATH_TO_CHECKPOINT # modify ~/.bashrc for permanent setting
```

## Pretraining

### Download sequence-structure dataset
The pretrianing sequence-structure dataset is constructed using RNAcentral and BPfold. We filter sequences with a length limited to 512, resulting about 21 millions sequence-structure paired data. It can be downloaded at [Zenodo](https://doi.org/10.5281/zenodo.16754363) (4.5 GB).

### Run pretraining
Modify variables `USER_DIR`, `PROGRAM_DIR`, `DATA_DIR`, and `OUT_DIR` in `run.sh`, then run:

```bash
bash ./run.sh --print --batch_size 128 --epoch 100 --lr 0.0001 --tag mlm --mlm_structure
```

### Extract RNA sequence features

<details>

<summary>demo.py</summary>

```python
import os

from structRFM.infer import structRFM_infer

from_pretrained = os.getenv('structRFM_checkpoint')
model = structRFM_infer(from_pretrained=from_pretrained, max_length=514)

seq = 'AGUACGUAGUA'
output_attentions = True

print('seq len:', len(seq))
# (1+L+1)x 768,  [CLS] seq [SEP]
features, attentions = model.extract_feature(seq, return_all=True, output_attentions=output_attentions)

# feat  tuple: layer=12, tuple[i]: batch x L x hidden_dim(=768)
last_feat = features[-1]

# classification feature, 1x768
cls_feat = last_feat[0,:] # 1x768
# sequence feature, Lx768
feat1d = last_feat[1:-1, :] # Lx768
# matrix_feature, LxL
feat2d = feat1d @ feat1d.transpose(-1,-2) # LxL

print('classification feature:', cls_feat.shape)
print('sequence feature:', feat1d.shape)
print('matrix feature:', feat2d.shape)

# atten   tuple: layer=12, tuple[i]: batch x head(=12) x L x L
# remove special tokens
attentions = tuple([atten[:, :, 1:-1, 1:-1] for atten in attentions])
print('attentions', len(attentions), attentions[0].shape)
```

</details>

## Downstream Tasks
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
