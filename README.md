<p align="center">

  <h1 align="center">Structure-guided pretraining of RNA language model empowers robust structural and functional prediction</h1>
  <p align="center">
    <a href="https://heqin-zhu.github.io/"><strong>Heqin Zhu</strong></a>
    ·
    <a href="https://fenghetan9.github.io/"><strong>Fenghe Tang</strong></a>
    ·
    <a href="https://scholar.google.com/citations?user=mlTXS0YAAAAJ"><strong>Quan Quan</strong></a>
    ·
    <a href="https://bme.ustc.edu.cn/2023/0918/c28132a612449/page.htm"><strong>Ke Chen</strong></a>
    ·
    <a href="https://bme.ustc.edu.cn/2023/0322/c28131a596069/page.htm"><strong>Peng Xiong*</strong></a>
    ·
    <a href="https://scholar.google.com/citations?user=8eNm2GMAAAAJ"><strong>S. Kevin Zhou*</strong></a>
  </p>
  <!--<h2 align="center">Submitted</h2>-->
  <div align="center">
    <img src="images/BPfold.png", width="800">
  </div>
  <p align="center">
    <a href="https://www.nature.com/articles/s41467-025-60048-1">Paper</a> | 
    <a href="https://www.nature.com/articles/s41467-025-60048-1.pdf">PDF</a> |
    <a href="https://heqin-zhu.github.io/files/poster_BPfold.pdf">poster</a> |
    <a href="https://github.com/heqin-zhu/BPfold">GitHub</a> |
    <a href="https://pypi.org/project/BPfold">PyPI</a>
    
  </p>
</p>



<!-- vim-markdown-toc GFM -->

* [Introduction](#introduction)
* [Installation](#installation)
    * [Requirements](#requirements)
    * [Instructions](#instructions)
* [Usage](#usage)
    * [Base pair motif library](#base-pair-motif-library)
    * [BPfold Prediction](#bpfold-prediction)
* [Reproduction](#reproduction)
* [Acknowledgement](#acknowledgement)
* [LICENSE](#license)
* [Citation](#citation)

<!-- vim-markdown-toc -->

## Introduction
Deep learning methods have demonstrated great performance for RNA secondary structure prediction. However, generalizability is a common unsolved issue on unseen out-of-distribution RNA families, which hinders further improvement of the accuracy and robustness of deep learning methods. Here we construct a base pair motif library that enumerates the complete space of locally adjacent three-neighbor base pair and records the thermodynamic energy of corresponding base pair motifs through _de novo_ modeling of tertiary structures, and we further develop a deep learning approach for RNA secondary structure prediction, named BPfold, which learns relationship between RNA sequence and the energy map of base pair motif. Experiments on sequence-wise and family-wise datasets have demonstrated the great superiority of BPfold compared to other state-of-the-art approaches in accuracy and generalizability. We hope this work contributes to integrating physical priors and deep learning methods for the further discovery of RNA structures and functionalities.


## Installation
### Requirements
- python3.8+
- anaconda

### Instructions
0. Clone this repo
```shell
git clone git@github.com:heqin-zhu/BPfold.git
cd BPfold
```
1. Create and activate BPfold environment.
```shell
conda env create -f BPfold_environment.yaml
conda activate BPfold
```
2. Install BPfold
```shell
pip3 install BPfold --index-url https://pypi.org/simple
```
3. Download [model_predict.tar.gz](https://github.com/heqin-zhu/BPfold/releases/latest/download/model_predict.tar.gz) in [releases](https://github.com/heqin-zhu/BPfold/releases) and decompress it.
```shell
wget https://github.com/heqin-zhu/BPfold/releases/latest/download/model_predict.tar.gz
tar -xzf model_predict.tar.gz
```
4. Optional: Download datasets [BPfold_data.tar.gz](https://github.com/heqin-zhu/BPfold/releases/latest/download/BPfold_data.tar.gz) in [releases](https://github.com/heqin-zhu/BPfold/releases) and decompress them.
```shell
wget https://github.com/heqin-zhu/BPfold/releases/latest/download/BPfold_data.tar.gz
tar -xzf BPfold_data.tar.gz 
```

## Usage
### Base pair motif library
The base pair motif library is publicly available in [releases](https://github.com/heqin-zhu/BPfold/releases), which contains the `motif`:`energy` pairs. The motif is represented as `sequence`\_`pairIdx`\_`pairIdx`\-`chainBreak` where pairIdx is 0-indexed, and the energy is a reference score of statistical and physical thermodynamic energy.
For instance, `CAAAAUG_0_6-3 -49.7835` represents motif `CAAAAUG` has a known pair `C-G` whose indexes are `0` and `6`, with chainBreak lying at position `3`.

>[!NOTE]
>The base pair motif library can be used as thermodynamic priors in other models.

For an input RNA sequence `seq`, the base pair motif energy matrix `mat` can be directly obatined as follows:
```python3
from BPfold.util.base_pair_motif import BPM_energy

BPM = BPM_energy()

seq = 'AUGCGUAGTa'
# default, recommended, normed to [-1, 1], BPfold used, shape 2xLxL
mat = BPM.get_energy(seq)

# origin energy, value may be -50.3, 49.7, ..., shape 1xLxL
mat2 = BPM.get_energy(seq, normalize_energy=False, dispart_outer_inner=False)
```


### BPfold Prediction
Use BPfold to predict RNA secondary structures.
Args:
- `--checkpoint_dir`: required, specify checkpoint dir path.
- `--seq`: specify one or more input RNA sequences.
- `--input`: specify input file of RNA seqs in format of `.fasta`(multiple seqs are supported), `.bpseq`, `.ct`, or `.dbn`.
- `--output`: output dir (will be created automatically), default `BPfold_results`.
- `--out_type`: out format of RNA secondary structures, can be `.csv`, `.bpseq`, `.ct`, or `.dbn`, default `.csv`
Here are some examples:
```shell
BPfold --checkpoint_dir PATH_TO_CHECKPOINT_DIR --seq GGUAAAACAGCCUGU AGUAGGAUGUAUAUG --output BPfold_results
BPfold --checkpoint_dir PATH_TO_CHECKPOINT_DIR --input examples/examples.fasta --out_type csv # (multiple sequences are supported)
BPfold --checkpoint_dir PATH_TO_CHECKPOINT_DIR --input examples/URS0000D6831E_12908_1-117.bpseq
```

<details>

<summary>Example of BPfold prediction</summary>

Here are the outputs after running `BPfold --checkpoint_dir model_predict --input examples/examples.fasta --out_type bpseq`:
```txt
>> Welcome to use "BPfold" for predicting RNA secondary structure!
Loading model_predict/BPfold_1-6.pth
Loading model_predict/BPfold_2-6.pth
Loading model_predict/BPfold_3-6.pth
Loading model_predict/BPfold_4-6.pth
Loading model_predict/BPfold_5-6.pth
Loading model_predict/BPfold_6-6.pth
[      1] saved in "BPfold_results/5s_Shigella-flexneri-3.bpseq", CI=0.973
CUGGCGGCAGUUGCGCGGUGGUCCCACCUGACCCCAUGCCGAACUCAGAAGUGAAACGCCGUAGCGCCGAUGGUAGUGUGGGGUCUCCCCAUGCGAGAGUAGGGAACUGCCAG
(((((((.....((((((((.....((((((.............))))..))....)))))).)).((.((....((((((((...))))))))....)).))...)))))))
[      2] saved in "BPfold_results/URS0000D6831E_12908_1-117.bpseq", CI=0.915
UUAUCUCAUCAUGAGCGGUUUCUCUCACAAACCCGCCAACCGAGCCUAAAAGCCACGGUGGUCAGUUCCGCUAAAAGGAAUGAUGUGCCUUUUAUUAGGAAAAAGUGGAACCGCCUG
......((((((....(((((.......)))).)(((.((((.((......))..))))))).................))))))..(((......)))..................
Confidence indexes are saved in "BPfold_results_confidence_TIMESTR.yaml"
Program Finished!
```

</details>

For more help information, please run command `BPfold -h` to see.

## Reproduction
For reproduction of all the quantitative results, we provide the predicted secondary structures and model parameters of BPfold in experiments. You can **directly downalod** the predicted secondary structures by BPfold *or* **use BPfold v0.2.0** with trained parameters to predict these secondary structures, and then **evaluate** the predicted results.

**Directly download**
```shell
wget https://github.com/heqin-zhu/BPfold/releases/download/v0.2/BPfold_test_results.tar.gz
tar -xzf BPfold_test_results.tar.gz
```
**Use BPfold**
1. Download the checkpoints of BPfold: [BPfold_reproduce.tar.gz](https://github.com/heqin-zhu/BPfold/releases/download/v0.2/model_reproduce.tar.gz).
```shell
wget https://github.com/heqin-zhu/BPfold/releases/download/v0.2/model_reproduce.tar.gz
tar -xzf model_reproduce.tar.gz
```
2. Install BPfold v<=0.2.4.
```shell
pip install BPfold==0.2.4
```
3. Use BPfold to predict RNA sequences in test datasets.

**Evaluate**
```shell
BPfold_eval --gt_dir BPfold_data --pred_dir BPfold_test_results
```

After running above commands for evaluation, you will see the following outputs:

<details>

<summary>Outputs of evaluating BPfold</summary>

```txt
Time used: 29s
[Summary] eval_BPfold_test_results.yaml
 Pred/Total num: [('PDB_test', 116, 116), ('Rfam12.3-14.10', 10791, 10791), ('archiveII', 3966, 3966), ('bpRNA', 1305, 1305), ('bpRNAnew', 5401, 5401)]
-------------------------len>600-------------------------
dataset         & num   & INF   & F1    & P     & R    \\
Rfam12.3-14.10  & 64    & 0.395 & 0.387 & 0.471 & 0.333\\
archiveII       & 55    & 0.352 & 0.311 & 0.580 & 0.242\\
------------------------len<=600-------------------------
dataset         & num   & INF   & F1    & P     & R    \\
PDB_test        & 116   & 0.817 & 0.814 & 0.840 & 0.801\\
Rfam12.3-14.10  & 10727 & 0.696 & 0.690 & 0.662 & 0.743\\
archiveII       & 3911  & 0.829 & 0.827 & 0.821 & 0.843\\
bpRNA           & 1305  & 0.670 & 0.658 & 0.599 & 0.770\\
bpRNAnew        & 5401  & 0.655 & 0.647 & 0.604 & 0.723\\
---------------------------all---------------------------
dataset         & num   & INF   & F1    & P     & R    \\
PDB_test        & 116   & 0.817 & 0.814 & 0.840 & 0.801\\
Rfam12.3-14.10  & 10791 & 0.694 & 0.689 & 0.660 & 0.741\\
archiveII       & 3966  & 0.823 & 0.820 & 0.818 & 0.834\\
bpRNA           & 1305  & 0.670 & 0.658 & 0.599 & 0.770\\
bpRNAnew        & 5401  & 0.655 & 0.647 & 0.604 & 0.723\\
```

</details>

## Acknowledgement
We appreciate the following open source projects:
- [UFold](https://github.com/uci-cbcl/UFold)
- [vigg_ribonanza](https://github.com/autosome-ru/vigg_ribonanza/)
- [e2efold](https://github.com/ml4bio/e2efold)

## LICENSE
[MIT LICENSE](LICENSE)

## Citation
If you use our code, please cite the following paper:
```bibtex
@article{BPfold,
    title = {Deep generalizable prediction of {RNA} secondary structure via base pair motif energy},
    author = {Zhu, Heqin and Tang, Fenghe and Quan, Quan and Chen, Ke and Xiong, Peng and Zhou, S. Kevin},
    volume = {16},
    issn = {2041-1723},
    url = {https://doi.org/10.1038/s41467-025-60048-1},
    doi = {10.1038/s41467-025-60048-1},
    number = {1},
    journal = {Nature Communications},
    month = jul,
    year = {2025},
    pages = {5856},
}
```
