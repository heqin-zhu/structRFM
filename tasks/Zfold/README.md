# Zfold
For RNA tertiary structure prediction.

> Derived from [trRosettaRNA](https://yanglab.qd.sdu.edu.cn/trRosettaRNA/).

- trRosettaRNA
    - inputs: MSA (rMSA), SS (SPOT-RNA)
- Zfold
    - inputs: MSA (rMSA), matrix feature (structRFM)

## Installation
```shell
conda env create -f Zfold_environment.yaml
conda activate Zfold
conda install -y pyrosetta
```
## Train
```shel
cd `src/Zfold/training`
# Modify run.sh
bash ./run.sh
```
## Pred and Fold
`pred_and_fold.py`

## Eval
`run_RNAeval.py`
