# Zfold
For RNA tertiary structure prediction.

> Derived from [trRosettaRNA](https://yanglab.qd.sdu.edu.cn/trRosettaRNA/).


- trRosettaRNA
    - inputs: MSA (rMSA), SS (SPOT-RNA)
- Zfold
    - inputs: MSA (rMSA), matrix feature (structRFM)
    

## Installation
```shell
conda env create -f linux-cu102.yml
conda activate Zfold
conda install -y pyrosetta
```

## Train
```shel
cd `src/Zfold/training`
```
Modify `run.sh` and run `run.sh`

## Eval
Prepare data as `DATA_DIR` and `PREFIX` specified in `eval.py`, then run `eval.py`
