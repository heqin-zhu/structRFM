#!/bin/bash
source activate /root/miniconda3/envs/RNA3d # wrong: conda activate
conda env list

python run.py "$@"
cmd="python3 -m src.structRFM.pretrain $@ "

echo $cmd
# exec $cmd  # wrong
# exec bash -c "$cmd" # correct
eval $cmd  # correct
