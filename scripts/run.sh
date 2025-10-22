#!/bin/bash
source activate /root/miniconda3/envs/RNA3d # wrong: conda activate
conda env list

USER_DIR=/heqinzhu
PROGRAM_DIR=$USER_DIR/gitrepo/LLM/structRFM

cmd="python3 $PROGRAM_DIR/main.py $@ "

echo $cmd
# exec $cmd  # wrong
# exec bash -c "$cmd" # correct
eval $cmd  # correct
