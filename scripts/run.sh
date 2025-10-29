#!/bin/bash
source activate /root/miniconda3/envs/RNA3d # wrong: conda activate
conda env list

USER_DIR=/heqinzhu
PROGRAM_DIR=$USER_DIR/gitrepo/LLM/structRFM


if [[ $@ == *"--use_DDP"* ]]; then
  cmd="torchrun --nproc_per_node=2 $PROGRAM_DIR/main.py $@ "
else
  cmd="python3 $PROGRAM_DIR/main.py $@ "
fi

echo $cmd
# exec $cmd  # wrong
# exec bash -c "$cmd" # correct
eval $cmd  # correct
