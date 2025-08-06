#!/bin/bash
current_dir=$(pwd)
for i in $(seq 1 1); do
    #file="save_model/ireseek_lm_s_y/oversampling/"
    file="save_model/ireseek_lm_s_y/undersampling/ens_3"
    path="$current_dir/$file"
    mkdir -p $path $path/log
    echo "Absolute path: $path"
    export RUN_NAME=$file
    sbatch --output="$path/log/train_%j.log" run_train.slurm --export=RUN_NAME
done
