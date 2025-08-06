#!/bin/bash
#SBATCH --job-name=IRES
#SBATCH --output=/public2/home/ustc_zfw/IRES/log/train_%j.log
#SBATCH --ntasks=1
#SBATCH --gres=gpu:NVIDIAA10:1
#SBATCH --partition=gpu
#SBATCH --time=10-24:00:00

python train.py --run_name "$RUN_NAME" --batch_size 128 --epoch 100 --lr 0.002
