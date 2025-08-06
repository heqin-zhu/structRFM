#!/bin/bash
#SBATCH --job-name=process_data     # 设置作业名称
#SBATCH --output=/public2/home/ustc_zfw/IRES/save_model/log/process_%j.log
#SBATCH --ntasks=1                   # 设置任务数
#SBATCH --gres=gpu:1                 # 请求1个GPU
#SBATCH --time=90-24:00:00           # 设置最大运行时间为1小时
#SBATCH --partition=gpu              # 选择GPU节点的分区，具体分区名称取决于集群设置

python ensemble_auc_pth.py
