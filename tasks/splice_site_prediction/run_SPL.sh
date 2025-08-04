#!/bin/bash

# Modified from https://github.com/terry-r123/RNABenchmark
set -x

data_root=./data
MODEL_PATH=${structRFM_checkpoint}

MODEL_TYPE='structRFM'

token='single'
kmer=-1 # -1 means not using kmer

nproc_per_node=1
model_max_length=514
seed=666
data=''


task='SPL'
data_file_train=train.csv; data_file_val=val.csv; data_file_test=test.csv
batch_size=32
lr=3e-5


DATA_PATH=${data_root}/${task}

### begin cmd
gpu_device="6"
freeze_flag="--freeze_base False"
run_name='tmp'
master_port=$(shuf -i 10000-45000 -n 1)
# master_port=10001
echo "Using port $master_port for communication."
OUTPUT_PATH=./output_${task}/$run_name
EXEC_PREFIX="env CUDA_VISIBLE_DEVICES=$gpu_device torchrun --nproc_per_node=$nproc_per_node --master_port=$master_port"
echo "running $run_name:${MODEL_PATH}, ${freeze_flag}"
nohup ${EXEC_PREFIX} \
train.py \
    --model_name_or_path $MODEL_PATH \
    --data_path  $DATA_PATH/$data \
    --data_train_path ${data_file_train} --data_val_path ${data_file_val} --data_test_path ${data_file_test}   \
    --run_name $run_name \
    --model_max_length ${model_max_length} \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --learning_rate ${lr} \
    --num_train_epochs 30 \
    --fp16 \
    --save_steps 400 \
    --output_dir ${OUTPUT_PATH} \
    --evaluation_strategy steps \
    --eval_steps 200 \
    --warmup_steps 50 \
    --logging_steps 200 \
    --overwrite_output_dir True \
    --log_level info \
    --seed ${seed} \
    --token_type ${token} \
    --model_type ${MODEL_TYPE} \
    $freeze_flag  > log_tmp 2>&1 &
