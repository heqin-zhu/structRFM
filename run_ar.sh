source activate /root/miniconda3/envs/RNALM # wrong: conda activate

USER_DIR=/heqinzhu
PROGRAM_DIR=$USER_DIR/gitrepo/LLM/NucleoLM
DATA_DIR=$USER_DIR/gitrepo/LLM/RNAcentral/RNAcentral_BPfold_SS
OUT_DIR=$USER_DIR/runs

RUN_NAME=llama_ar_768x12_lr3_ep10
cmd="python3 $PROGRAM_DIR/main.py --run_name $OUT_DIR/$RUN_NAME --data_path $DATA_DIR --tag ar --max_length 514 --dim 768 --layer 12 --batch_size 128 --epoch 10 --lr 0.0003"
echo $cmd
exec $cmd
