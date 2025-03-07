source activate /root/miniconda3/envs/RNA3d # wrong: conda activate

USER_DIR=/heqinzhu
PROGRAM_DIR=$USER_DIR/gitrepo/LLM/NucleoLM
DATA_DIR=$USER_DIR/gitrepo/LLM/RNAcentral/RNAcentral_BPfold_SS
OUT_DIR=$USER_DIR/runs

RUN_NAME=llama_ar_768x12_lr5_ep50
cmd="python3 $PROGRAM_DIR/main.py --run_name $OUT_DIR/$RUN_NAME --data_path $DATA_DIR --tag ar --max_length 512 --dim 768 --layer 12 --batch_size 128 --epoch 50 --lr 0.0005"
echo $cmd
exec $cmd
