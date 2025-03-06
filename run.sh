source activate /root/miniconda3/envs/RNA3d # wrong: conda activate

USER_DIR=/heqinzhu
PROGRAM_DIR=$USER_DIR/gitrepo/LLM/NucleoLM
DATA_DIR=$USER_DIR/gitrepo/LLM/RNAcentral/RNAcentral_BPfold_SS
OUT_DIR=$USER_DIR/runs

RUN_NAME=bert_768x12_lr3_ep50
TAG='mlm'
cd $PROGRAM_DIR
cmd="python3 -m src.NucleoLM.run_pretrain --run_name $OUT_DIR/$RUN_NAME --batch_size 160 -g 0 --phase train --lr 0.001 --dim 256 --depth 9 --nfolds 1  --Lmax 500 --epoch $epoch --data_dir $DATA_DIR --dropout 0.1 --save_freq 2 --error_band 1"
cmd="python3 main.py --run_name $OUT_DIR/$RUN_NAME --data_path $DATA_DIR --tag $TAG --max_length 512 --dim 768 --layer 12 --batch_size 32 --epoch 50 --lr 0.0003"
echo $cmd
exec $cmd
