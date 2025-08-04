# DATA_DIR="/data/heqinzhu/gitrepo/npz_3633/label"
DATA_DIR="/public2/home/heqinzhu/gitrepo/RNA/RNA3d/Zfold/data/processed_npz/npz_3633/label"

nohup python3 train.py $DATA_DIR  .runs/tmp --warning --gpu 4 --init_lr 0.0005 --crop_size 200 --stru_feat_type LM --LM_path $structRFM_checkpoint  >> log 2>&1 &
