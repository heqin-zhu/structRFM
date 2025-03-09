# source activate /root/miniconda3/envs/RNA3d # wrong: conda activate
conda env list

# parse args
options=$(getopt -o b:e:l:t:p --long batch_size:,epoch:,lr:,tag:,print -- "$@")
eval set -- "$options"
# 提取选项和参数
while true; do
  case $1 in 
  	-b | --batch_size) shift; batch_size=$1 ; shift ;;
    -e | --epoch) shift; epoch=$1 ; shift ;;
    -l | --lr) shift; lr=$1 ; shift ;;
    -t | --tag) shift; tag=$1; shift ;;
    -p | --print) print=true; shift ;;
    --) shift ; break ;;
    *) echo "Invalid option: $1" exit 1 ;;
  esac
done

if [ -z "$batch_size" ]; then
    batch_size=64
fi
if [ -z "$epoch" ]; then
    epoch=1
fi
if [ -z "$lr" ]; then
    lr=0.0002
fi
if [ -z "$tag" ]; then
    tag="mlm"
    # echo "Error: tag is required"
    # exit 1
fi

USER_DIR=/heqinzhu
PROGRAM_DIR=$USER_DIR/gitrepo/LLM/NucleoLM
DATA_DIR=$USER_DIR/gitrepo/LLM/RNAcentral/RNAcentral_BPfold_SS
OUT_DIR=$USER_DIR/runs

RUN_NAME="${tag}_768x12_lr${lr}"

cmd="python3 $PROGRAM_DIR/main.py --run_name $OUT_DIR/$RUN_NAME --data_path $DATA_DIR --tag ${tag} --max_length 514 --dim 768 --layer 12 --batch_size ${batch_size} --epoch ${epoch} --lr ${lr} --mlm_structure --resume_from_checkpoint 2>&1 | tee -a $OUT_DIR/$RUN_NAME/log_train"

if [ "$print" = true ]; then
    echo $cmd
fi
    echo $cmd

# exec $cmd  # wrong
# exec bash -c "$cmd" # correct
eval $cmd  # correct
