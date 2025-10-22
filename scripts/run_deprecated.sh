source activate /root/miniconda3/envs/RNA3d # wrong: conda activate
conda env list

# parse args
options=$(getopt -o b:e:L:l:d:t:n:c:p:s --long batch_size:,epoch:,max_length:,lr:,data_path:,tag:,run_name:,resume_from_checkpoint:,print,mlm_structure -- "$@")
eval set -- "$options"
# 提取选项和参数
while true; do
  case $1 in 
  	-b | --batch_size) shift; batch_size=$1 ; shift ;;
    -e | --epoch) shift; epoch=$1 ; shift ;;
    -L | --max_length) shift; max_length=$1 ; shift ;;
    -l | --lr) shift; lr=$1 ; shift ;;
    -d | --data_path) shift; data_path=$1 ; shift ;;
    -t | --tag) shift; tag=$1; shift ;;
    -n | --run_name) shift; run_name=$1; shift ;;
    -c | --resume_from_checkpoint) shift; resume_from_checkpoint=$1; shift ;;
    -p | --print) print=true; shift ;;
    -s | --mlm_structure) mlm_structure=true; shift ;;
    --) shift ; break ;;
    *) echo "Invalid option: $1" exit 1 ;;
  esac
done

USER_DIR=/heqinzhu
PROGRAM_DIR=$USER_DIR/gitrepo/LLM/structRFM
OUT_DIR=$USER_DIR/runs
DEFAULT_DATA_PATH=$USER_DIR/gitrepo/LLM/RNAcentral/RNAcentral_512_MUSES_connects.csv


if [ -z "$batch_size" ]; then
    batch_size=128
fi
if [ -z "$epoch" ]; then
    epoch=30
fi
if [ -z "$max_length" ]; then
    max_length=514
fi
if [ -z "$lr" ]; then
    lr=0.0001
fi
if [ -z "$data_path" ]; then
    data_path=$DEFAULT_DATA_PATH
fi
if [ -z "$tag" ]; then
    tag="mlm"
    # echo "Error: tag is required"
    # exit 1
fi
if [ -z "$run_name" ]; then
    run_name="structRFM_768x12"
fi
if [ -z "$resume_from_checkpoint" ]; then
    resume_from_checkpoint="not_exist"
fi
mlm_stru_flag=""
mlm_stru_str="_nostru"
if [ "$mlm_structure" = true ]; then
    mlm_stru_flag="--mlm_structure"
    mlm_stru_str="_stru"

fi

echo "mlm $mlm_stru_str"

RUN_NAME="${run_name}_${tag}_lr${lr}${mlm_stru_str}"

cmd="python3 $PROGRAM_DIR/main.py --run_name $OUT_DIR/$RUN_NAME --data_path $data_path --tag ${tag} --max_length ${max_length} --dim 768 --layer 12 --batch_size ${batch_size} --epoch ${epoch} --lr ${lr} ${mlm_stru_flag} --resume_from_checkpoint ${resume_from_checkpoint} 2>&1 | tee -a $OUT_DIR/$RUN_NAME/log_train"

if [ "$print" = true ]; then
    echo $cmd
fi

# exec $cmd  # wrong
# exec bash -c "$cmd" # correct
eval $cmd  # correct
