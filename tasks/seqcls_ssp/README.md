# Downstream tasks
> Modified from [RNAErnie_baselines](https://github.com/CatIIIIIIII/RNAErnie_baselines)

Download data and prepare it as follows:

Data directory structure
- data
    - seq_cls
        - nRC
    - ssp
        - ...

## structRFM for ncRNA classification
```shell
nohup python3 run_seq_cls.py --device 'cuda:0' --model_name structRFM --LM_path /public/share/heqinzhu_share/structRFM/structRFM_checkpoint --batch_size 16 --output_dir outputs_seqcls/tmp > log_seqcls 2>&1 &
nohup python3 run_seq_cls.py --device 'cuda:0' --model_name structRFM --LM_path /public/share/heqinzhu_share/structRFM/structRFM_checkpoint --batch_size 16 --output_dir outputs_seqcls/tmp --use_automodelforseqcls > log_seqcls_auto 2>&1 &
```

## structRFM for secondary structure prediction
```shell
nohup python3 run_ss_pred.py  --model_name structRFM --LM_path /public/share/heqinzhu_share/structRFM/structRFM_checkpoint --batch_size 1 --task_name bpRNA1m --lr 0.0005 --output_dir outputs_ssp/bpRNA1m_structRFM  > log_ssp_bpRNA1m_structRFM 2>&1 &
nohup python3 run_ss_pred.py  --model_name structRFM --LM_path /public/share/heqinzhu_share/structRFM/structRFM_checkpoint --batch_size 1 --task_name RNAStrAlign --lr 0.0005 --output_dir outputs_ssp/RNAStrAlign_structRFM  > log_ssp_RNAStrAlign_structRFM 2>&1 &
```
