## seq_cls
nohup python3 run_seq_cls.py --device 'cuda:0' --model_name structRFM --LM_path /public/share/heqinzhu_share/structRFM/structRFM_checkpoint --batch_size 16 --output_dir outputs_seqcls/tmp --use_automodelforseqcls > log 2>&1 &
# nohup python3 run_seq_cls.py --device 'cuda:0' --model_name structRFM --LM_path /public/share/heqinzhu_share/structRFM/structRFM_checkpoint --batch_size 16 --output_dir outputs_seqcls/tmp > log 2>&1 &
