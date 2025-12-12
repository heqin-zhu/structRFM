## seq_cls
CUDA_VISIBLE_DEVICES=7 nohup python3 run_seq_cls.py --device 'cuda:7' --model_name structRFM --LM_path ${structRFM_checkpoint} --batch_size 16 --output_dir outputs_seqcls/nRC_512 --max_seq_len 512 > log_nRC_512 2>&1 &
