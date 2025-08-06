## seq_cls
# nohup python3 run_seq_cls.py --device 'cuda:0' --model_name structRFM --LM_path ${structRFM_checkpoint} --batch_size 16 --output_dir outputs_seqcls/tmp --use_automodelforseqcls > log 2>&1 &
# nohup python3 run_seq_cls.py --device 'cuda:0' --model_name structRFM --LM_path ${structRFM_checkpoint} --batch_size 16 --output_dir outputs_seqcls/tmp > log 2>&1 &
# nohup python3 run_seq_cls.py --device 'cuda:6' --model_name structRFM --LM_path ${structRFM_checkpoint} --batch_size 16 --output_dir outputs_seqcls/lncRNA_H_512 --dataset lncRNA_H_uni --max_seq_len 512 > log_lncRNA_H_uni_512 2>&1 &
CUDA_VISIBLE_DEVICES=6,7 nohup python3 run_seq_cls.py --device 'cuda' --model_name structRFM --LM_path ${structRFM_checkpoint} --batch_size 16 --output_dir outputs_seqcls/lncRNA_H_1024 --dataset lncRNA_H_uni --max_seq_len 1024 > log_lncRNA_H_uni_1024 2>&1 &
