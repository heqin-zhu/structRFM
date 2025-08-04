nohup python3 run_ss_pred.py  --model_name structRFM --LM_path $structRFM_checkpoint --batch_size 1 --task_name RNAStrAlign --lr 0.0005 --output_dir outputs_ssp/RNAStrAlign_structRFM  > log_ssp_RNAStrAlign_structRFM 2>&1 &

nohup python3 run_ss_pred.py  --model_name structRFM --LM_path $structRFM_checkpoint --batch_size 1 --task_name bpRNA1m --lr 0.0005 --output_dir outputs_ssp/bpRNA1m_structRFM  > log_ssp_bpRNA1m_structRFM 2>&1 &
