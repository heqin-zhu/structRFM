import os
import re
import glob
import shutil
import subprocess
from pathlib import Path

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm


from Bio import SeqIO

from BPfold.util.RNA_kit import read_fasta, write_fasta, read_SS


def fasta_to_a3m(fasta_file, a3m_file):
    """
    Convert FASTA MSA to A3M format using HH-suite's reformat.pl or Python parsing.
    """
    try:

        # Option 1: Use HH-suite's reformat.pl (recommended)
        subprocess.run(
            ["reformat.pl", "fas", "a3m", fasta_file, a3m_file],
            check=True, capture_output=True,
            env=os.environ,
        )
    except FileNotFoundError:
        # Option 2: Python-based conversion (simplified)
        records = list(SeqIO.parse(fasta_file, "fasta"))
        with open(a3m_file, "w") as f:
            for rec in records:
                # Remove gaps (except '-') and convert to uppercase
                seq = re.sub(r"[^A-Z-]", "-", str(rec.seq).upper())
                f.write(f">{rec.id}\n{seq}\n")
        print(f"Converted {fasta_file} to {a3m_file} using Python")


def pred_and_fold_all_with_MSA(name_seq_pairs, OUTPUT_DIR, PREFIX, data_dir, rerun=False, fast_test=False, LM_para_name='', Zfold_para_name='', window=100, stru_feat_type='SS', gpu='0', LM_name='structRFM', evo2_embedding_dir='evo2_embedding', num_cpu=16, run_fold=False):
    ''' Use MSA, used '''
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    PREDICT_PY = os.path.join(PREFIX, "src/Zfold/predict.py")
    FOLD_PY = os.path.join(PREFIX, "src/Zfold/fold.py")
    PARA_DIR = os.path.join(PREFIX, 'Zfold_test_paras')

    msa_exts = ['afa', 'a3m']

    if fast_test:
        name_seq_pairs = name_seq_pairs[:1]

    # Process each test sequence
    for (target_id, sequence) in name_seq_pairs:
        torch.cuda.empty_cache()
        L = len(sequence)
        # Step 1: Convert MSA, not used for Zfold
        # MSA_DIR = 'tmp'
        # fasta_msa = os.path.join(MSA_DIR, f"{target_id}.MSA.fasta")
        # a3m_msa = os.path.join(OUTPUT_DIR, f"{target_id}.a3m")
        # fasta_to_a3m(fasta_msa, a3m_msa)

        # Step 3: Run predict.py
        npz_file = os.path.join(OUTPUT_DIR, f"{target_id}.npz")
        for ext in msa_exts:
            msa_path = os.path.join(data_dir, 'msa', target_id+'.'+ext)
            if os.path.exists(msa_path):
                break
        else:
            print(f'[Warning] No msa found for {target_id}: {msa_path}')
            continue
        ss_path = os.path.join(data_dir, 'ss', target_id+'.ct')
        print(f"Processing {target_id} (length {L})...")
        print(sequence)
        print(msa_path, ss_path)
        predict_cmd = ["python", PREDICT_PY, "-i", msa_path, "-o", npz_file, '-ss', ss_path, '-ss_fmt', 'ct', "-gpu", gpu, "--cpu", "16", '--LM_para_name', LM_para_name, '--Zfold_para_name', Zfold_para_name, '--window', window, '--stru_feat_type', stru_feat_type, '--para_dir', PARA_DIR, '--LM_name', LM_name, '--evo2_embedding_dir', evo2_embedding_dir]
        predict_cmd = [str(i) for i in predict_cmd]
        print(' '.join(predict_cmd))
        try:
            if not os.path.exists(npz_file) or rerun:
                result = subprocess.run(predict_cmd, check=True, capture_output=True, text=True, env=os.environ)
                print(f"predict.py output: {result.stdout}")
        except subprocess.CalledProcessError as e:
            print(f"predict.py failed for {target_id}: {e.stderr}")
            continue
    for (target_id, sequence) in name_seq_pairs:
        # Step 4: Run fold.py
        npz_file = os.path.join(OUTPUT_DIR, f"{target_id}.npz")
        temp_dir = os.path.join(OUTPUT_DIR, f"temp_{target_id}")

        pdb_out = os.path.join(OUTPUT_DIR, f"{target_id}.pdb")
        os.makedirs(temp_dir, exist_ok=True)
        fasta_file = os.path.join(temp_dir, 'seq.fasta')
        write_fasta(fasta_file, [(target_id, sequence)])
        fold_cmd = ["python", FOLD_PY, "-npz", npz_file, "-fa", fasta_file, "-out", pdb_out, "-nm", "5", "-tmp", temp_dir, '--CPU', num_cpu]
        fold_cmd = [str(i) for i in fold_cmd]
        fold_cmd_str = ' '.join(fold_cmd)
        print(fold_cmd_str)
        with open(os.path.join(OUTPUT_DIR, target_id + '_fold.sh'), 'w') as fp:
            fp.write(fold_cmd_str)
        if run_fold:
            try:
                if not all([os.path.exists(os.path.join(temp_dir, f'model_{i}.pdb')) for i in range(1, 6)]) or rerun:
                    result = subprocess.run(fold_cmd, check=True, capture_output=True, text=True, env=os.environ)
                    print(f"fold.py output: {result.stdout}")
            except subprocess.CalledProcessError as e:
                print(f"fold.py failed for {target_id}: {e.stderr}")


def pred_and_fold_all(name_seq_pairs, OUTPUT_DIR, PREFIX, rerun=False, fast_test=False, LM_para_name='', Zfold_para_name='', num_cpu=16, run_fold=False):
    MODEL_DIR = os.path.join(PREFIX, "src/Zfold")
    PARA_DIR = os.path.join(PREFIX, 'Zfold_test_paras')
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    PREDICT_PY = os.path.join(MODEL_DIR, "predict.py")
    FOLD_PY = os.path.join(MODEL_DIR, "fold.py")

    if fast_test:
        name_seq_pairs = name_seq_pairs[:1]

    # Process each test sequence
    print('name seq pairs', name_seq_paris)
    for (target_id, sequence) in name_seq_pairs:
        torch.cuda.empty_cache()
        L = len(sequence)
        print(f"Processing {target_id} (length {L})...")
        print(sequence)

        # Step 1: Convert MSA, not used for Zfold
        # MSA_DIR = 'tmp'
        # fasta_msa = os.path.join(MSA_DIR, f"{target_id}.MSA.fasta")
        # a3m_msa = os.path.join(OUTPUT_DIR, f"{target_id}.a3m")
        # fasta_to_a3m(fasta_msa, a3m_msa)

        # Step 3: Run predict.py
        npz_file = os.path.join(OUTPUT_DIR, f"{target_id}.npz")
        print('predict.py')
        predict_cmd = ["python", PREDICT_PY, "--input", sequence, "--output", npz_file, "--para_dir", PARA_DIR, "--cpu", "16", '--LM_para_name', LM_para_name, '--Zfold_para_name', Zfold_para_name]
        try:
            if not os.path.exists(npz_file) or rerun:
                result = subprocess.run(predict_cmd, check=True, capture_output=True, text=True, env=os.environ)
                print(f"predict.py output: {result.stdout}")
        except subprocess.CalledProcessError as e:
            print(f"predict.py failed for {target_id}: {e.stderr}")
            continue

    for (target_id, sequence) in name_seq_pairs:
        # Step 4: Run fold.py
        npz_file = os.path.join(OUTPUT_DIR, f"{target_id}.npz")
        temp_dir = os.path.join(OUTPUT_DIR, f"temp_{target_id}")
        pdb_out = os.path.join(OUTPUT_DIR, f"{target_id}.pdb")
        os.makedirs(temp_dir, exist_ok=True)
        fasta_file = os.path.join(temp_dir, 'seq.fasta')
        write_fasta(fasta_file, [(target_id, sequence)])
        fold_cmd = ["python", FOLD_PY, "-npz", npz_file, "-fa", fasta_file, "-out", pdb_out, "-nm", "5", "-tmp", temp_dir, '--CPU', num_cpu]
        fold_cmd = [str(i) for i in fold_cmd]
        fold_cmd_str = ' '.join(fold_cmd)
        print(fold_cmd_str)
        with open(os.path.join(OUTPUT_DIR, target_id + '_fold.sh'), 'w') as fp:
            fp.write(fold_cmd_str)
        if run_fold:
            try:
                if not all([os.path.exists(os.path.join(temp_dir, f'model_{i}.pdb')) for i in range(1, 6)]) or rerun:
                    result = subprocess.run(fold_cmd, check=True, capture_output=True, text=True, env=os.environ)
                    print(f"fold.py output: {result.stdout}")
            except subprocess.CalledProcessError as e:
                print(f"fold.py failed for {target_id}: {e.stderr}")
