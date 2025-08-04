import os
import re
from glob import glob
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm
from Bio.Seq import Seq
from Bio.PDB import PDBParser
from biopandas.pdb import PandasPdb


def pdb2seq(pdb_path:str, merge_multi_chain:bool=True, model_idx:int=0)->str:
    '''
        If empty, return None
    '''
    chain_seq_pairs = []
    try:
        parser = PDBParser()
        structure = parser.get_structure('RNA_structure', pdb_path)
        for m_idx, model in enumerate(structure):
            if model_idx==m_idx:
                for chain_idx, chain in enumerate(model):
                    sequence = []
                    for res_idx, residue in enumerate(chain):
                        resname = residue.get_resname().strip()
                        if resname in ['A', 'DA']: # 'ADE'
                            sequence.append('A')
                        elif resname in ['U', 'DT']: # URA'
                            sequence.append('U')
                        elif resname in ['G', 'DG']: # GUA'
                            sequence.append('G')
                        elif resname in ['C', 'DC']: # CYT
                            sequence.append('C')
                        else:
                            # discard
                            print(f'Unknown nucleotide in {pdb_path}: model{m_idx}-chain{chain_idx}-residue{res_idx}: "{resname}", discarded')
                            break
                    else:
                        chain_seq_pairs.append((chain.id, ''.join(sequence)))
    except Exception as e:
        print(e, pdb_path, 'discarded')

    if len(chain_seq_pairs)==0:
        return None
    if len(chain_seq_pairs)>1:
        print(f'multi chains: {pdb_path}')
        print(chain_seq_pairs)
    if merge_multi_chain:
        seq = ''.join([s for chain, s in chain_seq_pairs])
    else:
        seq = chain_seq_pairs[0][1]
    return seq


def pdb2SS(pdb_path):
    ### TODO    To correct
    helix_segments = []
    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith('HELIX'):
                # 解析 HELIX 记录中的起始和结束残基
                helix_id = line[7:10].strip()
                start_chain = line[15]
                start_res = int(line[17:21].strip())
                end_chain = line[27]
                end_res = int(line[29:33].strip())
                helix_segments.append((start_chain, start_res, end_res))
    return helix_segments



def preprocess_pdb_df(df):
    df = df[df.atom_name=="C1'"].sort_values('residue_number').reset_index(drop = True)
    df = df[['residue_name', 'residue_number', 'x_coord', 'y_coord', 'z_coord']].copy()
    df.columns = ['resname', 'resid', 'x_1', 'y_1', 'z_1']
    df[['x_1', 'y_1', 'z_1']] =  df[['x_1', 'y_1', 'z_1']].astype(np.float32) 
    df['resid'] =  df['resid'].astype(np.int32)
    assert len(df['resid'].unique())  == (df['resid'].max()-df['resid'].min())+1
    return df


def pdb2df(
    pdb_path: Optional[str] = None,
    model_index: int=0, # TODO model_index = 0
    ) -> pd.DataFrame:
    """
    Read a PDB file, and return a Pandas DataFrame containing the atomic coordinates and metadata.

    Args:
        pdb_path (str, optional): Path to a local PDB file to read. Defaults to None.
        model_index (int, optional): Index of the model to extract from the PDB file, in case it contains multiple models. Defaults to 0.
    Returns:
        pd.DataFrame: A DataFrame containing the atomic coordinates and metadata, with one row per atom
    """
    atomic_df = PandasPdb().read_pdb(pdb_path)
    header = None
    atomic_df = atomic_df.get_model(model_index)
    if len(atomic_df.df["ATOM"]) == 0:
        raise ValueError(f"No model found for index: {model_index}")
    df = pd.concat([atomic_df.df["ATOM"], atomic_df.df["HETATM"]])
    df = preprocess_pdb_df(df)
    return df


def convert_pdb(pdb_paths, model_index=0):
    ext_label_df = []
    for pdb_path in tqdm(pdb_paths):
        try:
            # pdb_id = os.path.basename(pdb_path).split('.', 1)[0]
            pdb_id = pdb_path.split('/')[-2]
            ext_label_df_ = pdb2df(pdb_path, model_index)
            ext_label_df_['ID'] =  [f'{pdb_id}_{resid}' for resid in ext_label_df_['resid']]
            ext_label_df_['pdb_id'] =  pdb_id
            ext_label_df.append(ext_label_df_[['ID', 'resname', 'resid', 'x_1', 'y_1', 'z_1', 'pdb_id']]) # columns
        except Exception as e:
            print(f'Failed to read{pdb_path}')
            print('Error:' + str(e))
    ext_label_df = pd.concat(ext_label_df).reset_index(drop = True)
    ext_sequence_df = ext_label_df[["pdb_id", 'resname']].groupby("pdb_id", as_index = False).apply(lambda x: ''.join(x['resname']), include_groups=False).reset_index(drop = True)
    ext_sequence_df.columns = ["target_id", 'sequence'] # columns
    ext_label_df = ext_label_df.drop("pdb_id", axis = 1)
    return ext_label_df, ext_sequence_df


def extract_sequences(pdb_paths):
    df_lst = []
    for pdb_path in tqdm(pdb_paths):
        f = os.path.basename(pdb_path)
        if f.endswith('.pdb'):
            target_id = f[:f.rfind('.')]
            seq = pdb2seq(pdb_path)
            if seq is not None:
                df_lst.append({
                               'target_id': target_id,
                               'sequence': seq,
                             })
    return pd.DataFrame(df_lst)


def convert_pdb_from_kaggle():
    ext_dir = '/kaggle/input/stanford-ribonanza-rna-folding/rhofold_pdbs/rhofold_pdbs'
    pdb_paths = glob(f'{ext_dir}/*/*/*/*/*.pdb')
    print(len(pdb_paths)) # 124902
    ext_label_df, ext_sequence_df = convert_pdb(pdb_paths)
    ext_label_df.to_parquet(f'ext_ribonanza_labels.parquet')
    ext_sequence_df.to_parquet(f'ext_ribonanza_sequences.parquet')

    train_label_df = pd.read_csv("/kaggle/input/stanford-rna-3d-folding/train_labels.csv")
    display(train_label_df.head())
    '''
            ID  resname resid   x_1 y_1 z_1
    0   1SCL_A_1    G   1   13.760  -25.974001  0.102
    1   1SCL_A_2    G   2   9.310   -29.638000  2.669
    2   1SCL_A_3    G   3   5.529   -27.813000  5.878
    '''

    test_sequence_df = pd.read_csv("/kaggle/input/stanford-rna-3d-folding/test_sequences.csv")
    display(test_sequence_df.head())
    '''
        target_id   sequence    temporal_cutoff description all_sequences
    0   R1107   GGGGGCCACAGCAGAAGCGUUCACGUCGCAGCCCCUGUCAGCCAUU...   2022-05-28  CPEB3 ribozyme\nHuman\nhuman CPEB3 HDV-like ri...   >7QR4_1|Chain A|U1 small nuclear ribonucleopro...
    1   R1108   GGGGGCCACAGCAGAAGCGUUCACGUCGCGGCCCCUGUCAGCCAUU...   2022-05-27  CPEB3 ribozyme\nChimpanzee\nChimpanzee CPEB3 H...
    '''

    ## check data
    train_label_df["pdb_id"] = train_label_df["ID"].apply(lambda x: x.split("_")[0]+'_'+x.split("_")[1])
    check_df_train=train_label_df[train_label_df.pdb_id=='1RNK_A'].reset_index(drop=True)
    display(check_df_train.head())

    pdb_path='/kaggle/input/some-pdb-files-for-rna-competition/1rnk.pdb'
    check_df_mine=preprocess_pdb(pdb_path)
    display(check_df_mine.head())

    #Check sequences are the same
    print(all(check_df_train.resname == check_df_mine.resname))
    #Check coordinates are the same
    print(np.abs(check_df_train[['x_1','y_1','z_1']].values-check_df_mine[['x_1','y_1','z_1']].values).sum())

    import matplotlib.pyplot as plt
    alpha = 0.3
    coord=[check_df_train[['x_1','y_1','z_1']].values, check_df_mine[['x_1','y_1','z_1']].values]
    COLOR = ['red', 'blue', 'green', 'black', 'yellow', 'cyan', 'magenta']
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    # ax.clear()

    for j in range(len(coord)):
        x, y, z = coord[j][:, 0], coord[j][:, 1], coord[j][:, 2]
        ax.scatter(x, y, z, c=COLOR[j], s=30, alpha=alpha)
        ax.plot(x, y, z, color=COLOR[j], linewidth=1, alpha=alpha, label=f'{j}')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    src='../rfdiff_data/rfdiff_data_flat'
    # convert_pdb_from_kaggle()
    df = extract_sequences([os.path.join(src, f) for f in os.listdir(src)])
    df.to_csv('extract_rfdiff_seq.csv', index=False)
