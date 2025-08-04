# Modified from trRosettaRNA

import os
from collections import defaultdict

import numpy as np
from Bio import PDB
from scipy.spatial.distance import cdist


from .pdb2seq import pdb2seq

ATOM_TYPES = ['P', 'OP1', 'OP2', "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'", 'N1', 'N2', 'C2', 'O2',
             'O4', 'N3', 'C4', 'N4', 'C5', 'C6', 'C8', 'O6', 'N9', 'N7', 'N8', 'N6']
def load_residue_map(path=None):
    if path is None:
        CUR_DIR = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(CUR_DIR, 'non_res2res_list') # non_res2res_list_demo
    residue_map = dict((line.split()[0], line.split()[1]) for line in open(path))
    return residue_map
RESIDUE_MAP = load_residue_map()


def get_atom_positions_pdb(pdb_path, seq=None, strict_residue_map=True, strict_residue_continuousness=False):
    """
    :param pdb_path:
    :param seq: full seq(including linkers), for checking seq consistency
    :param strict_residue_map:
    """
    if not strict_residue_map:
        _collected = defaultdict(lambda: 'N')
        _collected.update(RESIDUE_MAP)
    else:
        _collected = RESIDUE_MAP
    p = PDB.PDBParser()
    structure = p.get_structure('', pdb_path)[0]
    chains = list(structure.get_chains())
    # L = sum(chain.child_list[-1].id[1] for chain in chains) # Wrong when idx 0-indexed
    L = sum(len(chain.child_list) for chain in chains)
    xyzs = {}
    for atom in ATOM_TYPES:
        xyzs[atom] = np.nan * np.zeros((L, 3))

    chain_idx_lst = []
    pdb_seq_lst = []
    for chain_idx, chain in enumerate(chains):
        residues_chain = [r for r in chain.child_list if not PDB.is_aa(r)]
        for res in residues_chain:
            ## res_idx may be incontinuous
            res_idx = res.id[1] - 1
            if strict_residue_continuousness:
                if res_idx != len(pdb_seq_lst):
                    raise Exception(f'[Error] Residue not continuous in {pdb_path}: res_idx{res_idx}!=seq_idx{len(pdb_seq_lst)}')

            ## save xyz
            for atom in ATOM_TYPES:
                try:
                    atom_name = atom
                    coord = res[atom_name].coord
                    xyzs[atom][len(pdb_seq_lst)] = coord
                except KeyError as e:
                    print(f'{e}. In {pdb_path}:chain{chain_idx}-res{res_idx}-seq{len(pdb_seq_lst)}, ignored')
                    continue
            pdb_seq_lst.append(_collected[res.resname.strip()])
            chain_idx_lst.append(chain_idx)
    pdb_seq = ''.join(pdb_seq_lst)
    if seq and seq!=pdb_seq:
        raise Exception(f'[Error] Seq mismatch in {pdb_path} \nprovided: {seq}\npdb_seq : {pdb_seq}')
    return xyzs, chain_idx_lst, pdb_seq


def calcu_labels(pdb_path: str, seq: str=None):
    """
    :param pdb_path: path to PDB file
    :param seq: seq, for checking seq consistency
    """
    xyzs_all, chain_idx, seq = get_atom_positions_pdb(pdb_path, seq)

    len_ = len(seq)
    n9_mask = (np.array(seq) == 'A') | (np.array(seq) == 'G')
    xyzs_all['N1/9'] = np.where(n9_mask, xyzs_all['N9'], xyzs_all['N1'])
    xyzs_all['C4/2'] = np.where(n9_mask, xyzs_all['C2'], xyzs_all['C4'])
    labels = ['CiNj', 'PiNj', "C3'", "C1'", "C4", "P", 'N1', 'all', 'contact']
    inter_labels = {k: np.nan * np.zeros((len_, len_)) for k in labels}
    for atm in ["C3'", "C1'", "C4'", "P"]:
        inter_labels[atm] = cdist(xyzs_all[atm], xyzs_all[atm])
    inter_labels['N1'] = cdist(xyzs_all['N1/9'], xyzs_all['N1/9'])
    inter_labels['C4'] = cdist(xyzs_all['C4/2'], xyzs_all['C4/2'])
    inter_labels['CiNj'] = cdist(xyzs_all["C4'"], xyzs_all['N1/9'])
    inter_labels['PiNj'] = cdist(xyzs_all["P"], xyzs_all['N1/9'])

    for i in range(len_):
        for j in range(i + 1, len_):
            coords_i = np.stack([xyzs_all[atm][i] for atm in xyzs_all], axis=0)
            coords_j = np.stack([xyzs_all[atm][j] for atm in xyzs_all], axis=0)
            allatm_dist = np.nanmin(cdist(coords_i, coords_j))
            inter_labels['all'][i, j] = allatm_dist
            inter_labels['contact'][i, j] = int(allatm_dist < 8)
            inter_labels['all'][j, i] = inter_labels['all'][i, j]
            inter_labels['contact'][j, i] = inter_labels['contact'][i, j]
    return inter_labels, xyzs_all, chain_idx


def pdb2label(dest_dir, pdb_path, seq=None):
    if seq is None:
        seq = pdb2seq(pdb_path)
        if seq is None:
            print(f'Error when obtaining seq from {pdb_path}, discarded.')
            return
    f = os.path.basename(pdb_path)
    name = f[:f.rfind('.')]
    label_dir = os.path.join(dest_dir, 'label')
    xyz_dir = os.path.join(dest_dir, 'xyz')
    chain_dir = os.path.join(dest_dir, 'chain')
    for d in [label_dir, xyz_dir, chain_dir]:
        os.makedirs(d, exist_ok=True)
    inter_labels, xyzs_all, chain_idx = calcu_labels(pdb_path, seq)
    label_path = os.path.join(label_dir, name+'.npz')
    xyz_path = os.path.join(xyz_dir, name+'.npz')
    np.savez_compressed(label_path, **inter_labels)
    np.savez_compressed(xyz_path, **xyzs_all)
    return inter_labels, xyzs_all, chain_idx


if __name__ == '__main__':
    pdb_path = 'example/seq.pdb'
    fasta_path = 'example/seq.fasta'
    dest = 'tmp'
    seq = open(fasta_path).readlines()[1].strip()
    inter_labels, xyzs_all, chain_idx = pdb2label(dest, pdb_path, seq)
    for k, v in inter_labels.items():
        print(k, v.shape)
    # for k, v in xyzs_all.items():
    #     print(k, v.shape)
