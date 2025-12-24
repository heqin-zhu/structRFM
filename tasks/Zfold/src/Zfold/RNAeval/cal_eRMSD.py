'''
输入：pred.pdb + gt.pdb
        |
        v
[1] barnaba.annotate() 解析两边 RNA，得到 labels + seq
        |
        v
[2] 直接算 εRMSD：bb.ermsd(pred, gt)
   | 成功 -> 输出 ermsd, status=ok
   |
   v 失败
[3] 尝试“裁 pred 多余”：
    在 pred_seq 里找 gt_seq 的子序列位置 pick_pred
    - 找到 -> 用 mdtraj 按 labels 映射裁 pred -> bb.ermsd(trim_pred, gt)
             成功 -> status=trimmed_pred_ok
    - 找不到 -> 进入下一步
        |
        v
[4] 若 pred 比 gt 短：尝试“裁 gt 多余”
    在 gt_seq 里找 pred_seq 的子序列位置 pick_gt
    - 找到 -> 裁 gt -> bb.ermsd(pred, trim_gt)
             成功 -> status=trimmed_gt_ok
    - 找不到 -> 进入下一步
        |
        v
[5] 全局序列比对（Needleman–Wunsch）
    得到对齐 pairs -> 只保留字符相同的位置 (keep_pred, keep_gt)
    -> 裁 pred & gt -> bb.ermsd(aln_pred, aln_gt)
       成功 -> status=align_trim_ok (aligned_len=...)
       失败 -> status=fail
'''


import os
from pathlib import Path
import csv

import numpy as np
import barnaba as bb
import mdtraj as md

# -----------------------------
# barnaba: labels + seq
# -----------------------------
def _barnaba_labels_and_seq(pdb_path: str):
    """
    ann[2] labels: e.g. 'G_49_0'
    返回 (labels, seq)
    """
    ann = bb.annotate(pdb_path)
    labels = ann[2]
    seq = "".join(x.split("_")[0] for x in labels)
    return labels, seq


def _parse_label(label: str):
    """
    'G_49_0' -> (base='G', resseq=49, chain_tag=0)
    注意：label 的第三段通常可当作“链/片段标识”，至少能区分不同链的 residue 编号重复情况
    """
    base, num, tag = label.split("_")
    return base, int(num), int(tag)


# -----------------------------
# subsequence mapping (allow insertions)
# -----------------------------
def _find_subsequence_indices(super_seq: str, sub_seq: str):
    i = 0
    pick = []
    for ch in sub_seq:
        while i < len(super_seq) and super_seq[i] != ch:
            i += 1
        if i >= len(super_seq):
            return None
        pick.append(i)
        i += 1
    return pick


# -----------------------------
# mdtraj residue mapping using (resSeq, base)
# -----------------------------
def _md_base(resname: str):
    r = resname.strip().upper()
    if r in {"A", "RA", "ADE"}: return "A"
    if r in {"C", "RC", "CYT"}: return "C"
    if r in {"G", "RG", "GUA"}: return "G"
    if r in {"U", "RU", "URA"}: return "U"
    return None


def _build_mdtraj_lookup(traj: md.Trajectory):
    """
    建立映射： (chain_idx, resSeq, base) -> residue.index
    用 chain_idx（mdtraj 的 chain 序号）减少重复冲突。
    """
    lookup = {}
    for r in traj.topology.residues:
        b = _md_base(r.name)
        if b is None:
            continue
        # mdtraj Residue 通常有 resSeq（PDB residue number）
        resseq = getattr(r, "resSeq", None)
        if resseq is None:
            # 兜底：没有 resSeq 就不用这个映射法了
            continue
        key = (r.chain.index, int(resseq), b)
        # 如果重复出现，保留第一次（也可以换成列表，这里先简单处理）
        if key not in lookup:
            lookup[key] = r.index
    return lookup


def _slice_pdb_by_barnaba_positions(pdb_path: str, keep_pos, cache_dir: str, tag: str):
    """
    用 barnaba 的 labels（含 resSeq 和链tag）来精确定位 mdtraj residue，再裁剪 PDB。
    这样不会受 mdtraj/barnaba “计数不一致”影响。
    """
    traj = md.load(pdb_path)
    labels, _ = _barnaba_labels_and_seq(pdb_path)

    # 解析要保留的 labels
    keep_labels = [labels[i] for i in keep_pos]

    # mdtraj lookup
    lookup = _build_mdtraj_lookup(traj)

    keep_residue_indices = set()
    # barnaba label 的 tag 不一定等于 mdtraj chain.index，但通常顺序一致。
    # 我们优先尝试 (chain_tag, resseq, base)，失败则在所有 chain 上搜索 resseq+base。
    for lab in keep_labels:
        base, resseq, chain_tag = _parse_label(lab)

        # 1) 先用 chain_tag 当作 chain.index 试
        key = (chain_tag, resseq, base)
        ridx = lookup.get(key, None)

        # 2) 如果没找到，退化：忽略 chain，在任何 chain 找第一个匹配
        if ridx is None:
            for (ci, rs, b), rr in lookup.items():
                if rs == resseq and b == base:
                    ridx = rr
                    break

        if ridx is None:
            raise ValueError(f"Cannot map barnaba label to mdtraj residue: {lab}")

        keep_residue_indices.add(ridx)

    keep_atom_indices = [
        atom.index for atom in traj.topology.atoms
        if atom.residue.index in keep_residue_indices
    ]
    trimmed = traj.atom_slice(np.array(keep_atom_indices, dtype=int))

    out_pdb = os.path.join(cache_dir, f"{tag}_{Path(pdb_path).stem}.pdb")
    trimmed.save_pdb(out_pdb)
    return out_pdb


def _to_float_ermsd(val):
    if isinstance(val, np.ndarray):
        if val.size == 1:
            return float(val.ravel()[0])
        return float(np.mean(val))
    if isinstance(val, (list, tuple)) and len(val) == 1:
        return float(val[0])
    return float(val)


# -----------------------------
# global alignment (Needleman–Wunsch)
# -----------------------------
def _global_align_mapping(seq_a: str, seq_b: str, match=1, mismatch=-1, gap=-1):
    n, m = len(seq_a), len(seq_b)
    dp = np.zeros((n + 1, m + 1), dtype=int)
    bt = np.zeros((n + 1, m + 1), dtype=np.int8)  # 0 diag, 1 up, 2 left

    for i in range(1, n + 1):
        dp[i, 0] = dp[i - 1, 0] + gap
        bt[i, 0] = 1
    for j in range(1, m + 1):
        dp[0, j] = dp[0, j - 1] + gap
        bt[0, j] = 2

    for i in range(1, n + 1):
        ai = seq_a[i - 1]
        for j in range(1, m + 1):
            bj = seq_b[j - 1]
            s_diag = dp[i - 1, j - 1] + (match if ai == bj else mismatch)
            s_up = dp[i - 1, j] + gap
            s_left = dp[i, j - 1] + gap
            best = max(s_diag, s_up, s_left)
            dp[i, j] = best
            bt[i, j] = 0 if best == s_diag else (1 if best == s_up else 2)

    i, j = n, m
    pairs = []
    while i > 0 or j > 0:
        move = bt[i, j]
        if i > 0 and j > 0 and move == 0:
            pairs.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif i > 0 and (j == 0 or move == 1):
            i -= 1
        else:
            j -= 1

    pairs.reverse()
    return pairs




def epsilon_RMSD(pred_pdb_path, gt_pdb_path, cache_dir=".RNAeval_cache"):
    os.makedirs(cache_dir, exist_ok=True)

    # 0) sequences/labels (barnaba truth)
    try:
        pred_labels, pred_seq = _barnaba_labels_and_seq(pred_pdb_path)
        gt_labels, gt_seq = _barnaba_labels_and_seq(gt_pdb_path)
    except Exception as e0:
        return np.nan, "fail", f"annotate failed: {e0}"

    # 1) direct
    try:
        val = bb.ermsd(gt_pdb_path, pred_pdb_path)
        return _to_float_ermsd(val), "ok", ""
    except Exception as e1:
        msg1 = str(e1)
        print(e1)

    # 2) trim pred extras (gt subseq of pred)
    pick_pred = _find_subsequence_indices(pred_seq, gt_seq)
    if pick_pred is not None and len(pred_seq) >= len(gt_seq):
        try:
            pred_trim_pdb = _slice_pdb_by_barnaba_positions(
                pred_pdb_path, keep_pos=pick_pred, cache_dir=cache_dir,
                tag=f"trim_pred_to_gt_{Path(gt_pdb_path).stem}"
            )
            val = bb.ermsd(gt_pdb_path, pred_trim_pdb, )
            return _to_float_ermsd(val), "trimmed_pred_ok", ""
        except Exception as e2:
            msg2 = str(e2)
            print(e2)
    else:
        msg2 = "gt not subsequence of pred"

    # 3a) if pred shorter: trim gt extras (pred subseq of gt)
    pick_gt = None
    if len(pred_seq) < len(gt_seq):
        pick_gt = _find_subsequence_indices(gt_seq, pred_seq)
        if pick_gt is not None:
            try:
                gt_trim_pdb = _slice_pdb_by_barnaba_positions(
                    gt_pdb_path, keep_pos=pick_gt, cache_dir=cache_dir,
                    tag=f"trim_gt_to_pred_{Path(pred_pdb_path).stem}"
                )
                val = bb.ermsd( gt_trim_pdb, pred_pdb_path,)
                return _to_float_ermsd(val), "trimmed_gt_ok", ""
            except Exception as e3:
                print(e3)
                msg3 = str(e3)
        else:
            msg3 = "pred shorter than gt AND pred not subsequence of gt"
    else:
        msg3 = "not pred shorter"

    # 3b) global alignment: trim both to common matched positions
    try:
        pairs = _global_align_mapping(pred_seq, gt_seq, match=1, mismatch=-1, gap=-1)

        keep_pred = []
        keep_gt = []
        for i_p, i_g in pairs:
            # 更稳：只保留字符相同的对齐位点
            if pred_seq[i_p] == gt_seq[i_g]:
                keep_pred.append(i_p)
                keep_gt.append(i_g)

        if len(keep_pred) < 5:
            note = f"direct:{msg1} | trim_pred:{msg2} | trim_gt:{msg3} | align:too few matched"
            return np.nan, "fail", note

        pred_aln_pdb = _slice_pdb_by_barnaba_positions(
            pred_pdb_path, keep_pos=keep_pred, cache_dir=cache_dir,
            tag=f"align_pred_{Path(pred_pdb_path).stem}_to_{Path(gt_pdb_path).stem}"
        )
        gt_aln_pdb = _slice_pdb_by_barnaba_positions(
            gt_pdb_path, keep_pos=keep_gt, cache_dir=cache_dir,
            tag=f"align_gt_{Path(gt_pdb_path).stem}_to_{Path(pred_pdb_path).stem}"
        )

        val = bb.ermsd(gt_aln_pdb, pred_aln_pdb, )
        return _to_float_ermsd(val), "align_trim_ok", f"aligned_len={len(keep_pred)}"

    except Exception as e4:
        note = f"direct:{msg1} | trim_pred:{msg2} | trim_gt:{msg3} | align:{e4}"
        return np.nan, "fail", note


# -----------------------------
# batch run + csv
# -----------------------------
if __name__ == "__main__":
    DATASET = "CASP15_RNAs"   # CASP15_RNAs    20_RNA_Puzzles
    GT_dir   = f"/public2/home/aleck/RNA3d_eval/RNA3d_eval/gt/{DATASET}"
    PRED_dir = f"/public2/home/aleck/RNA3d_eval/RNA3d_eval/pred/{DATASET}"

    out_csv = os.path.abspath(f"{DATASET}_ermsd.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    rows = []

    for fname in sorted(os.listdir(PRED_dir)):
        if not fname.endswith(".pdb"):
            continue
        pred_pdb = os.path.join(PRED_dir, fname)
        gt_pdb   = os.path.join(GT_dir, fname)

        val, status, note = epsilon_RMSD(pred_pdb, gt_pdb)
        rows.append({
            "puzzle_id": fname.replace(".pdb", ""),
            "gt_pdb": fname,
            "pred_pdb": pred_pdb.split("/")[-1],
            "ermsd": val,
            "status": status,
            "note": note
        })

        print(fname, status, val, note)

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["puzzle_id", "gt_pdb", "pred_pdb", "ermsd", "status", "note"]
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved results to {out_csv}")
