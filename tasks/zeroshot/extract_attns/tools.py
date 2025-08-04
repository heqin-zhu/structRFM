import numpy as np
import torch
from torch import Tensor
import os, sys
eps = 1e-6

def symmetrize(x):
    "Make layer symmetric in final two dimensions, used for contact prediction."
    return x + x.transpose(-1, -2)

def apc(x):
    "Perform average product correct, used for contact prediction."
    a1 = x.sum(-1, keepdims=True)
    a2 = x.sum(-2, keepdims=True)
    a12 = x.sum((-1, -2), keepdims=True)

    avg = a1 * a2
    avg.div_(a12)  # in-place to reduce memory
    normalized = x - avg
    return normalized


def contacts_to_array(contacts: list, max_seq_len=None):
    if max_seq_len is None:
        max_seq_len = 0
        for contact in contacts:
            if max_seq_len < contact.shape[-1]: max_seq_len = contact.shape[-1]
    array = np.empty((len(contacts), max_seq_len, max_seq_len))
    array.fill(-100)
    for i, contact in enumerate(contacts):
        seq_len = contact.shape[-1]
        array[i, :seq_len, :seq_len] = np.array(contact)

    return array

def attnmap_to_cont(attentions: Tensor, use_sigmoid=True):
    # b, n, h, l, l
    contacts = apc(symmetrize(attentions))
    if use_sigmoid:
        return torch.sigmoid(contacts)
    else:
        return contacts

def batch_metric(
    test_data,
    layer, head,
    step: float = 0.001,
    
):
    T = np.arange(0, 1 + step, step)[None, :]  # threshold value [1,1001]
    tp = 0; fn = 0; fp = 0; tn = 0

    for batch in test_data:
        rna_id = batch["rna_id"]
        predictions = batch["pred"][layer, head]
        targets = batch["contacts"]
        missing_nt_index = batch["missing_nt_index"]


        if predictions.dim() == 3:
            predictions = predictions.squeeze()     # [BS, L, L]-> [L,L]
        if targets.dim() == 3:
            targets = targets.squeeze()     # [BS, L, L]-> [L,L]

        # Check sizes
        if predictions.size() != targets.size():
            raise ValueError(
                f"{rna_id} Size mismatch. Received predictions of size {predictions.size()}, "
                f"targets of size {targets.size()}"
            )

        seqlen, _ = predictions.size()
        device = predictions.device
        mask = torch.triu(torch.ones((seqlen, seqlen),device=device), 1) > 0
        if missing_nt_index is None:
            targets = targets.masked_select(mask)
            predictions = predictions.masked_select(mask)
        else:
            for i in missing_nt_index:
                mask[i, :] = 0
                mask[:, i] = 0
            targets = targets.masked_select(mask)
            predictions = predictions.masked_select(mask)

        targets = targets.reshape(-1, 1).cpu().numpy()    #[n,1]
        predictions = predictions.reshape(-1, 1).cpu().numpy()    #[n,1]

        outputs_T = np.greater_equal(predictions, T)
        tp += np.sum(np.logical_and(outputs_T, targets).astype(float), axis=0)
        tn += np.sum(np.logical_and(np.logical_not(outputs_T), np.logical_not(targets)).astype(float), axis=0)
        fp += np.sum(np.logical_and(outputs_T, np.logical_not(targets)).astype(float), axis=0)
        fn += np.sum(np.logical_and(np.logical_not(outputs_T), targets).astype(float), axis=0)
    prec = tp / (tp + fp + eps)  # precision
    recall = tp / (tp + fn + eps)  # recall
    sens = tp / (tp + fn + eps)  # senstivity
    spec = tn / (tn + fp + eps)  # spec
    TPR = tp / (tp + fn + eps)
    FPR = fp / (tn + fp + eps)
    prec[np.isnan(prec)] = 0
    F1 = 2 * ((prec * sens) / (prec + sens + eps))
    Recall = torch.tensor(recall)

    sorted_recall, sorted_indices = torch.sort(recall, descending=False)
    sorted_prec = prec[sorted_indices]
    PR = torch.trapz(x=sorted_recall, y=sorted_prec).numpy() 
    sorted_FPR, sorted_indices = torch.sort(FPR, descending=False)
    sorted_recall = recall[sorted_indices]
    AUC = torch.trapz(x=sorted_FPR, y=sorted_recall).numpy()

    MCC = (tp * tn - fp * fn) / (np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) + eps)

    threshold = torch.tensor(T[:, np.nanargmax(F1)])
    max_F1 = torch.tensor(np.nanmax(F1))   # F1 Score

    return {
        "F1":F1.tolist(), 
        "PR":PR.tolist(), 
        "AUC":AUC.tolist(), 
        "Recall":Recall.tolist(), 
        'MCC': MCC.tolist(), 
        "thresh": float(threshold), 
        'max_F1': float(max_F1)
        }

def target_T_metric(
    test_data,
    l, h, 
    T
):
    tp = 0; fn = 0; fp = 0; tn = 0

    for batch in test_data:
        rna_id = batch["rna_id"]
        predictions = batch["pred"][l, h]
        targets = batch["contacts"]
        missing_nt_index = batch["missing_nt_index"]


        if predictions.dim() == 3:
            predictions = predictions.squeeze()     # [BS, L, L]-> [L,L]
        if targets.dim() == 3:
            targets = targets.squeeze()     # [BS, L, L]-> [L,L]

        # Check sizes
        if predictions.size() != targets.size():
            raise ValueError(
                f"{rna_id} Size mismatch. Received predictions of size {predictions.size()}, "
                f"targets of size {targets.size()}"
            )

        seqlen, _ = predictions.size()
        device = predictions.device
        mask = torch.triu(torch.ones((seqlen, seqlen),device=device), 1) > 0
        if missing_nt_index is None:
            targets = targets.masked_select(mask)
            predictions = predictions.masked_select(mask)
        else:
            for i in missing_nt_index:
                mask[i, :] = 0
                mask[:, i] = 0
            targets = targets.masked_select(mask)
            predictions = predictions.masked_select(mask)

        targets = targets.reshape(-1, 1).cpu().numpy()    #[n,1]
        predictions = predictions.reshape(-1, 1).cpu().numpy()    #[n,1]

        outputs_T = np.greater_equal(predictions, T)
        tp += np.sum(np.logical_and(outputs_T, targets).astype(float), axis=0)
        tn += np.sum(np.logical_and(np.logical_not(outputs_T), np.logical_not(targets)).astype(float), axis=0)
        fp += np.sum(np.logical_and(outputs_T, np.logical_not(targets)).astype(float), axis=0)
        fn += np.sum(np.logical_and(np.logical_not(outputs_T), targets).astype(float), axis=0)
    prec = tp / (tp + fp + eps)  # precision
    recall = tp / (tp + fn + eps)  # recall
    sens = tp / (tp + fn + eps)  # senstivity
    spec = tn / (tn + fp + eps)  # spec
    TPR = tp / (tp + fn + eps)
    FPR = fp / (tn + fp + eps)
    prec[np.isnan(prec)] = 0
    F1 = 2 * ((prec * sens) / (prec + sens  + eps))
    MCC = ((tp * tn) - (fp * fn)) / (np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) + eps)

    return {"F1": float(F1), 'MCC': float(MCC), 'thresh': float(T)}

def target_T_metric_onepdb(
    test_data,
    l, h, 
    T
):
    rna_id_to_metric = {}
    for batch in test_data:
        tp = 0; fn = 0; fp = 0; tn = 0
        rna_id = batch["rna_id"]
        predictions = batch["pred"][l, h]
        targets = batch["contacts"]
        missing_nt_index = batch["missing_nt_index"]


        if predictions.dim() == 3:
            predictions = predictions.squeeze()     # [BS, L, L]-> [L,L]
        if targets.dim() == 3:
            targets = targets.squeeze()     # [BS, L, L]-> [L,L]

        # Check sizes
        if predictions.size() != targets.size():
            raise ValueError(
                f"{rna_id} Size mismatch. Received predictions of size {predictions.size()}, "
                f"targets of size {targets.size()}"
            )

        seqlen, _ = predictions.size()
        device = predictions.device
        mask = torch.triu(torch.ones((seqlen, seqlen),device=device), 1) > 0
        if missing_nt_index is None:
            targets = targets.masked_select(mask)
            predictions = predictions.masked_select(mask)
        else:
            for i in missing_nt_index:
                mask[i, :] = 0
                mask[:, i] = 0
            targets = targets.masked_select(mask)
            predictions = predictions.masked_select(mask)

        targets = targets.reshape(-1, 1).cpu().numpy()    #[n,1]
        predictions = predictions.reshape(-1, 1).cpu().numpy()    #[n,1]

        outputs_T = np.greater_equal(predictions, T)
        tp = np.sum(np.logical_and(outputs_T, targets).astype(float), axis=0)
        tn = np.sum(np.logical_and(np.logical_not(outputs_T), np.logical_not(targets)).astype(float), axis=0)
        fp = np.sum(np.logical_and(outputs_T, np.logical_not(targets)).astype(float), axis=0)
        fn = np.sum(np.logical_and(np.logical_not(outputs_T), targets).astype(float), axis=0)

        prec = tp / (tp + fp + eps)  # precision
        recall = tp / (tp + fn + eps)  # recall
        sens = tp / (tp + fn + eps)  # senstivity
        spec = tn / (tn + fp + eps)  # spec
        TPR = tp / (tp + fn + eps)
        FPR = fp / (tn + fp + eps)
        prec[np.isnan(prec)] = 0
        F1 = 2 * ((prec * sens) / (prec + sens  + eps))
        MCC = ((tp * tn) - (fp * fn)) / (np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) + eps)

        rna_id_to_metric[rna_id] = {"F1": float(F1), 'MCC': float(MCC), 'thresh': float(T)}
    return rna_id_to_metric

# CUDA version （used in Paper）
def batch_metric_cuda(
    test_data,
    layer, head,
    step: float = 0.001,
    device = 'cuda'
    
):
    T = np.arange(0, 1 + step, step)[None, :]  # threshold value [1,1001]
    T = torch.tensor(T).to(device)

    tp = 0; fn = 0; fp = 0; tn = 0

    for batch in test_data:
        rna_id = batch["rna_id"]
        predictions = batch["pred"][layer, head]
        targets = batch["contacts"]
        missing_nt_index = batch["missing_nt_index"]

        predictions, targets = predictions.to(device), targets.to(device)

        if predictions.dim() == 3:
            predictions = predictions.squeeze()     # [BS, L, L]-> [L,L]
        if targets.dim() == 3:
            targets = targets.squeeze()     # [BS, L, L]-> [L,L]

        # Check sizes
        if predictions.size() != targets.size():
            raise ValueError(
                f"{rna_id} Size mismatch. Received predictions of size {predictions.size()}, "
                f"targets of size {targets.size()}"
            )

        seqlen, _ = predictions.size()
        mask = torch.triu(torch.ones((seqlen, seqlen),device=device), 1).to(device) > 0
        if missing_nt_index is None:
            targets = targets.masked_select(mask)
            predictions = predictions.masked_select(mask)
        else:
            for i in missing_nt_index:
                mask[i, :] = 0
                mask[:, i] = 0
            targets = targets.masked_select(mask)
            predictions = predictions.masked_select(mask)

        targets = targets.reshape(-1, 1)    #[n,1]
        predictions = predictions.reshape(-1, 1)    #[n,1]

        outputs_T = torch.greater_equal(predictions, T)
        
        tp += torch.sum(torch.logical_and(outputs_T, targets).to(torch.float), dim=0)
        tn += torch.sum(torch.logical_and(torch.logical_not(outputs_T), torch.logical_not(targets)).to(torch.float), dim=0)
        fp += torch.sum(torch.logical_and(outputs_T, torch.logical_not(targets)).to(torch.float), dim=0)
        fn += torch.sum(torch.logical_and(torch.logical_not(outputs_T), targets).to(torch.float), dim=0)
    prec = tp / (tp + fp + eps)  # precision
    recall = tp / (tp + fn + eps)  # recall
    sens = tp / (tp + fn + eps)  # senstivity
    spec = tn / (tn + fp + eps)  # spec
    TPR = tp / (tp + fn + eps)
    FPR = fp / (tn + fp + eps)
    prec[torch.isnan(prec)] = 0
    F1 = 2 * ((prec * sens) / (prec + sens + eps))
    sorted_recall, sorted_indices = torch.sort(recall, descending=False)
    sorted_prec = prec[sorted_indices]
    PR = torch.trapz(x=sorted_recall, y=sorted_prec).cpu().numpy()
    sorted_FPR, sorted_indices = torch.sort(FPR, descending=False)
    sorted_recall = recall[sorted_indices]
    AUC = torch.trapz(x=sorted_FPR, y=sorted_recall).cpu().numpy()
    MCC = (tp * tn - fp * fn) / (torch.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) + eps)

    F1 = F1.cpu().numpy()
    threshold = T[:, np.nanargmax(F1)] # torch 中没有平替函数
    max_F1 = np.nanmax(F1)  # F1 Score

    return {
        "F1": F1.tolist(), 
        "PR":PR.tolist(), 
        "AUC":AUC.tolist(), 
        "Recall": recall.cpu().numpy().tolist(), 
        'MCC': MCC.cpu().numpy().tolist(), 
        "thresh": float(threshold.cpu().numpy()), 
        'max_F1': float(max_F1)
        }

def target_T_metric_cuda(
    test_data,
    l, h, 
    T,
    device='cuda'
):
    tp = 0; fn = 0; fp = 0; tn = 0
    T = torch.tensor(T).to(device)
    for batch in test_data:
        rna_id = batch["rna_id"]
        predictions = batch["pred"][l, h]
        targets = batch["contacts"]
        missing_nt_index = batch["missing_nt_index"]

        predictions, targets = predictions.to(device), targets.to(device)

        if predictions.dim() == 3:
            predictions = predictions.squeeze()     # [BS, L, L]-> [L,L]
        if targets.dim() == 3:
            targets = targets.squeeze()     # [BS, L, L]-> [L,L]

        # Check sizes
        if predictions.size() != targets.size():
            raise ValueError(
                f"{rna_id} Size mismatch. Received predictions of size {predictions.size()}, "
                f"targets of size {targets.size()}"
            )

        seqlen, _ = predictions.size()
        mask = torch.triu(torch.ones((seqlen, seqlen),device=device), 1).to(device) > 0
        if missing_nt_index is None:
            targets = targets.masked_select(mask)
            predictions = predictions.masked_select(mask)
        else:
            for i in missing_nt_index:
                mask[i, :] = 0
                mask[:, i] = 0
            targets = targets.masked_select(mask)
            predictions = predictions.masked_select(mask)

        targets = targets.reshape(-1, 1)    #[n,1]
        predictions = predictions.reshape(-1, 1)   #[n,1]

        outputs_T = torch.greater_equal(predictions, T)
        tp += torch.sum(torch.logical_and(outputs_T, targets).to(torch.float), dim=0)
        tn += torch.sum(torch.logical_and(torch.logical_not(outputs_T), torch.logical_not(targets)).to(torch.float), dim=0)
        fp += torch.sum(torch.logical_and(outputs_T, torch.logical_not(targets)).to(torch.float), dim=0)
        fn += torch.sum(torch.logical_and(torch.logical_not(outputs_T), targets).to(torch.float), dim=0)
    prec = tp / (tp + fp + eps)  # precision
    recall = tp / (tp + fn + eps)  # recall
    sens = tp / (tp + fn + eps)  # senstivity
    spec = tn / (tn + fp + eps)  # spec
    #TPR = tp / (tp + fn + eps)
    FPR = fp / (tn + fp + eps)
    prec[torch.isnan(prec)] = 0
    F1 = 2 * ((prec * sens) / (prec + sens  + eps))
    MCC = ((tp * tn) - (fp * fn)) / (torch.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) + eps)

    return {
        "F1": float(F1.cpu().numpy()), 
        'MCC': float(MCC.cpu().numpy()), 
        "thresh": float(T.cpu().numpy()), 
        }

def target_T_metric_onepdb_cuda(
    test_data,
    l, h, 
    T,
    device='cuda'
):
    
    T = torch.tensor(T).to(device)
    rna_id_to_metric = {}
    for batch in test_data:
        tp = 0; fn = 0; fp = 0; tn = 0
        rna_id = batch["rna_id"]
        predictions = batch["pred"][l, h]
        targets = batch["contacts"]
        missing_nt_index = batch["missing_nt_index"]

        predictions, targets = predictions.to(device), targets.to(device)

        if predictions.dim() == 3:
            predictions = predictions.squeeze()     # [BS, L, L]-> [L,L]
        if targets.dim() == 3:
            targets = targets.squeeze()     # [BS, L, L]-> [L,L]

        # Check sizes
        if predictions.size() != targets.size():
            raise ValueError(
                f"{rna_id} Size mismatch. Received predictions of size {predictions.size()}, "
                f"targets of size {targets.size()}"
            )

        seqlen, _ = predictions.size()
        mask = torch.triu(torch.ones((seqlen, seqlen),device=device), 1).to(device) > 0
        if missing_nt_index is None:
            targets = targets.masked_select(mask)
            predictions = predictions.masked_select(mask)
        else:
            for i in missing_nt_index:
                mask[i, :] = 0
                mask[:, i] = 0
            targets = targets.masked_select(mask)
            predictions = predictions.masked_select(mask)

        targets = targets.reshape(-1, 1)    #[n,1]
        predictions = predictions.reshape(-1, 1)   #[n,1]

        outputs_T = torch.greater_equal(predictions, T)
        tp = torch.sum(torch.logical_and(outputs_T, targets).to(torch.float), dim=0)
        tn = torch.sum(torch.logical_and(torch.logical_not(outputs_T), torch.logical_not(targets)).to(torch.float), dim=0)
        fp = torch.sum(torch.logical_and(outputs_T, torch.logical_not(targets)).to(torch.float), dim=0)
        fn = torch.sum(torch.logical_and(torch.logical_not(outputs_T), targets).to(torch.float), dim=0)
        
        prec = tp / (tp + fp + eps)  # precision
        recall = tp / (tp + fn + eps)  # recall
        sens = tp / (tp + fn + eps)  # senstivity
        spec = tn / (tn + fp + eps)  # spec

        FPR = fp / (tn + fp + eps)
        prec[torch.isnan(prec)] = 0
        F1 = 2 * ((prec * sens) / (prec + sens  + eps))
        MCC = ((tp * tn) - (fp * fn)) / (torch.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) + eps)

        rna_id_to_metric[rna_id] = {
            "F1": float(F1.cpu().numpy()), 
            'MCC': float(MCC.cpu().numpy()), 
            'thresh': float(T.cpu().numpy())
            }
    
    return rna_id_to_metric

def prepare_metric_data(dataset, preds, sel_lens=None):

    test_data = []
    for i, dt in enumerate(dataset):
        if sel_lens is None:
            seq_len = len(dt[3])
        else:
            seq_len = sel_lens[i]
        
        if isinstance(preds, list) or isinstance(preds, tuple):
            pred = preds[i][:, :, :seq_len, :seq_len]
        else:
            pred = preds[i, :, :, :seq_len, :seq_len]
        data_i = {
            "rna_id": dt[0], 
            "pred": pred,
            "contacts": torch.tensor(dt[2][:seq_len, :seq_len]),
            "missing_nt_index": None
            }
        test_data.append(data_i)
    
    return test_data

def val_metric_perform(val_data, step=0.001, device='cuda'):
    # VL数据集上计算各头-层注意力对所有PDB的预测F1
    data_0 = val_data[0]

    layers, heads = data_0["pred"].shape[:2]
    max_F1 = np.zeros((layers, heads))
    thresh = np.zeros((layers, heads))
    all_metric = {}
    for l in range(layers):
        for h in range(heads):      
            key = str(l) + '-' + str(h)
            if device == 'cpu':
                metric_lh = batch_metric(val_data, l, h, step)
            else:
                metric_lh = batch_metric_cuda(val_data, l, h, step, device)
            all_metric[key] = metric_lh
            max_F1[l, h] = metric_lh['max_F1']
            thresh[l, h] = metric_lh['thresh']

    all_metric['thresh'] = thresh.tolist()
    all_metric['max_F1'] = max_F1.tolist()

    return all_metric

def test_metric_perform(test_data, thresh, device='cuda'):
    # TS数据集上计算各头-层注意力对所有PDB的预测F1
    layers, heads = thresh.shape
    F1 = np.zeros((layers, heads))
    all_metric = {}
    for l in range(layers):
        for h in range(heads):      
            key = str(l) + '-' + str(h)
            if device == 'cpu':
                metric_lh = target_T_metric(test_data, l, h, thresh[l, h])
            else:
                metric_lh = target_T_metric_cuda(test_data, l, h, thresh[l, h], device)
            all_metric[key] = metric_lh
            F1[l, h] = metric_lh['F1']
                
    all_metric['F1'] = F1.tolist()
    all_metric['max_F1'] = float(np.max(F1))
    all_metric['max_F1_thresh'] = float(thresh[F1 == np.max(F1)])
    return all_metric

def test_metric_maxval_perform(test_data, max_l, max_h, thresh, device='cuda'):
    # TS数据集上计算指定头-层注意力对所有PDB的预测F1
    if device == 'cpu':
        metric_lh = target_T_metric(test_data, max_l, max_h, thresh[max_l, max_h])
    else:
        metric_lh = target_T_metric_cuda(test_data, max_l, max_h, thresh[max_l, max_h], device)
                
    return metric_lh

def test_metric_maxval_onepdb_perform(test_data, max_l, max_h, thresh, device='cuda'):
    # TS数据集上计算指定头-层注意力对每个PDB的预测F1
    if device == 'cpu':
        metric_lh = target_T_metric_onepdb(test_data, max_l, max_h, thresh[max_l, max_h])
    else:
        metric_lh = target_T_metric_onepdb_cuda(test_data, max_l, max_h, thresh[max_l, max_h], device)
                
    return metric_lh
