from torch import nn
import torch
import torch.nn.functional as F


class SeqClsLoss(nn.Module):

    def __init__(self):
        super(SeqClsLoss, self).__init__()

    def forward(self, outputs, labels):
        # convert labels to int64
        loss = F.cross_entropy(outputs, labels)
        return loss


class RRInterLoss(nn.Module):

    def __init__(self):
        super(RRInterLoss, self).__init__()

    def forward(self, outputs, labels):
        loss = F.cross_entropy(outputs, labels)
        return loss


class StructuredLoss(nn.Module):
    def __init__(self,
                 loss_pos_paired=0,
                 loss_neg_paired=0,
                 loss_pos_unpaired=0,
                 loss_neg_unpaired=0,
                 l1_weight=0.,
                 l2_weight=0.):

        super(StructuredLoss, self).__init__()
        # self.model = model
        self.loss_pos_paired = loss_pos_paired
        self.loss_neg_paired = loss_neg_paired
        self.loss_pos_unpaired = loss_pos_unpaired
        self.loss_neg_unpaired = loss_neg_unpaired
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight

    def forward(self, model, seq, pairs, embeddings, fname=None):
        pred, pred_s, _, param = model(seq,
                                       embeddings,
                                       return_param=True,
                                       reference=pairs,
                                       loss_pos_paired=self.loss_pos_paired,
                                       loss_neg_paired=self.loss_neg_paired,
                                       loss_pos_unpaired=self.loss_pos_unpaired,
                                       loss_neg_unpaired=self.loss_neg_unpaired)
        ref, ref_s, _ = model(seq, embeddings, param=param,
                              constraint=pairs, max_internal_length=None)
        length = torch.tensor([len(s) for s in seq]).to(pred.device)
        loss = (pred - ref) / length

        if loss.item() > 1e10 or torch.isnan(loss):
            print()
            print(fname)
            print(loss.item(), pred.item(), ref.item())
            print(seq)

        if self.l1_weight > 0.0:
            for p in model.parameters():
                loss += self.l1_weight * torch.sum(torch.abs(p))
        return torch.sum(loss)
