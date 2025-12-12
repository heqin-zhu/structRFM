from collections import defaultdict
from base_classes import MlpProjector
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

import interface

from multimolecule import RnaTokenizer, RiNALMoModel


class SparseEmbedding(nn.Module):

    def __init__(self, dim):
        super(SparseEmbedding, self).__init__()
        self.n_out = dim
        self.embedding = nn.Embedding(6, dim, padding_idx=0)
        self.vocb = defaultdict(
            lambda: 5, {'0': 0, 'A': 1, 'C': 2, 'G': 3, 'T': 4})

    def __call__(self, seq):
        seq = np.array([[self.vocb[c] for c in s] for s in seq], dtype='int64')
        seq = torch.from_numpy(seq).to(self.embedding.weight.device)
        output = self.embedding(seq)
        output = output.permute(0, 2, 1)
        return output


class CNNLayer(nn.Module):
    def __init__(self,
                 n_in,
                 num_filters=(128, ),
                 filter_size=(7, ),
                 pool_size=(1, ),
                 dilation=1,
                 dropout_rate=0.0,
                 resnet=False):
        super(CNNLayer, self).__init__()
        self.resnet = resnet
        self.net = nn.ModuleList()
        for n_out, ksize, p in zip(num_filters, filter_size, pool_size):
            self.net.append(
                nn.Sequential(
                    nn.Conv1d(n_in, n_out, kernel_size=ksize, dilation=2 **
                              dilation, padding=2**dilation * (ksize // 2)),
                    nn.MaxPool1d(p, stride=1, padding=p //
                                 2) if p > 1 else nn.Identity(),
                    nn.GroupNorm(1, n_out),  # same as LayerNorm?
                    nn.CELU(),
                    nn.Dropout(p=dropout_rate)))
            n_in = n_out

    def forward(self, x):  # (B=1, 4, N)
        for net in self.net:
            x_a = net(x)
            x = x + x_a if self.resnet and x.shape[1] == x_a.shape[1] else x_a
        return x


class CNNLSTMEncoder(nn.Module):
    def __init__(self,
                 n_in,
                 num_filters=(256, ),
                 filter_size=(7, ),
                 pool_size=(1, ),
                 dilation=0,
                 num_lstm_layers=0,
                 num_lstm_units=0,
                 num_att=0,
                 dropout_rate=0.0,
                 resnet=True):
        super(CNNLSTMEncoder, self).__init__()
        self.resnet = resnet
        self.n_in = self.n_out = n_in
        while len(num_filters) > len(filter_size):
            filter_size = tuple(filter_size) + (filter_size[-1], )
        while len(num_filters) > len(pool_size):
            pool_size = tuple(pool_size) + (pool_size[-1], )
        if num_lstm_layers == 0 and num_lstm_units > 0:
            num_lstm_layers = 1

        self.dropout = nn.Dropout(p=dropout_rate)
        self.conv = self.lstm = self.att = None

        if len(num_filters) > 0 and num_filters[0] > 0:
            self.conv = CNNLayer(n_in,
                                 num_filters,
                                 filter_size,
                                 pool_size,
                                 dilation,
                                 dropout_rate=dropout_rate,
                                 resnet=self.resnet)
            self.n_out = n_in = num_filters[-1]

        if num_lstm_layers > 0:
            self.lstm = nn.LSTM(
                n_in,
                num_lstm_units,
                num_layers=num_lstm_layers,
                bidirectional=True,
                dropout=dropout_rate if num_lstm_layers > 1 else 0)
            self.n_out = num_lstm_units * 2
            self.lstm_ln = nn.LayerNorm(self.n_out)

        if num_att > 0:
            self.att = nn.MultiheadAttention(
                self.n_out, num_att, dropout=dropout_rate)

    def forward(self, x):  # (B, n_in, N)

        if self.conv is not None:
            x = self.conv(x)  # (B, C, N)
        x = x.permute(0, 2, 1)  # (B, N, C)

        if self.lstm is not None:
            x_a, _ = self.lstm(x)
            x_a = self.lstm_ln(x_a)
            x_a = self.dropout(F.celu(x_a))  # (B, N, H*2)
            x = x + x_a if self.resnet and x.shape[2] == x_a.shape[2] else x_a

        if self.att is not None:
            x = x.permute(1, 0, 2)
            x_a, _ = self.att(x, x, x)
            x = x + x_a
            x = x.permute(1, 0, 2)

        return x


class Transform2D(nn.Module):

    def __init__(self, join='cat', context_length=0):
        super(Transform2D, self).__init__()
        self.join = join

    def forward(self, x_l, x_r):

        assert (x_l.shape == x_r.shape)
        B, N, C = x_l.shape
        x_l = x_l.view(B, N, 1, C).expand(B, N, N, C)
        x_r = x_r.view(B, 1, N, C).expand(B, N, N, C)
        if self.join == 'cat':
            x = torch.cat((x_l, x_r), dim=3)  # (B, N, N, C*2)
        elif self.join == 'add':
            x = x_l + x_r  # (B, N, N, C)
        elif self.join == 'mul':
            x = x_l * x_r  # (B, N, N, C)

        return x


class PairedLayer(nn.Module):
    def __init__(self, n_in, n_out=1, filters=(), ksize=(), fc_layers=(), dropout_rate=0.0, exclude_diag=True, resnet=True):

        super(PairedLayer, self).__init__()

        self.resnet = resnet
        self.exclude_diag = exclude_diag
        while len(filters) > len(ksize):
            ksize = tuple(ksize) + (ksize[-1], )

        self.conv = nn.ModuleList()
        for m, k in zip(filters, ksize):
            self.conv.append(
                nn.Sequential(
                    nn.Conv2d(n_in, m, k, padding=k//2),
                    nn.GroupNorm(1, m),
                    nn.CELU(),
                    nn.Dropout(p=dropout_rate)))
            n_in = m

        fc = []
        for m in fc_layers:
            fc += [nn.Linear(n_in, m), nn.LayerNorm(m),
                   nn.CELU(), nn.Dropout(p=dropout_rate)]
            n_in = m
        fc += [nn.Linear(n_in, n_out)]
        self.fc = nn.Sequential(*fc)

    def forward(self, x):

        diag = 1 if self.exclude_diag else 0
        B, N, _, C = x.shape
        x = x.permute(0, 3, 1, 2)
        x_u = torch.triu(x.view(B*C, N, N), diagonal=diag).view(B, C, N, N)
        x_l = torch.tril(x.view(B*C, N, N), diagonal=-1).view(B, C, N, N)
        x = torch.cat((x_u, x_l), dim=0).view(B*2, C, N, N)
        for conv in self.conv:
            x_a = conv(x)
            # (B*2, n_out, N, N
            x = x + x_a if self.resnet and x.shape[1] == x_a.shape[1] else x_a
        x_u, x_l = torch.split(x, B, dim=0)  # (B, n_out, N, N) * 2
        x_u = torch.triu(x_u.view(B, -1, N, N), diagonal=diag)
        x_l = torch.tril(x_u.view(B, -1, N, N), diagonal=-1)
        x = x_u + x_l  # (B, n_out, N, N)
        x = x.permute(0, 2, 3, 1).view(B*N*N, -1)
        x = self.fc(x)
        return x.view(B, N, N, -1)  # (B, N, N, n_out)


class NeuralNet(nn.Module):
    def __init__(self, embed_dim=0,
                 embed_size=0, num_filters=(96, ), filter_size=(5, ),
                 dilation=0, pool_size=(1, ), num_lstm_layers=0, num_lstm_units=0, num_att=0, no_split_lr=False,
                 pair_join='cat', num_paired_filters=(), paired_filter_size=(),
                 num_hidden_units=(32, ), dropout_rate=0.0, fc_dropout_rate=0.0,
                 exclude_diag=True, n_out_paired_layers=0, n_out_unpaired_layers=0, num_transformer_layers=0):

        super(NeuralNet, self).__init__()

        self.no_split_lr = no_split_lr
        self.pair_join = pair_join
        self.embedding = SparseEmbedding(embed_size)
        n_in = self.embedding.n_out

        self.proj_head = MlpProjector(embed_dim, 128)
        self.encoder = CNNLSTMEncoder(n_in + 128,
                                      num_filters=num_filters,
                                      filter_size=filter_size,
                                      pool_size=pool_size,
                                      dilation=dilation,
                                      num_att=num_att,
                                      num_lstm_layers=num_lstm_layers,
                                      num_lstm_units=num_lstm_units,
                                      dropout_rate=dropout_rate)
        n_in = self.encoder.n_out

        self.transform2d = Transform2D(join=pair_join)

        n_in_paired = n_in // 2 if pair_join != 'cat' else n_in

        self.fc_paired = PairedLayer(n_in_paired,
                                     n_out_paired_layers,
                                     filters=num_paired_filters,
                                     ksize=paired_filter_size,
                                     exclude_diag=exclude_diag,
                                     fc_layers=num_hidden_units,
                                     dropout_rate=fc_dropout_rate)
        self.fc_unpaired = None

    def forward(self, seq, embeddings):

        x = self.embedding(['0' + s for s in seq])
        embeddings = self.proj_head(embeddings)
        x = torch.concat([x, embeddings.permute(0, 2, 1)], axis=1)
        x = self.encoder(x)

        # if self.no_split_lr:
        #     x_l, x_r = x, x
        # else:
        x_l = x[:, :, 0::2]
        x_r = x[:, :, 1::2]
        x_r = x_r[:, :, torch.arange(x_r.shape[-1]-1, -1, -1)]
        x_lr = self.transform2d(x_l, x_r)

        score_paired = self.fc_paired(x_lr)
        if self.fc_unpaired is not None:
            score_unpaired = self.fc_unpaired(x)
        else:
            score_unpaired = None

        return score_paired, score_unpaired


class LengthLayer(nn.Module):
    def __init__(self, n_in, layers=(), dropout_rate=0.5):
        super(LengthLayer, self).__init__()
        self.n_in = n_in
        n = n_in if isinstance(n_in, int) else np.prod(n_in)

        l = []
        for m in layers:
            l += [nn.Linear(n, m), nn.CELU(), nn.Dropout(p=dropout_rate)]
            n = m
        l += [nn.Linear(n, 1)]
        self.net = nn.Sequential(*l)

        if isinstance(self.n_in, int):
            self.x = torch.tril(torch.ones((self.n_in, self.n_in)))
        else:
            n = np.prod(self.n_in)
            x = np.fromfunction(lambda i, j, k, l: np.logical_and(
                k <= i, l <= j), (*self.n_in, *self.n_in))
            self.x = torch.from_numpy(x.astype(np.float32)).reshape(n, n)

    def forward(self, x):
        return self.net(x)

    def make_param(self):
        device = next(self.net.parameters()).device
        x = self.forward(self.x.to(device))
        return x.reshape((self.n_in,) if isinstance(self.n_in, int) else self.n_in)


class AbstractFold(nn.Module):

    def __init__(self, predict):
        super(AbstractFold, self).__init__()
        self.predict = predict

    def clear_count(self, param):
        param_count = {}
        for n, p in param.items():
            if n.startswith("score_"):
                param_count["count_"+n[6:]] = torch.zeros_like(p)
        param.update(param_count)
        return param

    def calculate_differentiable_score(self, v, param, count):
        s = None
        for n, p in param.items():
            if n.startswith("score_"):
                if not s:
                    s = torch.zeros(1, device=p.device)
                s += torch.sum(p * count["count_"+n[6:]].to(p.device))
        s += v - s.item()
        return s


class ZukerFold(AbstractFold):
    def __init__(self, max_helix_length=30, **kwargs):
        super(ZukerFold, self).__init__(predict=interface.predict_zuker)

        self.max_helix_length = max_helix_length
        self.net = NeuralNet(**kwargs)

        self.fc_length = nn.ModuleDict({
            'score_hairpin_length': LengthLayer(31),
            'score_bulge_length': LengthLayer(31),
            'score_internal_length': LengthLayer(31),
            'score_internal_explicit': LengthLayer((5, 5)),
            'score_internal_symmetry': LengthLayer(16),
            'score_internal_asymmetry': LengthLayer(29),
            'score_helix_length': LengthLayer(31)
        })

    def make_param(self, seq, embeddings):
        score_paired, score_unpaired = self.net(seq, embeddings)
        B, N, _, _ = score_paired.shape

        score_basepair = torch.zeros((B, N, N))
        score_helix_stacking = score_paired[:, :, :, 0]  # (B, N, N)
        score_mismatch_external = score_paired[:, :, :, 1]  # (B, N, N)
        score_mismatch_internal = score_paired[:, :, :, 1]  # (B, N, N)
        score_mismatch_multi = score_paired[:, :, :, 1]  # (B, N, N)
        score_mismatch_hairpin = score_paired[:, :, :, 1]  # (B, N, N)
        score_unpaired = score_paired[:, :, :, 2]  # (B, N, N)
        score_base_hairpin = score_unpaired
        score_base_internal = score_unpaired
        score_base_multi = score_unpaired
        score_base_external = score_unpaired

        param = [{
            'score_basepair': score_basepair[i],
            'score_helix_stacking': score_helix_stacking[i],
            'score_mismatch_external': score_mismatch_external[i],
            'score_mismatch_hairpin': score_mismatch_hairpin[i],
            'score_mismatch_internal': score_mismatch_internal[i],
            'score_mismatch_multi': score_mismatch_multi[i],
            'score_base_hairpin': score_base_hairpin[i],
            'score_base_internal': score_base_internal[i],
            'score_base_multi': score_base_multi[i],
            'score_base_external': score_base_external[i],
            'score_hairpin_length': self.fc_length['score_hairpin_length'].make_param(),
            'score_bulge_length': self.fc_length['score_bulge_length'].make_param(),
            'score_internal_length': self.fc_length['score_internal_length'].make_param(),
            'score_internal_explicit': self.fc_length['score_internal_explicit'].make_param(),
            'score_internal_symmetry': self.fc_length['score_internal_symmetry'].make_param(),
            'score_internal_asymmetry': self.fc_length['score_internal_asymmetry'].make_param(),
            'score_helix_length': self.fc_length['score_helix_length'].make_param()
        } for i in range(B)]

        return param


class RNAFold(AbstractFold):
    def __init__(self, init_param=None):
        super(RNAFold, self).__init__(interface.predict_turner)

        for n in dir(init_param):
            if n.startswith("score_"):
                setattr(self, n, nn.Parameter(
                        torch.tensor(getattr(init_param, n))))

    def make_param(self, seq):
        param = {n: getattr(self, n)
                 for n in dir(self) if n.startswith("score_")}
        return [param for s in seq]


class MixedFold(AbstractFold):
    def __init__(self, init_param=None, max_helix_length=30, **kwargs):
        super(MixedFold, self).__init__(interface.predict_mxfold)
        self.turner = RNAFold(init_param=init_param)
        self.zuker = ZukerFold(max_helix_length=max_helix_length, **kwargs)
        self.max_helix_length = max_helix_length

    def forward(self,
                seq,
                embeddings,
                return_param=False,
                param=None,
                max_internal_length=30,
                constraint=None,
                reference=None,
                loss_pos_paired=0.0,
                loss_neg_paired=0.0,
                loss_pos_unpaired=0.0,
                loss_neg_unpaired=0.0):

        param = self.make_param(
            seq, embeddings) if param is None else param  # reuse param or not
        ss = []
        preds = []
        pairs = []
        for i in range(len(seq)):
            param_on_cpu = {
                'turner': {k: v.to("cpu") for k, v in param[i]['turner'].items()},
                'positional': {k: v.to("cpu") for k, v in param[i]['positional'].items()}
            }
            param_on_cpu = {k: self.clear_count(
                v) for k, v in param_on_cpu.items()}
            with torch.no_grad():
                v, pred, pair = interface.predict_mxfold(
                    seq[i],
                    param_on_cpu,
                    max_internal_length=max_internal_length if max_internal_length is not None else len(
                        seq[i]),
                    max_helix_length=self.max_helix_length,
                    constraint=constraint[i].tolist(
                    ) if constraint is not None else None,
                    reference=reference[i].tolist(
                    ) if reference is not None else None,
                    loss_pos_paired=loss_pos_paired,
                    loss_neg_paired=loss_neg_paired,
                    loss_pos_unpaired=loss_pos_unpaired,
                    loss_neg_unpaired=loss_neg_unpaired)

            if torch.is_grad_enabled():
                v = self.calculate_differentiable_score(
                    v, param[i]['positional'], param_on_cpu['positional'])

            ss.append(v)
            preds.append(pred)
            pairs.append(pair)

        ss = torch.stack(ss) if torch.is_grad_enabled(
        ) else torch.tensor(ss)
        if return_param:
            return ss, preds, pairs, param
        else:
            return ss, preds, pairs

    def make_param(self, seq, embeddings):
        ts = self.turner.make_param(seq)
        ps = self.zuker.make_param(seq, embeddings)
        return [{'turner': t, 'positional': p} for t, p in zip(ts, ps)]


class RNABertForSsp(nn.Module):
    def __init__(self, bert):
        super(RNABertForSsp, self).__init__()
        self.bert = bert

    def _load_pretrained_bert(self, path):
        self.load_state_dict(torch.load(path, map_location="cpu"), strict=False)

    def forward(self, input_ids):
        encoded_layers, _ = self.bert(input_ids)
        embeddings = encoded_layers[-1].detach()
        return embeddings


class RNAMsmForSsp(nn.Module):
    def __init__(self, bert):
        super(RNAMsmForSsp, self).__init__()
        self.bert = bert

    def _load_pretrained_bert(self, path):
        self.load_state_dict(torch.load(path, map_location="cpu"), strict=False)

    def forward(self, input_ids):
        output = self.bert(input_ids, repr_layers=[10])
        embeddings = output["representations"][10][:, 0, ...].detach()
        return embeddings


class RNAFmForSsp(nn.Module):
    def __init__(self, bert):
        super(RNAFmForSsp, self).__init__()
        self.bert = bert

    def _load_pretrained_bert(self, path):
        self.load_state_dict(torch.load(path, map_location="cpu"), strict=False)

    def forward(self, input_ids):
        output = self.bert(input_ids, repr_layers=[12])
        embeddings = output["representations"][12].detach()
        return embeddings


class RiNALMoForSsp(nn.Module):
    def __init__(self, bert):
        super(RiNALMoForSsp, self).__init__()
        self.bert = bert


    def forward(self, input_ids, attention_mask=None):
        output = self.bert(input_ids, output_hidden_states=True, attention_mask=attention_mask)
        embeddings = output.hidden_states[-1][:, 1:, :].detach()
        return embeddings


class structRFMForSsp(nn.Module):
    def __init__(self, bert):
        super(structRFMForSsp, self).__init__()
        self.bert = bert

    def _load_pretrained_bert(self, path):
        self.load_state_dict(torch.load(path, map_location="cpu"), strict=False)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        embeddings = outputs.hidden_states[-1].detach()
        return embeddings
