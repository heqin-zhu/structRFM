from collections import defaultdict
from pathlib import Path

import sys
import torch
import torch.nn as nn

from einops import rearrange
from einops.layers.torch import Rearrange

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(1, str(Path(__file__).parent / 'network'))

from modules import BasicBlock, Symm, ResNetBlock, RelPos
from config import *


class InputEmbedder(nn.Module):
    def __init__(self, dim=48, stru_dim=47):
        super(InputEmbedder, self).__init__()
        self.bn1 = nn.InstanceNorm2d(stru_dim, affine=True)
        self.elu1 = nn.ELU(inplace=True)
        self.conv1 = nn.Conv2d(stru_dim, dim, 1)
        self.linear1 = nn.Sequential(
            self.bn1,
            self.elu1,
            self.conv1
        )
        self.token_emb = nn.Embedding(5, dim)

    def forward(self, msa, ss, msa_cutoff=500):
        '''
            msa: B x N x L
            ss : B x L x L x C

            B = 1
        '''
        with torch.no_grad():
            f2d = self.get_f2d(msa[0], ss)
        # f2d: B x L x L x 47  (47 = 26 + 4*5 + 1)
        pair = self.linear1(f2d.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        m = self.token_emb(msa[:, :msa_cutoff, :].long())
        return {'pair': pair, 'msa': m}

    def get_f2d(self, msa, ss):
        nrow, ncol = msa.size()[-2:]
        if nrow == 1:
            msa = msa.view(nrow, ncol).repeat(2, 1)
            nrow = 2
        msa1hot = (torch.arange(5).to(msa.device) == msa[..., None].long()).float()
        # msa1hot: N x L x 5
        w = self.reweight(msa1hot, .8)

        # 1D features
        f1d_seq = msa1hot[0, :, :4]
        f1d_pssm = self.msa2pssm(msa1hot, w)

        # f1d: L x 10   10=5x2
        f1d = torch.cat([f1d_seq, f1d_pssm], dim=1)

        # 2D features
        # f2d_dca: L x L x 26
        f2d_dca = self.fast_dca(msa1hot, w)

        f2d = torch.cat([f1d[:, None, :].repeat([1, ncol, 1]),
                         f1d[None, :, :].repeat([ncol, 1, 1]),
                         f2d_dca], dim=-1)
        # f2d: L x L x 46
        f2d = f2d.view([1, ncol, ncol, 26 + 4 * 5])
        return torch.cat([f2d, ss.float()], dim=-1)

    @staticmethod
    def msa2pssm(msa1hot, w):
        beff = w.sum()
        f_i = (w[:, None, None] * msa1hot).sum(dim=0) / beff + 1e-9
        h_i = (-f_i * torch.log(f_i)).sum(dim=1)
        return torch.cat([f_i, h_i[:, None]], dim=1)

    @staticmethod
    def reweight(msa1hot, cutoff):
        id_min = msa1hot.size(1) * cutoff
        id_mtx = torch.tensordot(msa1hot, msa1hot, [[1, 2], [1, 2]])
        id_mask = id_mtx > id_min
        w = 1.0 / id_mask.sum(dim=-1).float()
        return w

    @staticmethod
    def fast_dca(msa1hot, weights, penalty=4.5):
        nr, nc, ns = msa1hot.size()
        try:
            x = msa1hot.view(nr, nc * ns)
        except RuntimeError:
            x = msa1hot.contiguous().view(nr, nc * ns)
        num_points = weights.sum() - torch.sqrt(weights.mean())

        mean = torch.sum(x * weights[:, None], dim=0, keepdim=True) / num_points
        x = (x - mean) * torch.sqrt(weights[:, None])
        cov = torch.matmul(x.permute(1, 0), x) / num_points

        cov_reg = cov + torch.eye(nc * ns).to(x.device) * penalty / torch.sqrt(weights.sum())
        inv_cov = torch.inverse(cov_reg)

        x1 = inv_cov.view(nc, ns, nc, ns)
        x2 = x1.permute(0, 2, 1, 3)
        features = x2.reshape(nc, nc, ns * ns)

        x3 = torch.sqrt((x1[:, :-1, :, :-1] ** 2).sum((1, 3))) * (1 - torch.eye(nc).to(x.device))
        apc = x3.sum(dim=0, keepdim=True) * x3.sum(dim=1, keepdim=True) / x3.sum()
        contacts = (x3 - apc) * (1 - torch.eye(nc).to(x.device))

        return torch.cat([features, contacts[:, :, None]], dim=2)


class RecyclingEmbedder(nn.Module):
    def __init__(self, dim=48):
        super(RecyclingEmbedder, self).__init__()
        self.linear = nn.Linear(38, dim)
        self.norm_pair = nn.LayerNorm(dim)
        self.norm_msa = nn.LayerNorm(dim)

    def forward(self, reprs_prev):
        pair = self.norm_pair(reprs_prev['pair'])
        single = self.norm_msa(reprs_prev['single'])
        return single, pair


class Distogram(nn.Module):
    def __init__(self, dim=48):
        super(Distogram, self).__init__()
        self.out_elu_2d = nn.Sequential(
            nn.InstanceNorm2d(dim, affine=True),
            nn.ELU(inplace=True)
        )
        self.fc_2d = nn.ModuleDict(
            {
                'distance': nn.ModuleDict(
                    dict((a,
                          nn.Sequential(
                              Symm('b i j d->b j i d'),
                              nn.Linear(dim, n_bins['2D']['distance'])))
                         for a in obj['2D']['distance']),
                ),
                'contact': nn.ModuleDict(
                    dict(
                        (a, nn.Sequential(
                            Symm('b i j d->b j i d'),
                            nn.Linear(dim, n_bins['2D']['contact'])))
                        for a in obj['2D']['contact'])
                ),
            }
        )

    def forward(self, pair_repr):
        pair_repr = rearrange(self.out_elu_2d(rearrange(pair_repr, 'b i j d->b d i j')), 'b d i j->b i j d')
        pred_dict = defaultdict(dict)
        for k in obj['2D']:
            if k != 'contact':
                for a in obj['2D'][k]:
                    pred_dict[k][a] = self.fc_2d[k][a](pair_repr).softmax(-1).squeeze(0)

            else:
                for a in obj['2D'][k]:
                    pred_dict['contact'] = self.fc_2d[k][a](pair_repr).sigmoid().squeeze()

        return pred_dict


class RNAformer(nn.Module):
    def __init__(
            self,
            *,
            dim=32,
            in_dim=526,
            emb_dim=640,
            depth=32,
            heads=8,
            dim_head=64,
            num_tokens=5,
            attn_dropout=0.,
            ff_dropout=0.,
            msa_tie_row_attn=False,
    ):
        super().__init__()

        self.bn1 = nn.InstanceNorm2d(in_dim, affine=True)
        self.elu1 = nn.ELU(inplace=True)
        self.conv1 = nn.Conv2d(in_dim, dim, 1)
        self.linear1 = nn.Sequential(
            self.bn1,
            self.elu1,
            self.conv1
        )
        self.token_emb = nn.Embedding(num_tokens, dim)
        self.linear_emb = nn.Linear(emb_dim, dim)
        self.input_emb = RelPos(dim)

        # main trunk modules

        self.net = nn.ModuleList([
            BasicBlock(dim=dim, heads=heads, dim_head=dim_head, msa_tie_row_attn=msa_tie_row_attn,
                       attn_dropout=attn_dropout, ff_dropout=ff_dropout)
            for _ in range(depth)
        ])

    def forward(
            self,
            f2d,
            msa=None,
            res_id=None,
            msa_emb=None,
            preprocess=True,
            return_msa=True,
            return_attn=False,
            return_mid=False,
            relpos_enc=True
    ):
        device = f2d.device
        if preprocess:
            # add dca
            # x = torch.cat([x, f2d], dim=-1).permute(0, 3, 1, 2)
            x = f2d.permute(0, 3, 1, 2)
            x = self.linear1(x).permute(0, 2, 3, 1)

            # embed multiple sequence alignment (msa)
            m = self.token_emb(msa.long())
            if msa_emb is not None:
                m += self.linear_emb(msa_emb)
        else:
            x, m = f2d, msa_emb

        if res_id is not None or relpos_enc:
            if res_id is None:
                res_id = torch.arange(x.size(1), device=device)
            res_id = res_id.view(1, x.size(1))
            x = self.input_emb(x, res_id)

        attn_maps = []
        mid_reprs = []
        for layer in self.net:
            outputs = layer(m, x, return_attn=return_attn)
            m, x = outputs[:2]
            if return_attn:
                attn_maps.append(outputs[-1])
            if return_mid:
                mid_reprs.append(x.squeeze(0))

        out = [x]
        if return_msa:
            out.append(m)
        if return_attn:
            out.append(attn_maps)
        if return_mid:
            out.append(mid_reprs)
        return tuple(out)


class DistPredictor(nn.Module):
    def __init__(self, dim_2d=48, layers_2d=12, stru_feat_type='SS'):
        '''
            stru_feat_type:
                SS: no LM feature, only SS: SS
                LM: only LM mat feature, no SS: LM_mat
                both: concat LM mat feature with SS: [SS, LM_mat_feat]
        '''
        super(DistPredictor, self).__init__()

        self.stru_feat_type_list = ['SS', 'LM', 'both']
        assert stru_feat_type in self.stru_feat_type_list, f"Unknown feature type: {stru_feat_type}, should be in {self.stru_feat_type_list}"
        self.stru_feat_type = stru_feat_type
        if stru_feat_type == 'SS' or stru_feat_type == 'LM':
            stru_dim = 47
        elif stru_feat_type == 'both':
            stru_dim = 48
        self.input_embedder = InputEmbedder(dim=dim_2d, stru_dim=stru_dim)
        self.recycle_embedder = RecyclingEmbedder(dim=dim_2d)
        self.net2d = RNAformer(dim=dim_2d,
                               depth=layers_2d,
                               msa_tie_row_attn=False,
                               attn_dropout=0.,
                               ff_dropout=0.)
        self.to_dist = Distogram(dim_2d)
        self.to_ss = nn.Sequential(
            nn.LayerNorm(dim_2d),
            nn.Linear(dim_2d, dim_2d),
            Symm('b i j d->b j i d'),
            nn.Linear(dim_2d, dim_2d),
            nn.ReLU(),
            nn.LayerNorm(dim_2d),
            nn.Dropout(.1),
            nn.Linear(dim_2d, 1)
        )
        self.to_mask = nn.Sequential(
            Rearrange('b i j d->b d i j'),
            *[ResNetBlock(in_channel=dim_2d, out_channel=dim_2d) for _ in range(4)],
            Rearrange('b d i j->b i j d'),
            nn.ELU(),
            nn.Linear(dim_2d, 1),
            Symm('b i j d->b j i d'),
            nn.Sigmoid()
        )

        self.to_estogram = nn.Sequential(
            Rearrange('b i j d->b d i j'),
            *[ResNetBlock(in_channel=dim_2d, out_channel=dim_2d) for _ in range(4)],
            Rearrange('b d i j->b i j d'),
            nn.ELU(),
            nn.Linear(dim_2d, 19),
            Symm('b i j d->b j i d'),
            nn.Softmax(-1)
        )

    def forward(self, msa, ss, res_id=None, num_recycle=3, msa_cutoff=500, is_training=False):
        '''
            msa: B x N x L
            ss : B x L x L x C
            
            B=1
        '''
        reprs_prev = None
        for c in range(1 + num_recycle):
            with torch.set_grad_enabled(is_training):
                with torch.cuda.amp.autocast(enabled=False):
                    reprs = self.input_embedder(msa, ss, msa_cutoff=msa_cutoff)

                    if reprs_prev is None:
                        reprs_prev = {
                            'pair': torch.zeros_like(reprs['pair']),
                            'single': torch.zeros_like(reprs['msa'][:, 0]),
                        }
                    # print('reprs msa pair', reprs['msa'].shape, reprs['pair'].shape)
                    # print('reprs_prev single pair', reprs_prev['single'].shape, reprs_prev['pair'].shape)
                    rec_msa, rec_pair = self.recycle_embedder(reprs_prev)
                    # print('rec_msa msa', rec_msa.shape, reprs['msa'].shape)
                    reprs['msa'] = reprs['msa'] + rec_msa
                    reprs['pair'] = reprs['pair'] + rec_pair
                    out = self.net2d(reprs['pair'], msa_emb=reprs['msa'], return_msa=True, res_id=res_id,
                                     preprocess=False, return_attn=c == num_recycle)
                    if c != num_recycle:
                        pair_repr, msa_repr = out
                    else:
                        pair_repr, msa_repr, attn_maps = out
                reprs_prev = {
                    'single': msa_repr[..., 0, :, :].detach(),
                    'pair': pair_repr.detach(),
                }

                outputs = {}
                outputs['geoms'] = self.to_dist(pair_repr)

                """ Contact mask prediction """
                outputs['contact_mask'] = self.to_mask(pair_repr).squeeze(-1)

                """ Distance error prediction """
                outputs['estogram'] = self.to_estogram(pair_repr)
                #
                # plddt_prob = self.to_plddt(outputs['single'][-1]).softmax(-1)
                # plddt = torch.einsum('bik,k->bi', plddt_prob, torch.arange(0.01, 1.01, 0.02, device=device))
                # outputs['plddt_prob'] = plddt_prob
                # outputs['plddt'] = plddt

                outputs['ss'] = self.to_ss(pair_repr).sigmoid()

        return outputs
