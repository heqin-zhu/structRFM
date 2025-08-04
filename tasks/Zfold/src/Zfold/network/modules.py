import torch, math
import torch.nn as nn
from inspect import isfunction
from einops import rearrange
from einops.layers.torch import Rearrange
from torch.utils.checkpoint import checkpoint
from functools import partial
from dropout import *


class Symm(nn.Module):
    def __init__(self, pattern):
        super(Symm, self).__init__()
        self.pattern = pattern

    def forward(self, x):
        return (x + Rearrange(self.pattern)(x)) / 2


class ResNetBlock(nn.Module):
    def __init__(self, dilation=1, in_channel=64, out_channel=64, kernel_size=3, dropout=.15, norm=nn.InstanceNorm2d):
        super(ResNetBlock, self).__init__()
        self.bn1 = norm(in_channel, affine=True)
        self.bn2 = norm(out_channel, affine=True)
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size, padding=dilation, dilation=dilation)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size, padding=dilation, dilation=dilation)
        self.net = nn.Sequential(
            self.bn1,
            nn.ELU(inplace=True),
            self.conv1,
            nn.Dropout(dropout),
            self.bn2,
            nn.ELU(inplace=True),
            self.conv2,
        )

    def forward(self, x):
        out = self.net(x)
        return x + out


class Bottle2neck(nn.Module):

    def __init__(self, inplanes, planes, stride=1, dilation=1, baseWidth=26, scale=4, stype='normal', expansion=4,
                 shortcut=True):
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
            downsample: None when stride = 1
            baseWidth: basic width of conv3x3
            scale: number of scale.
            type: 'normal': normal set. 'stage': first block of a new stage.
        """
        super(Bottle2neck, self).__init__()
        self.expansion = expansion

        width = int(math.floor(planes * (baseWidth / 64.0)))
        self.conv1 = nn.Conv2d(inplanes, width * scale, kernel_size=1)
        self.bn1 = nn.InstanceNorm2d(inplanes, affine=True)

        if scale == 1:
            self.nums = 1
        else:
            self.nums = scale - 1
        convs = []
        bns = []
        for i in range(self.nums):
            convs.append(nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=dilation, dilation=dilation))
            bns.append(nn.InstanceNorm2d(width, affine=True))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = nn.Conv2d(width * scale, planes * self.expansion, kernel_size=1)
        self.bn3 = nn.InstanceNorm2d(width * scale, affine=True)

        self.conv_st = nn.Conv2d(inplanes, planes * self.expansion, kernel_size=1)

        self.relu = nn.ELU(inplace=True)
        self.stype = stype
        self.scale = scale
        self.width = width
        self.shortcut = shortcut

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0 or self.stype == 'stage':
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.relu(self.bns[i](sp))
            sp = self.convs[i](sp)
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        out = torch.cat((out, spx[self.nums]), 1)
        if self.stype == 'stage':
            residual = self.conv_st(residual)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.shortcut:
            out += residual

        return out


class TriangleMultiplication(nn.Module):
    def __init__(self, in_dim=128, dim=128, direct='outgoing'):
        super(TriangleMultiplication, self).__init__()
        self.direct = direct
        self.norm = nn.LayerNorm(in_dim)
        self.linear1 = nn.Linear(in_dim, dim * 2)
        self.linear2 = nn.Sequential(
            nn.Linear(in_dim, dim * 2),
            nn.Sigmoid()
        )
        self.to_gate = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.Sigmoid()
        )
        self.linear_out = nn.Linear(dim, in_dim)
        # self.linear_out.weight.data.fill_(0.)
        # self.linear_out.bias.data.fill_(0.)
        self.to_out = nn.Sequential(
            nn.LayerNorm(dim),
            self.linear_out
        )

    def forward(self, z):
        direct = self.direct
        z = self.norm(z)
        a, b = torch.chunk(self.linear2(z) * self.linear1(z), 2, -1)
        gate = self.to_gate(z)
        if direct == 'outgoing':
            prod = torch.einsum('bikd,bjkd->bijd', a, b)
        elif direct == 'incoming':
            prod = torch.einsum('bkid,bkjd->bijd', a, b)
        else:
            raise ValueError('direct should be outgoing or incoming!')
        out = gate * self.to_out(prod)
        return out


class TriangleAttention(nn.Module):
    def __init__(self, in_dim=128, dim=32, n_heads=4, wise='row'):
        super(TriangleAttention, self).__init__()
        self.n_heads = n_heads
        self.wise = wise
        self.norm = nn.LayerNorm(in_dim)
        self.to_qkv = nn.Linear(in_dim, dim * 3 * n_heads, bias=False)
        self.linear_for_pair = nn.Linear(in_dim, n_heads, bias=False)
        self.to_gate = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.Sigmoid()
        )
        self.to_out = nn.Linear(n_heads * dim, in_dim)
        # self.to_out.weight.data.fill_(0.)
        # self.to_out.bias.data.fill_(0.)

    def forward(self, z):
        wise = self.wise
        z = self.norm(z)
        q, k, v = torch.chunk(self.to_qkv(z), 3, -1)
        q, k, v = map(lambda x: rearrange(x, 'b i j (h d)->b i j h d', h=self.n_heads), (q, k, v))
        b = self.linear_for_pair(z)
        gate = self.to_gate(z)
        scale = q.size(-1) ** .5
        if wise == 'row':
            eq_attn = 'brihd,brjhd->brijh'
            eq_multi = 'brijh,brjhd->brihd'
            b = rearrange(b, 'b i j (r h)->b r i j h', r=1)
            softmax_dim = 3
        elif wise == 'col':
            eq_attn = 'bilhd,bjlhd->bijlh'
            eq_multi = 'bijlh,bjlhd->bilhd'
            b = rearrange(b, 'b i j (l h)->b i j l h', l=1)
            softmax_dim = 2

        else:
            raise ValueError('wise should be col or row!')
        attn = (torch.einsum(eq_attn, q, k) / scale + b).softmax(softmax_dim)
        out = torch.einsum(eq_multi, attn, v)
        out = gate * rearrange(out, 'b i j h d-> b i j (h d)')
        z_ = self.to_out(out)
        return z_


class PairTransition(nn.Module):
    def __init__(self, dim=128, n=4):
        super(PairTransition, self).__init__()
        self.norm = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(dim, dim * n)
        self.linear2 = nn.Sequential(
            nn.ReLU(),
            nn.Linear(dim * n, dim)
        )

    def forward(self, z):
        z = self.norm(z)
        a = self.linear1(z)
        z = self.linear2(a)
        return z


def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)


class PreNormCross(nn.Module):
    def __init__(self, dim1, dim2, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim1)
        self.norm_context = nn.LayerNorm(dim2)

    def forward(self, x, context, *args, **kwargs):
        x = self.norm(x)
        context = self.norm_context(context)
        return self.fn(x, context, *args, **kwargs)


class FeedForward(nn.Module):
    def __init__(
            self,
            dim,
            mult=4,
            dropout=0.
    ):
        super().__init__()
        self.feed_forward = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x, **kwargs):
        return self.feed_forward(x)


class TriUpdate(nn.Module):
    def __init__(
            self,
            in_dim=128,
            n_heads=4,
            dim_pair_multi=64,
            dropout_rate_pair=.10,
    ):
        super(TriUpdate, self).__init__()

        self.ps_dropout_row_layer = DropoutRowwise(dropout_rate_pair)
        self.ps_dropout_col_layer = DropoutColumnwise(dropout_rate_pair)

        self.pair_multi_out = TriangleMultiplication(in_dim=in_dim, dim=dim_pair_multi, direct='outgoing')
        self.pair_multi_in = TriangleMultiplication(in_dim=in_dim, dim=dim_pair_multi, direct='incoming')

        dim_pair_attn = in_dim / n_heads
        assert dim_pair_attn == int(dim_pair_attn)
        self.pair_row_attn = TriangleAttention(in_dim=in_dim, dim=int(dim_pair_attn), n_heads=n_heads, wise='row')
        self.pair_col_attn = TriangleAttention(in_dim=in_dim, dim=int(dim_pair_attn), n_heads=n_heads, wise='col')

        self.pair_trans = PairTransition(dim=in_dim)

        self.conv_stem = nn.ModuleList(
            [
                nn.Sequential(
                    Rearrange('b i j d->b d i j'),
                    Bottle2neck(in_dim, in_dim, expansion=1, dilation=1, shortcut=False),
                    Rearrange('b d i j->b i j d'),
                )
                for _ in range(4)
            ]
        )

    def forward(self, z):
        z = z + self.ps_dropout_row_layer(self.pair_multi_out(z)) + self.conv_stem[0](z)
        z = z + self.ps_dropout_row_layer(self.pair_multi_in(z)) + self.conv_stem[1](z)
        pair_row_attn = self.pair_row_attn
        if z.requires_grad:
            z = z + self.ps_dropout_row_layer(checkpoint(pair_row_attn, z)) + self.conv_stem[2](z)
        else:
            z = z + self.ps_dropout_row_layer(pair_row_attn(z)) + self.conv_stem[2](z)
        pair_col_attn = self.pair_col_attn
        if z.requires_grad:
            z = z + self.ps_dropout_row_layer(checkpoint(pair_col_attn, z)) + self.conv_stem[3](z)
        else:
            z = z + self.ps_dropout_row_layer(pair_col_attn(z)) + self.conv_stem[3](z)
        z = z + self.pair_trans(z)

        return z


class SelfAttention(nn.Module):
    def __init__(
            self,
            dim,
            dim_pair=None,
            heads=8,
            dim_head=64,
            dropout=0.,
            tie_attn_dim=None
    ):
        super().__init__()

        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.heads = heads
        self.to_out = nn.Linear(inner_dim, dim)
        self.pair_norm = nn.LayerNorm(dim_pair)
        self.pair_linear = nn.Linear(dim_pair, heads, bias=False)

        self.for_pair = nn.Sequential(
            self.pair_norm, self.pair_linear
        )

        self.dropout = nn.Dropout(dropout)

        self.tie_attn_dim = tie_attn_dim
        self.seq_weight = PositionalWiseWeight(n_heads=heads, d_msa=dim)

    def forward(self, *args, context=None, tie_attn_dim=None, return_attn=False, soft_tied=False):
        if len(args) == 2:
            x, pair_bias = args
        elif len(args) == 1:
            x, pair_bias = args[0], None
        device, orig_shape, h, has_context = x.device, x.shape, self.heads, exists(context)
        # orig: (B*R, L, D)
        context = default(context, x)

        q, k, v = (self.to_q(x), *self.to_kv(context).chunk(2, dim=-1))
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))

        # for tying row-attention, for MSA axial self-attention

        if exists(tie_attn_dim):
            q, k, v = map(lambda t: rearrange(t, '(b r) h n d -> b r h n d', r=tie_attn_dim), (q, k, v))
            if soft_tied:
                w = self.seq_weight(rearrange(x, '(b r) l d -> b r l d', r=tie_attn_dim))  # b, L, H, R
                dots = torch.einsum('b i h r, b r h i d, b r h j d -> b h i j', w, q, k) * self.scale
            else:
                dots = torch.einsum('b r h i d, b r h j d -> b h i j', q, k) * self.scale * (tie_attn_dim ** -0.5)

        else:
            # q, k, v = map(lambda t: rearrange(t, '(b r) h n d -> b r h n d', r=tie_attn_dim), (q, k, v))

            #  SA:(B R H L D), (B R H L D) -> (B H R L L)
            dots = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale  # b=R

        # attention
        if pair_bias is not None:
            dots += rearrange(self.for_pair(pair_bias), 'b i j h -> b h i j')  # b=1
        attn = dots.softmax(dim=-1)
        attn = self.dropout(attn)

        # aggregate

        if exists(tie_attn_dim):
            out = torch.einsum('b h i j, b r h j d -> b r h i d', attn, v)
            out = rearrange(out, 'b r h n d -> (b r) h n d')
        else:
            out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)

        # combine heads and project out
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)

        if return_attn:
            return rearrange(out, '(b r) n d -> b r n d', b=1), attn.mean(0)
        else:
            return rearrange(out, '(b r) n d -> b r n d', b=1)


class MSAAttention(nn.Module):
    def __init__(
            self,
            tie_row_attn=False,
            attn_class=SelfAttention,
            dim=64,
            **kwargs
    ):
        super().__init__()

        self.tie_row_attn = tie_row_attn  # tie the row attention, from the paper 'MSA Transformer'

        self.attn_width = attn_class(dim, **kwargs)
        self.attn_height = attn_class(dim, **kwargs)

    def forward(self, *args, return_attn=False):
        if len(args) == 2:
            x, pair_bias = args
        if len(args) == 1:
            x, pair_bias = args[0], None
        if len(x.shape) == 5:
            assert x.size(1) == 1, f'x has shape {x.size()}!'
            x = x[:, 0, ...]

        b, h, w, d = x.size()

        # col-wise
        w_x = rearrange(x, 'b h w d -> (b w) h d')
        if w_x.requires_grad:
            w_out = checkpoint(self.attn_width, w_x)
        else:
            w_out = self.attn_width(w_x)

        # row-wise
        tie_attn_dim = x.shape[1] if self.tie_row_attn else None
        h_x = rearrange(x, 'b h w d -> (b h) w d')
        attn_height = partial(self.attn_height, tie_attn_dim=tie_attn_dim, return_attn=return_attn)
        # h_out, attn = self.attn_height(h_x, pair_bias, tie_attn_dim=tie_attn_dim, soft_tied=self.soft_tied, return_attn=return_attn)
        if h_x.requires_grad:
            h_out = checkpoint(attn_height, h_x, pair_bias)
        else:
            h_out = attn_height(h_x, pair_bias)
        if return_attn:
            h_out, attn = h_out
        # h_out = rearrange(h_out, '(b t h) w d -> b t h w d', h=h, w=w, t=t)

        out = w_out.permute(0, 2, 1, 3) + h_out
        out /= 2
        if return_attn:
            return out, attn
        return out


class PositionalWiseWeight(nn.Module):
    def __init__(self, d_msa=128, n_heads=4):
        super(PositionalWiseWeight, self).__init__()
        self.to_q = nn.Linear(d_msa, d_msa)
        self.to_k = nn.Linear(d_msa, d_msa)
        self.n_heads = n_heads

    def forward(self, m):
        q = self.to_q(m[:, 0:1, :, :])  # b,1,L,d
        k = self.to_k(m)  # b,L,L,d

        q = rearrange(q, 'b i j (h d) -> b j h i d', h=self.n_heads)
        k = rearrange(k, 'b i j (h d) -> b j h i d', h=self.n_heads)
        scale = q.size(-1) ** .5
        attn = torch.einsum('bjhud,bjhid->bjhi', q, k) / scale
        return attn.softmax(dim=-1)  # b, L, H, R


class UpdateX(nn.Module):
    def __init__(self, in_dim=128, dim_msa=32, dim=128):
        super(UpdateX, self).__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.proj_down1 = nn.Linear(in_dim, dim_msa)
        # self.seq_weight = PositionalWiseWeight(in_dim)
        # self.proj_down2 = nn.Linear(dim_msa ** 2 * 4 + dim + 8, dim)
        self.proj_down2 = nn.Linear(dim_msa ** 2, dim)
        self.elu = nn.ELU(inplace=True)
        self.bn1 = nn.InstanceNorm2d(dim, affine=True)
        self.conv1 = nn.Conv2d(dim, dim, 3, padding=1)
        self.bn2 = nn.InstanceNorm2d(dim, affine=True)
        self.conv2 = nn.Conv2d(dim, dim, 3, padding=1)

    def forward(self, x, m, w=None):
        m = self.proj_down1(m)  # b,r,l,d
        nrows = m.shape[1]
        outer_product = torch.einsum('brid,brjc -> bijcd', m, m) / nrows
        outer_product = rearrange(outer_product, 'b i j c d -> b i j (c d)')
        outer_product = self.proj_down2(outer_product)
        # pair_feats = torch.cat([x, outer_product], dim=-1)
        pair_feats = x + outer_product
        # pair_feats = rearrange(pair_feats,'b i j d -> b d i j')
        # out = self.bn1(pair_feats)
        # out = self.elu(out)
        # out = self.conv1(out)
        # out = self.bn2(out)
        # out = self.elu(out)
        # out = self.conv2(out)
        # return rearrange(pair_feats + out, 'b d i j -> b i j d')
        return pair_feats


class UpdateM(nn.Module):
    def __init__(self, in_dim=128, pair_dim=128, n_heads=8):
        super(UpdateM, self).__init__()
        self.norm1 = nn.LayerNorm(pair_dim)
        self.norm2 = nn.LayerNorm(in_dim)
        self.norm3 = nn.LayerNorm(in_dim)
        self.linear1 = nn.Linear(pair_dim, n_heads)
        self.linear2 = nn.Linear(in_dim, in_dim // n_heads)
        self.ff = FeedForward(in_dim, dropout=.1)
        self.n_heads = n_heads

    def forward(self, x, m):
        pair_feats = (x + rearrange(x, 'b i j d->b j i d')) / 2
        pair_feats = self.norm1(pair_feats)
        attn = self.linear1(pair_feats).softmax(-2)  # b i j h
        values = self.norm2(m)
        values = self.linear2(values)  # b r l d
        attn_out = torch.einsum('bijh,brjd->brihd', attn, values)
        attn_out = rearrange(attn_out, 'b r l h d -> b r l (h d)')
        out = m + attn_out
        residue = self.norm3(out)
        return out + self.ff(residue)


class relpos(nn.Module):

    def __init__(self, dim=128):
        super(relpos, self).__init__()
        self.linear = nn.Linear(65, dim)

    def forward(self, res_id):
        device = res_id.device
        bin_values = torch.arange(-32, 33, device=device)
        d = res_id[:, :, None] - res_id[:, None, :]
        bdy = torch.tensor(32, device=device)
        d = torch.minimum(torch.maximum(-bdy, d), bdy)
        d_onehot = (d[..., None] == bin_values).float()
        assert d_onehot.sum(dim=-1).min() == 1
        p = self.linear(d_onehot)
        return p


class RelPos(nn.Module):
    def __init__(self, dim):
        super(RelPos, self).__init__()
        self.relpos = relpos(dim=dim)

    def forward(self, z, res_id):
        z = z + self.relpos(res_id)
        return z


# main class
class BasicBlock(nn.Module):
    def __init__(self, dim=64, heads=8, dim_head=32, msa_tie_row_attn=False, msa_conv=None, attn_dropout=.1,
                 ff_dropout=.1):
        super().__init__()
        prenorm = partial(PreNorm, dim)

        self.PairMSA2MSA = prenorm(
            MSAAttention(dim=dim, dim_pair=dim, heads=heads, dim_head=dim_head, dropout=attn_dropout,
                         tie_row_attn=msa_tie_row_attn
                         ))
        self.MSA_FF = prenorm(FeedForward(dim=dim, dropout=ff_dropout))
        self.MSA2Pair = UpdateX(in_dim=dim, dim=dim)
        self.Pair2Pair = TriUpdate(in_dim=dim, dropout_rate_pair=attn_dropout)
        self.Pair2MSA = UpdateM(in_dim=dim, pair_dim=dim)

    def forward(self, msa, pair, return_attn=False):
        if return_attn:
            m_out, attn_map = self.PairMSA2MSA(msa, pair, return_attn=True)
            attn_map = rearrange(attn_map, 'h i j -> i j h')
        else:
            m_out = self.PairMSA2MSA(msa, pair, return_attn=False)
        msa = msa + m_out
        msa = msa + self.MSA_FF(msa)
        pair = self.MSA2Pair(pair, msa)
        pair = self.Pair2Pair(pair)
        msa = self.Pair2MSA(pair, msa)
        if return_attn:
            return msa, pair, attn_map
        return msa, pair