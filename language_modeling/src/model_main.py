# Initial code from https://github.com/kimiyoung/transformer-xl
# All fast weight models added by Kazuki Irie
#
# Main code for the model
# Some duplications in the attention function for a better readability...

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.proj_adaptive_softmax import ProjectedAdaptiveLogSoftmax
from utils.log_uniform_sampler import LogUniformSampler, sample_logits
from utils.performer_helper import prime, draw_orthogonal_random_matrix
from utils.model_utils import PositionalEmbedding, PositionwiseFF

from utils.fast_weight import StepWiseLinearTransformerLayer
from utils.fast_weight import StepWiseDPFPLinearTransformerLayer
from utils.fast_weight import DebugStepWiseLinearTransformerLayer
from utils.cuda_fast_weight_layer import CudaFastWeightLinearTransformerLayer
from utils.cuda_fast_weight_layer import CudaFastWeightPerformerLayer
from utils.cuda_fast_weight_layer import CudaFastWeightSumLinearTransformerLayer
from utils.cuda_fast_weight_layer import CudaFastWeightSumPerformerLayer
from utils.cuda_fast_weight_layer import CudaNormFastWeightLinearTransformerLayer
from utils.cuda_fast_weight_layer import CudaNormFastWeightPerformerLayer
from utils.cuda_fast_weight_layer import CudaFastWeightDPFPTransformerLayer
from utils.cuda_fast_weight_layer import CudaNormFastWeightDPFPTransformerLayer

# Delta MLP
from utils.cuda_fast_weight_layer import CudaDeepFastNetLayer
# Delta Delta Net
from utils.cuda_fast_weight_layer import CudaDeltaDeltaLayer

# Delta RNNs
from utils.cuda_fast_weight_layer import CudaFastRNNLayer
from utils.cuda_fast_weight_layer import CudaFastRNNv2Layer

# Delta LSTMs
from utils.cuda_fast_weight_layer import CudaFastLSTMLayer
from utils.cuda_fast_weight_layer import CudaFastLSTMv2Layer
from utils.cuda_fast_weight_layer import CudaFastLSTMv3Layer
from utils.cuda_fast_weight_layer import CudaFastLSTMv4Layer

from utils.cuda_fast_weight_layer import CudaFastWeightSlowRNNLayer
# Recurrent Delta Net
from utils.cuda_fast_weight_layer import CudaFastWeightRecUpdateTanhLayer

from utils.model_utils import LSTMLayer, RNNLayer


# Standard multihead attention.
class MultiHeadAttn(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout, dropatt=0, 
                 pre_lnorm=False):
        super(MultiHeadAttn, self).__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout

        self.q_net = nn.Linear(d_model, n_head * d_head, bias=False)
        self.kv_net = nn.Linear(d_model, 2 * n_head * d_head, bias=False)

        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)

        self.layer_norm = nn.LayerNorm(d_model)

        self.scale = 1 / (d_head ** 0.5)

        self.pre_lnorm = pre_lnorm

    def forward(self, h, attn_mask=None, mems=None,
                carry_over_fast_weight=False):
        assert not carry_over_fast_weight, "Not supported."
        # multihead attention
        # [hlen x bsz x n_head x d_head]

        if mems is not None:
            c = torch.cat([mems, h], 0)
        else:
            c = h

        if self.pre_lnorm:
            # layer normalization
            c = self.layer_norm(c)

        head_q = self.q_net(h)
        head_k, head_v = torch.chunk(self.kv_net(c), 2, -1)

        head_q = head_q.view(h.size(0), h.size(1), self.n_head, self.d_head)
        head_k = head_k.view(c.size(0), c.size(1), self.n_head, self.d_head)
        head_v = head_v.view(c.size(0), c.size(1), self.n_head, self.d_head)

        # [qlen x klen x bsz x n_head]
        attn_score = torch.einsum('ibnd,jbnd->ijbn', (head_q, head_k))
        attn_score.mul_(self.scale)
        if attn_mask is not None and attn_mask.any().item():
            if attn_mask.dim() == 2:
                attn_score.masked_fill_(
                    attn_mask[None,:,:,None], -float('inf'))
            elif attn_mask.dim() == 3:
                attn_score.masked_fill_(attn_mask[:,:,:,None], -float('inf'))

        # [qlen x klen x bsz x n_head]
        attn_prob = F.softmax(attn_score, dim=1)
        attn_prob = self.dropatt(attn_prob)

        # [qlen x klen x bsz x n_head] + [klen x bsz x n_head x d_head]
        # -> [qlen x bsz x n_head x d_head]
        attn_vec = torch.einsum('ijbn,jbnd->ibnd', (attn_prob, head_v))
        attn_vec = attn_vec.contiguous().view(
            attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)

        # linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            # residual connection
            output = h + attn_out
        else:
            # residual connection + layer normalization
            output = self.layer_norm(h + attn_out)

        return output


# Linear multihead attention from Katharopoulos et al.
# Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention
# https://arxiv.org/abs/2006.16236
class LinearMultiHeadAttn(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout, dropatt=0,
                 pre_lnorm=False):
        super(LinearMultiHeadAttn, self).__init__()
        print("Using LinearMultiHeadAttn --")
        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout

        self.q_net = nn.Linear(d_model, n_head * d_head, bias=False)
        self.kv_net = nn.Linear(d_model, 2 * n_head * d_head, bias=False)

        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)

        self.layer_norm = nn.LayerNorm(d_model)

        self.scale = 1 / (d_head ** 0.5)

        self.pre_lnorm = pre_lnorm

    def forward(self, h, attn_mask=None, mems=None,
                carry_over_fast_weight=False):
        assert not carry_over_fast_weight, "Not supported."
        # multihead attention
        # [hlen x bsz x n_head x d_head]
        if mems is not None:
            c = torch.cat([mems, h], 0)
        else:
            c = h

        if self.pre_lnorm:
            # layer normalization
            c = self.layer_norm(c)

        head_q = self.q_net(h)
        head_k, head_v = torch.chunk(self.kv_net(c), 2, -1)

        head_q = head_q.view(h.size(0), h.size(1), self.n_head, self.d_head)
        head_k = head_k.view(c.size(0), c.size(1), self.n_head, self.d_head)
        head_v = head_v.view(c.size(0), c.size(1), self.n_head, self.d_head)

        # transform q and k
        head_q = F.elu(head_q, 1., False) + 1.
        head_k = F.elu(head_k, 1., False) + 1.

        # [qlen x klen x bsz x n_head]
        attn_score = torch.einsum('ibnd,jbnd->ijbn', (head_q, head_k))
        attn_score.mul_(self.scale)
        if attn_mask is not None and attn_mask.any().item():
            if attn_mask.dim() == 2:
                # masked position to 0
                attn_score.masked_fill_(attn_mask[None, :, :, None], 0)
            elif attn_mask.dim() == 3:
                attn_score.masked_fill_(attn_mask[:, :, :, None], 0)

        # normalize attn scores over keys
        eps = 1e-5
        denominator = torch.sum(attn_score, 1, keepdim=True) + eps
        # get (q_len, 1, B, n_head)

        attn_score = self.dropatt(attn_score)  # change
        attn_prob = attn_score / denominator

        # [qlen x klen x bsz x n_head] + [klen x bsz x n_head x d_head]
        # -> [qlen x bsz x n_head x d_head]
        attn_vec = torch.einsum('ijbn,jbnd->ibnd', (attn_prob, head_v))
        attn_vec = attn_vec.contiguous().view(
            attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)

        # linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            # residual connection
            output = h + attn_out
        else:
            # residual connection + layer normalization
            output = self.layer_norm(h + attn_out)

        return output


# DPFP linear attention.
class DPFPMultiHeadAttn(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout, dropatt=0,
                 pre_lnorm=False, n_roll=3):
        super(DPFPMultiHeadAttn, self).__init__()
        print(f"Using DPFPMultiHeadAttn with {n_roll} rolls --")
        self.n_roll = n_roll
        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout

        self.q_net = nn.Linear(d_model, n_head * d_head, bias=False)
        self.kv_net = nn.Linear(d_model, 2 * n_head * d_head, bias=False)

        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)

        self.layer_norm = nn.LayerNorm(d_model)

        self.scale = 1 / (d_head ** 0.5)

        self.pre_lnorm = pre_lnorm

    def mul_roll_repeat(self, x):
        rolls = []
        for i in range(1, self.n_roll + 1):
            rolls.append(x * x.roll(shifts=i, dims=-1))
        return torch.cat(rolls, dim=-1)

    def forward(self, h, attn_mask=None, mems=None,
                carry_over_fast_weight=False):
        assert not carry_over_fast_weight, "Not supported."
        # multihead attention
        # [hlen x bsz x n_head x d_head]

        if mems is not None:
            c = torch.cat([mems, h], 0)
        else:
            c = h

        if self.pre_lnorm:
            # layer normalization
            c = self.layer_norm(c)

        head_q = self.q_net(h)
        head_k, head_v = torch.chunk(self.kv_net(c), 2, -1)

        head_q = head_q.view(h.size(0), h.size(1), self.n_head, self.d_head)
        head_k = head_k.view(c.size(0), c.size(1), self.n_head, self.d_head)
        head_v = head_v.view(c.size(0), c.size(1), self.n_head, self.d_head)

        # transform q and k
        act = lambda x: F.relu(x)  # relu or exp
        head_k = torch.cat([act(head_k), act(-head_k)], dim=-1)
        head_q = torch.cat([act(head_q), act(-head_q)], dim=-1)

        head_k = self.mul_roll_repeat(head_k)
        head_q = self.mul_roll_repeat(head_q)

        # [qlen x klen x bsz x n_head]
        attn_score = torch.einsum('ibnd,jbnd->ijbn', (head_q, head_k))
        attn_score.mul_(self.scale)
        if attn_mask is not None and attn_mask.any().item():
            if attn_mask.dim() == 2:
                # masked position to 0
                attn_score.masked_fill_(attn_mask[None, :, :, None], 0)
            elif attn_mask.dim() == 3:
                attn_score.masked_fill_(attn_mask[:, :, :, None], 0)

        # normalize attn scores over keys
        eps = 1e-5
        denominator = torch.sum(attn_score, 1, keepdim=True) + eps
        # get (q_len, 1, B, n_head)

        attn_score = self.dropatt(attn_score)  # change
        attn_prob = attn_score / denominator

        # [qlen x klen x bsz x n_head] + [klen x bsz x n_head x d_head]
        # -> [qlen x bsz x n_head x d_head]
        attn_vec = torch.einsum('ijbn,jbnd->ibnd', (attn_prob, head_v))
        attn_vec = attn_vec.contiguous().view(
            attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)

        # linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            # residual connection
            output = h + attn_out
        else:
            # residual connection + layer normalization
            output = self.layer_norm(h + attn_out)

        return output


# Performer multihead attention from Choromanski et al.
# Rethinking Attention with Performers. https://arxiv.org/abs/2009.14794
class PerformerMultiHeadAttn(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout, dropatt=0,
                 pre_lnorm=False, proj_dim=256, device='cuda',
                 skip_attn_normalization=False):
        assert not skip_attn_normalization, "Not implemented."
        # proj_dim: projected dimension
        print(f"Using PerformerMultiHeadAttn -- proj_dim: {proj_dim}")

        super(PerformerMultiHeadAttn, self).__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout

        self.q_net = nn.Linear(d_model, n_head * d_head, bias=False)
        self.kv_net = nn.Linear(d_model, 2 * n_head * d_head, bias=False)

        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)

        self.layer_norm = nn.LayerNorm(d_model)

        self.scale = 1 / (d_head ** 0.5)

        self.pre_lnorm = pre_lnorm

        self.proj_dim = proj_dim
        # so that we can keep the same matrix at test time.
        self.proj_matrix = draw_orthogonal_random_matrix(
            d_head, proj_dim, device=device)  # TODO store this as param?

    def forward(self, h, attn_mask=None, mems=None, redraw=True,
                carry_over_fast_weight=False):
        assert not carry_over_fast_weight, "Not supported."
        # multihead attention
        # [hlen x bsz x n_head x d_head]

        if mems is not None:
            c = torch.cat([mems, h], 0)
        else:
            c = h

        if self.pre_lnorm:
            # layer normalization
            c = self.layer_norm(c)

        head_q = self.q_net(h)
        head_k, head_v = torch.chunk(self.kv_net(c), 2, -1)

        head_q = head_q.view(h.size(0), h.size(1), self.n_head, self.d_head)
        head_k = head_k.view(c.size(0), c.size(1), self.n_head, self.d_head)
        head_v = head_v.view(c.size(0), c.size(1), self.n_head, self.d_head)

        if redraw:
            self.proj_matrix = draw_orthogonal_random_matrix(
                self.d_head, self.proj_dim, device=h.device)

        # transform q and k
        head_q = prime(head_q, self.proj_matrix)  # (len, B, n_head, proj_dim)
        head_k = prime(head_k, self.proj_matrix)

        # [qlen x klen x bsz x n_head]
        attn_score = torch.einsum('ibnd,jbnd->ijbn', (head_q, head_k))
        attn_score.mul_(self.scale)
        if attn_mask is not None and attn_mask.any().item():
            if attn_mask.dim() == 2:
                # set masked positions to 0
                attn_score.masked_fill_(attn_mask[None, :, :, None], 0)
            elif attn_mask.dim() == 3:
                attn_score.masked_fill_(attn_mask[:, :, :, None], 0)

        # normalize attn scores over keys
        eps = 1e-5
        denominator = torch.sum(attn_score, 1, keepdim=True) + eps
        # get (q_len, 1, B, n_head)

        attn_score = self.dropatt(attn_score)  # change
        attn_prob = attn_score / denominator

        # [qlen x klen x bsz x n_head] + [klen x bsz x n_head x d_head]
        # -> [qlen x bsz x n_head x d_head]
        attn_vec = torch.einsum('ijbn,jbnd->ibnd', (attn_prob, head_v))
        attn_vec = attn_vec.contiguous().view(
            attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)

        # linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            # residual connection
            output = h + attn_out
        else:
            # residual connection + layer normalization
            output = self.layer_norm(h + attn_out)

        return output


class RelMultiHeadAttn(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout, dropatt=0,
                 tgt_len=None, ext_len=None, mem_len=None, pre_lnorm=False):
        super(RelMultiHeadAttn, self).__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout

        self.qkv_net = nn.Linear(d_model, 3 * n_head * d_head, bias=False)

        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)

        self.layer_norm = nn.LayerNorm(d_model)

        self.scale = 1 / (d_head ** 0.5)

        self.pre_lnorm = pre_lnorm

    def _parallelogram_mask(self, h, w, left=False):
        mask = torch.ones((h, w)).byte()
        m = min(h, w)
        mask[:m,:m] = torch.triu(mask[:m,:m])
        mask[-m:,-m:] = torch.tril(mask[-m:,-m:])

        if left:
            return mask
        else:
            return mask.flip(0)

    def _shift(self, x, qlen, klen, mask, left=False):
        if qlen > 1:
            zero_pad = torch.zeros((x.size(0), qlen-1, x.size(2), x.size(3)),
                                    device=x.device, dtype=x.dtype)
        else:
            zero_pad = torch.zeros(0, device=x.device, dtype=x.dtype)

        if left:
            mask = mask.flip(1)
            x_padded = torch.cat([zero_pad, x], dim=1).expand(qlen, -1, -1, -1)
        else:
            x_padded = torch.cat([x, zero_pad], dim=1).expand(qlen, -1, -1, -1)

        x = x_padded.masked_select(mask[:,:,None,None]) \
                    .view(qlen, klen, x.size(2), x.size(3))

        return x

    def _rel_shift(self, x, zero_triu=False):
        zero_pad = torch.zeros((x.size(0), 1, *x.size()[2:]),
                               device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=1)

        x_padded = x_padded.view(x.size(1) + 1, x.size(0), *x.size()[2:])

        x = x_padded[1:].view_as(x)

        if zero_triu:
            ones = torch.ones((x.size(0), x.size(1)))
            x = x * torch.tril(ones, x.size(1) - x.size(0))[:,:,None,None]

        return x

    def forward(self, w, r, attn_mask=None, mems=None):
        raise NotImplementedError


class RelPartialLearnableMultiHeadAttn(RelMultiHeadAttn):
    def __init__(self, *args, **kwargs):
        super(RelPartialLearnableMultiHeadAttn, self).__init__(*args, **kwargs)

        self.r_net = nn.Linear(
            self.d_model, self.n_head * self.d_head, bias=False)

    def forward(self, w, r, r_w_bias, r_r_bias, attn_mask=None, mems=None):
        qlen, rlen, bsz = w.size(0), r.size(0), w.size(1)

        if mems is not None:
            cat = torch.cat([mems, w], 0)
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(cat))
            else:
                w_heads = self.qkv_net(cat)
            r_head_k = self.r_net(r)

            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)
            w_head_q = w_head_q[-qlen:]
        else:
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(w))
            else:
                w_heads = self.qkv_net(w)
            r_head_k = self.r_net(r)

            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)

        klen = w_head_k.size(0)

        w_head_q = w_head_q.view(qlen, bsz, self.n_head, self.d_head)
        w_head_k = w_head_k.view(klen, bsz, self.n_head, self.d_head)
        w_head_v = w_head_v.view(klen, bsz, self.n_head, self.d_head)

        r_head_k = r_head_k.view(rlen, self.n_head, self.d_head)

        # compute attention score
        rw_head_q = w_head_q + r_w_bias
        AC = torch.einsum('ibnd,jbnd->ijbn', (rw_head_q, w_head_k))

        rr_head_q = w_head_q + r_r_bias
        BD = torch.einsum('ibnd,jnd->ijbn', (rr_head_q, r_head_k))
        BD = self._rel_shift(BD)

        # [qlen x klen x bsz x n_head]
        attn_score = AC + BD
        attn_score.mul_(self.scale)

        # compute attention probability
        if attn_mask is not None and attn_mask.any().item():
            if attn_mask.dim() == 2:
                attn_score = attn_score.float().masked_fill(
                    attn_mask[None,:,:,None], -float('inf')).type_as(
                    attn_score)

            elif attn_mask.dim() == 3:
                attn_score = attn_score.float().masked_fill(
                    attn_mask[:,:,:,None], -float('inf')).type_as(attn_score)

        # [qlen x klen x bsz x n_head]
        attn_prob = F.softmax(attn_score, dim=1)
        attn_prob = self.dropatt(attn_prob)

        # compute attention vector
        attn_vec = torch.einsum('ijbn,jbnd->ibnd', (attn_prob, w_head_v))

        # [qlen x bsz x n_head x d_head]
        attn_vec = attn_vec.contiguous().view(
            attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)

        # linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            # residual connection
            output = w + attn_out
        else:
            # residual connection + layer normalization
            output = self.layer_norm(w + attn_out)

        return output


class RelLearnableMultiHeadAttn(RelMultiHeadAttn):
    def __init__(self, *args, **kwargs):
        super(RelLearnableMultiHeadAttn, self).__init__(*args, **kwargs)

    def forward(self, w, r_emb, r_w_bias, r_bias, attn_mask=None, mems=None):
        # r_emb: [klen, n_head, d_head], used for term B
        # r_w_bias: [n_head, d_head], used for term C
        # r_bias: [klen, n_head], used for term D

        qlen, bsz = w.size(0), w.size(1)

        if mems is not None:
            cat = torch.cat([mems, w], 0)
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(cat))
            else:
                w_heads = self.qkv_net(cat)
            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)

            w_head_q = w_head_q[-qlen:]
        else:
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(w))
            else:
                w_heads = self.qkv_net(w)
            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)

        klen = w_head_k.size(0)

        w_head_q = w_head_q.view(qlen, bsz, self.n_head, self.d_head)
        w_head_k = w_head_k.view(klen, bsz, self.n_head, self.d_head)
        w_head_v = w_head_v.view(klen, bsz, self.n_head, self.d_head)

        if klen > r_emb.size(0):
            r_emb_pad = r_emb[0:1].expand(klen-r_emb.size(0), -1, -1)
            r_emb = torch.cat([r_emb_pad, r_emb], 0)
            r_bias_pad = r_bias[0:1].expand(klen-r_bias.size(0), -1)
            r_bias = torch.cat([r_bias_pad, r_bias], 0)
        else:
            r_emb = r_emb[-klen:]
            r_bias = r_bias[-klen:]

        # compute attention score
        # qlen x bsz x n_head x d_head
        rw_head_q = w_head_q + r_w_bias[None]

        # qlen x klen x bsz x n_head
        AC = torch.einsum('ibnd,jbnd->ijbn', (rw_head_q, w_head_k))
        # qlen x klen x bsz x n_head
        B_ = torch.einsum('ibnd,jnd->ijbn', (w_head_q, r_emb))
        # 1    x klen x 1   x n_head
        D_ = r_bias[None, :, None]
        BD = self._rel_shift(B_ + D_)

        # [qlen x klen x bsz x n_head]
        attn_score = AC + BD
        attn_score.mul_(self.scale)

        # compute attention probability
        if attn_mask is not None and attn_mask.any().item():
            if attn_mask.dim() == 2:
                attn_score.masked_fill_(
                    attn_mask[None,:,:,None], -float('inf'))
            elif attn_mask.dim() == 3:
                attn_score.masked_fill_(attn_mask[:,:,:,None], -float('inf'))

        # [qlen x klen x bsz x n_head]
        attn_prob = F.softmax(attn_score, dim=1)
        attn_prob = self.dropatt(attn_prob)

        # compute attention vector
        attn_vec = torch.einsum('ijbn,jbnd->ibnd', (attn_prob, w_head_v))

        # [qlen x bsz x n_head x d_head]
        attn_vec = attn_vec.contiguous().view(
            attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)

        # linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            # residual connection
            output = w + attn_out
        else:
            # residual connection + layer normalization
            output = self.layer_norm(w + attn_out)

        return output


class PerformerDecoderLayer(nn.Module):
    def __init__(self, n_head, d_model, d_head, d_inner, dropout, attn_type,
                 **kwargs):
        super(PerformerDecoderLayer, self).__init__()
        if attn_type == 25:
            attn_func = CudaFastWeightPerformerLayer
        elif attn_type == 35:
            attn_func = CudaFastWeightSumPerformerLayer
        elif attn_type == 45:
            attn_func = CudaNormFastWeightPerformerLayer
        elif attn_type == 5:
            attn_func = PerformerMultiHeadAttn
        else:
            raise Exception(f"attn_type {attn_type} not allowed "
                            f"in PerformerDecoderLayer.")

        self.dec_attn = attn_func(n_head, d_model, d_head, dropout, **kwargs)
        self.pos_ff = PositionwiseFF(
            d_model, d_inner, dropout, pre_lnorm=kwargs.get('pre_lnorm'))

    def forward(self, dec_inp, dec_attn_mask=None, mems=None, redraw=True,
                carry_over_fast_weight=False):

        output = self.dec_attn(
            dec_inp, attn_mask=dec_attn_mask, mems=mems, redraw=redraw,
            carry_over_fast_weight=carry_over_fast_weight)

        if carry_over_fast_weight:
            output, new_mem = output

        output = self.pos_ff(output)

        if carry_over_fast_weight:
            return output, new_mem

        return output


class DecoderLayer(nn.Module):
    def __init__(self, n_head, d_model, d_head, d_inner, dropout, attn_type,
                 use_ff=True, d_res=None, use_lnorm=True, use_res=True,
                 use_out_proj=True,
                 **kwargs):
        super(DecoderLayer, self).__init__()

        if attn_type == 2:
            attn_func = MultiHeadAttn
        elif attn_type == 4:
            attn_func = LinearMultiHeadAttn
        elif attn_type == 6:
            attn_func = DPFPMultiHeadAttn
        elif attn_type == 10:
            attn_func = DebugStepWiseLinearTransformerLayer
        elif attn_type == 14:
            attn_func = StepWiseLinearTransformerLayer
        elif attn_type == 16:
            attn_func = StepWiseDPFPLinearTransformerLayer
        elif attn_type == 24:
            attn_func = CudaFastWeightLinearTransformerLayer
        elif attn_type == 26:
            attn_func = CudaFastWeightDPFPTransformerLayer
        elif attn_type == 44:
            attn_func = CudaNormFastWeightLinearTransformerLayer
        elif attn_type == 46:
            attn_func = CudaNormFastWeightDPFPTransformerLayer
        elif attn_type == 34:
            attn_func = CudaFastWeightSumLinearTransformerLayer
        elif attn_type == 54:
            attn_func = CudaDeepFastNetLayer
        elif attn_type == 64:
            attn_func = CudaDeltaDeltaLayer
        elif attn_type == 80:
            attn_func = RNNLayer
        elif attn_type == 90:
            attn_func = LSTMLayer
        elif attn_type == 124:
            attn_func = CudaFastRNNLayer
        elif attn_type == 134:
            attn_func = CudaFastRNNv2Layer
        elif attn_type == 224:
            attn_func = CudaFastLSTMLayer
        elif attn_type == 234:
            attn_func = CudaFastLSTMv2Layer
        elif attn_type == 244:
            attn_func = CudaFastLSTMv3Layer  # with residual connection
        elif attn_type == 254:
            attn_func = CudaFastLSTMv4Layer  # no sigm on update
        elif attn_type == 324:
            attn_func = CudaFastWeightSlowRNNLayer  # no sigm on update
        elif attn_type == 924:
            assert False, "Removed"  # same as below without tanh
        elif attn_type == 934:
            attn_func = CudaFastWeightRecUpdateTanhLayer
        else:
            raise Exception(f"attn_type {attn_type} not allowed here.")

        if attn_type in [80, 90, 124, 224]:
            self.dec_attn = attn_func(
                n_head, d_model, d_head, dropout, d_res=d_res,
                use_out_proj=use_out_proj, use_lnorm=use_lnorm,
                use_res=use_res, **kwargs)
        else:
            self.dec_attn = attn_func(
                n_head, d_model, d_head, dropout, **kwargs)

        self.use_ff = use_ff
        if use_ff:
            self.pos_ff = PositionwiseFF(
                d_model, d_inner, dropout, pre_lnorm=kwargs.get('pre_lnorm'))

    def forward(self, dec_inp, dec_attn_mask=None, mems=None,
                carry_over_fast_weight=False):
        output = self.dec_attn(dec_inp, attn_mask=dec_attn_mask, mems=mems,
                               carry_over_fast_weight=carry_over_fast_weight)

        if carry_over_fast_weight:
            output, new_mem = output
        if self.use_ff:
            output = self.pos_ff(output)

        if carry_over_fast_weight:
            return output, new_mem
        return output


class RelLearnableDecoderLayer(nn.Module):
    def __init__(self, n_head, d_model, d_head, d_inner, dropout,
                 **kwargs):
        super(RelLearnableDecoderLayer, self).__init__()

        self.dec_attn = RelLearnableMultiHeadAttn(
            n_head, d_model, d_head, dropout, **kwargs)
        self.pos_ff = PositionwiseFF(
            d_model, d_inner, dropout, pre_lnorm=kwargs.get('pre_lnorm'))

    def forward(self, dec_inp, r_emb, r_w_bias, r_bias,
                dec_attn_mask=None, mems=None):

        output = self.dec_attn(dec_inp, r_emb, r_w_bias, r_bias,
                               attn_mask=dec_attn_mask, mems=mems)
        output = self.pos_ff(output)

        return output


class RelPartialLearnableDecoderLayer(nn.Module):
    def __init__(self, n_head, d_model, d_head, d_inner, dropout,
                 **kwargs):
        super(RelPartialLearnableDecoderLayer, self).__init__()

        self.dec_attn = RelPartialLearnableMultiHeadAttn(
            n_head, d_model, d_head, dropout, **kwargs)

        self.pos_ff = PositionwiseFF(
            d_model, d_inner, dropout, pre_lnorm=kwargs.get('pre_lnorm'))

    def forward(self, dec_inp, r, r_w_bias, r_r_bias, dec_attn_mask=None,
                mems=None):

        output = self.dec_attn(dec_inp, r, r_w_bias, r_r_bias,
                               attn_mask=dec_attn_mask, mems=mems)
        output = self.pos_ff(output)

        return output


class AdaptiveEmbedding(nn.Module):
    def __init__(self, n_token, d_embed, d_proj, cutoffs, div_val=1,
                 sample_softmax=False):
        super(AdaptiveEmbedding, self).__init__()

        self.n_token = n_token
        self.d_embed = d_embed

        self.cutoffs = cutoffs + [n_token]
        self.div_val = div_val
        self.d_proj = d_proj

        self.emb_scale = d_proj ** 0.5

        self.cutoff_ends = [0] + self.cutoffs

        self.emb_layers = nn.ModuleList()
        self.emb_projs = nn.ParameterList()
        if div_val == 1:
            self.emb_layers.append(
                nn.Embedding(n_token, d_embed, sparse=sample_softmax>0)
            )
            if d_proj != d_embed:
                self.emb_projs.append(
                    nn.Parameter(torch.Tensor(d_proj, d_embed)))
        else:
            for i in range(len(self.cutoffs)):
                l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i+1]
                d_emb_i = d_embed // (div_val ** i)
                self.emb_layers.append(nn.Embedding(r_idx-l_idx, d_emb_i))
                self.emb_projs.append(
                    nn.Parameter(torch.Tensor(d_proj, d_emb_i)))

    def forward(self, inp):
        if self.div_val == 1:
            embed = self.emb_layers[0](inp)
            if self.d_proj != self.d_embed:
                embed  = F.linear(embed, self.emb_projs[0])
        else:
            param = next(self.parameters())
            inp_flat = inp.view(-1)
            emb_flat = torch.zeros([inp_flat.size(0), self.d_proj], 
                dtype=param.dtype, device=param.device)
            for i in range(len(self.cutoffs)):
                l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]

                mask_i = (inp_flat >= l_idx) & (inp_flat < r_idx)
                indices_i = mask_i.nonzero().squeeze()

                if indices_i.numel() == 0:
                    continue

                inp_i = inp_flat.index_select(0, indices_i) - l_idx
                emb_i = self.emb_layers[i](inp_i)
                emb_i = F.linear(emb_i, self.emb_projs[i])

                emb_flat.index_copy_(0, indices_i, emb_i)

            embed = emb_flat.view(*inp.size(), self.d_proj)

        embed.mul_(self.emb_scale)

        return embed


class MemTransformerLM(nn.Module):
    def __init__(self,
                 n_token,
                 n_layer,
                 n_head,
                 d_model,
                 d_head,
                 d_inner,
                 dropout=0.0,
                 dropatt=0.0,
                 tie_weight=True,
                 d_embed=None,
                 div_val=1,
                 tie_projs=[False],
                 pre_lnorm=False,
                 remove_ff=False,  # remove ff block
                 remove_lnorm=False,  # for RNN and LSTM 80/90/124/224
                 remove_res=False,  # idem
                 remove_out_proj=False,  # idem
                 d_res=None,
                 tgt_len=None,
                 ext_len=None,
                 mem_len=None,
                 cutoffs=[],
                 adapt_inp=False,
                 same_length=False,
                 attn_type=0,
                 clamp_len=-1,
                 sample_softmax=-1,
                 proj_dim=256,  # for performer layers
                 n_roll=3,  # mirrored attention
                 skip_attn_normalization=False,
                 no_pos=False,  # no positional encoding
                 fast_net_depth=1,
                 use_slow_base_weights=False,
                 device='cuda'):
        super(MemTransformerLM, self).__init__()
        self.n_token = n_token
        self.no_pos = no_pos

        d_embed = d_model if d_embed is None else d_embed
        self.d_embed = d_embed
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_head

        # set residual connection dim
        if d_res is not None:
            self.d_res = d_res
        else:
            self.d_res = d_model

        # emb_out_dim = d_embed if remove_res else self.d_res
        # if there is no proj, there is no need for projection
        self.word_emb = AdaptiveEmbedding(
            n_token, d_embed, self.d_res, cutoffs, div_val=div_val)

        self.drop = nn.Dropout(dropout)

        self.n_layer = n_layer

        self.tgt_len = tgt_len
        self.mem_len = mem_len
        self.ext_len = ext_len
        self.max_klen = tgt_len + ext_len + mem_len

        self.attn_type = attn_type

        self.layers = nn.ModuleList()
        if attn_type == 0:  # the default attention
            for i in range(n_layer):
                self.layers.append(
                    RelPartialLearnableDecoderLayer(
                        n_head, d_model, d_head, d_inner, dropout,
                        tgt_len=tgt_len, ext_len=ext_len, mem_len=mem_len,
                        dropatt=dropatt, pre_lnorm=pre_lnorm)
                )
        elif attn_type == 1:  # learnable embeddings
            for i in range(n_layer):
                self.layers.append(
                    RelLearnableDecoderLayer(
                        n_head, d_model, d_head, d_inner, dropout,
                        tgt_len=tgt_len, ext_len=ext_len, mem_len=mem_len,
                        dropatt=dropatt, pre_lnorm=pre_lnorm)
                )
        elif attn_type in [2, 3, 4]:  # absolute embeddings
            # 2: baseline vanilla transformer
            # 3:
            # 4: linear transformer
            for i in range(n_layer):
                self.layers.append(
                    DecoderLayer(
                        n_head, d_model, d_head, d_inner, dropout,
                        dropatt=dropatt, pre_lnorm=pre_lnorm,
                        attn_type=attn_type)
                )
        elif attn_type in [6, 7]:  # absolute embeddings
            # 6: mirrored attention
            # 7: mirrored attention v2
            for i in range(n_layer):
                self.layers.append(
                    DecoderLayer(
                        n_head, d_model, d_head, d_inner, dropout,
                        dropatt=dropatt, pre_lnorm=pre_lnorm,
                        attn_type=attn_type, n_roll=n_roll)
                )
        elif attn_type in [10, 14, 24, 34, 44, 74, 80, 90,
                           104, 114, 124, 134, 144,
                           204, 224, 234, 244, 254,
                           324, 924, 934]:
            # fast weights
            # 10: debugging, same as linear trafo but step by step
            # 14: linear fast weight
            for i in range(n_layer):
                self.layers.append(
                    DecoderLayer(
                        n_head, d_model, d_head, d_inner, dropout,
                        dropatt=dropatt, pre_lnorm=pre_lnorm,
                        use_ff=(not remove_ff),
                        use_res=(not remove_res),
                        use_lnorm=(not remove_lnorm),
                        use_out_proj=(not remove_out_proj),
                        d_res=self.d_res,
                        attn_type=attn_type, layer_id=i, num_layer=n_layer,
                        skip_attn_normalization=skip_attn_normalization)
                )
        elif attn_type in [64]:  # transformer programmer
            for i in range(n_layer):
                self.layers.append(
                    DecoderLayer(
                        n_head, d_model, d_head, d_inner, dropout,
                        dropatt=dropatt, pre_lnorm=pre_lnorm,
                        attn_type=attn_type, layer_id=i, num_layer=n_layer,
                        skip_attn_normalization=skip_attn_normalization,
                        use_slow_base_weights=use_slow_base_weights)
                )
        elif attn_type in [54]:  # deep fast nets
            for i in range(n_layer):
                self.layers.append(
                    DecoderLayer(
                        n_head, d_model, d_head, d_inner, dropout,
                        dropatt=dropatt, pre_lnorm=pre_lnorm,
                        attn_type=attn_type, layer_id=i, num_layer=n_layer,
                        skip_attn_normalization=skip_attn_normalization,
                        fast_net_depth=fast_net_depth)
                )
        elif attn_type in [16, 26, 46]:  # fast weights w/ absolute embeddings
            # 10: debugging, same as linear trafo but step by step
            # 14: linear fast weight
            for i in range(n_layer):
                self.layers.append(
                    DecoderLayer(
                        n_head, d_model, d_head, d_inner, dropout,
                        dropatt=dropatt, pre_lnorm=pre_lnorm,
                        attn_type=attn_type, layer_id=i, num_layer=n_layer,
                        n_roll=n_roll,
                        skip_attn_normalization=skip_attn_normalization)
                )
        elif attn_type in [5, 25, 35, 45]:  # absolute embeddings, performer
            # performer case needs to be separate from the case above
            # such that we can del with custom logic for random projections.
            for i in range(n_layer):
                self.layers.append(
                    PerformerDecoderLayer(
                        n_head, d_model, d_head, d_inner, dropout,
                        dropatt=dropatt, pre_lnorm=pre_lnorm,
                        attn_type=attn_type, proj_dim=proj_dim, device=device,
                        skip_attn_normalization=skip_attn_normalization)
                )

        self.sample_softmax = sample_softmax
        # use sampled softmax
        if sample_softmax > 0:
            self.out_layer = nn.Linear(self.d_res, n_token)
            if tie_weight:
                self.out_layer.weight = self.word_emb.weight
            self.tie_weight = tie_weight
            self.sampler = LogUniformSampler(n_token, sample_softmax)
            self.crit = None

        # use adaptive softmax (including standard softmax)
        else:
            self.crit = ProjectedAdaptiveLogSoftmax(n_token, d_embed, self.d_res,
                                                    cutoffs, div_val=div_val)

            if tie_weight:
                for i in range(len(self.crit.out_layers)):
                    self.crit.out_layers[i].weight = self.word_emb.emb_layers[i].weight

            if tie_projs:
                for i, tie_proj in enumerate(tie_projs):
                    if tie_proj and div_val == 1 and self.d_res != d_embed:
                        self.crit.out_projs[i] = self.word_emb.emb_projs[0]
                    elif tie_proj and div_val != 1:
                        self.crit.out_projs[i] = self.word_emb.emb_projs[i]

        self.same_length = same_length
        self.clamp_len = clamp_len

        self._create_params()

    def backward_compatible(self):
        self.sample_softmax = -1

    def _create_params(self):
        if self.attn_type == 0:  # default attention
            self.pos_emb = PositionalEmbedding(self.d_model)
            self.r_w_bias = nn.Parameter(
                torch.Tensor(self.n_head, self.d_head))
            self.r_r_bias = nn.Parameter(
                torch.Tensor(self.n_head, self.d_head))

        elif self.attn_type == 1:  # learnable
            self.r_emb = nn.Parameter(torch.Tensor(
                    self.n_layer, self.max_klen, self.n_head, self.d_head))
            self.r_w_bias = nn.Parameter(torch.Tensor(
                    self.n_layer, self.n_head, self.d_head))
            self.r_bias = nn.Parameter(torch.Tensor(
                    self.n_layer, self.max_klen, self.n_head))

        elif self.attn_type in [2, 4, 5, 6, 7,
                                10, 14, 16,
                                24, 25, 26,
                                34, 35,
                                44, 45, 46,
                                54,
                                64,
                                74,
                                80, 90,
                                104, 114, 124, 134, 144,
                                204, 224, 234, 244, 254,
                                324, 924, 934]:
            # standard absolute pos
            self.pos_emb = PositionalEmbedding(self.d_res)

        elif self.attn_type == 3:  # absolute deeper SA
            self.r_emb = nn.Parameter(torch.Tensor(
                    self.n_layer, self.max_klen, self.n_head, self.d_head))

    def reset_length(self, tgt_len, ext_len, mem_len):
        self.tgt_len = tgt_len
        self.mem_len = mem_len
        self.ext_len = ext_len

    def init_mems(self):
        if self.mem_len > 0:
            mems = []
            param = next(self.parameters())
            for i in range(self.n_layer+1):
                empty = torch.empty(0, dtype=param.dtype, device=param.device)
                mems.append(empty)

            return mems
        else:
            return None

    def _update_mems(self, hids, mems, qlen, mlen):
        # does not deal with None
        if mems is None:
            return None

        # mems is not None
        assert len(hids) == len(mems), 'len(hids) != len(mems)'

        # There are `mlen + qlen` steps that can be cached into mems
        # For the next step, the last `ext_len` of the `qlen` tokens
        # will be used as the extended context. Hence, we only cache
        # the tokens from `mlen + qlen - self.ext_len - self.mem_len`
        # to `mlen + qlen - self.ext_len`.
        with torch.no_grad():
            new_mems = []
            end_idx = mlen + max(0, qlen - 0 - self.ext_len)
            beg_idx = max(0, end_idx - self.mem_len)
            for i in range(len(hids)):

                cat = torch.cat([mems[i], hids[i]], dim=0)
                new_mems.append(cat[beg_idx:end_idx].detach())

        return new_mems

    def _forward(self, dec_inp, mems=None, carry_over_fast_weight=False):
        qlen, bsz = dec_inp.size()

        word_emb = self.word_emb(dec_inp)
        if carry_over_fast_weight:
            mlen = 0
        else:
            mlen = mems[0].size(0) if mems is not None else 0
        klen = mlen + qlen

        if self.same_length:
            all_ones = word_emb.new_ones(qlen, klen)
            mask_len = klen - self.mem_len
            if mask_len > 0:
                mask_shift_len = qlen - mask_len
            else:
                mask_shift_len = qlen
            dec_attn_mask = (torch.triu(all_ones, 1+mlen)
                             + torch.tril(all_ones, -mask_shift_len)
                             ).bool()[:, :, None]
        else:
            dec_attn_mask = torch.triu(
                word_emb.new_ones(qlen, klen),
                diagonal=1+mlen).bool()[:, :, None]

        hids = []
        if self.attn_type == 0:  # default
            pos_seq = torch.arange(
                klen-1, -1, -1.0, device=word_emb.device, dtype=word_emb.dtype)

            if self.clamp_len > 0:
                pos_seq.clamp_(max=self.clamp_len)

            pos_emb = self.pos_emb(pos_seq)

            pos_emb = self.drop(pos_emb)
            core_out = self.drop(word_emb)

            hids.append(core_out)
            for i, layer in enumerate(self.layers):
                mems_i = None if mems is None else mems[i]
                core_out = layer(
                    core_out, pos_emb, self.r_w_bias, self.r_r_bias,
                    dec_attn_mask=dec_attn_mask, mems=mems_i)
                hids.append(core_out)

        elif self.attn_type == 1:  # learnable
            core_out = self.drop(word_emb)
            hids.append(core_out)
            for i, layer in enumerate(self.layers):
                if self.clamp_len > 0:
                    r_emb = self.r_emb[i][-self.clamp_len:]
                    r_bias = self.r_bias[i][-self.clamp_len:]
                else:
                    r_emb, r_bias = self.r_emb[i], self.r_bias[i]

                mems_i = None if mems is None else mems[i]
                core_out = layer(
                    core_out, r_emb, self.r_w_bias[i], r_bias,
                    dec_attn_mask=dec_attn_mask, mems=mems_i)
                hids.append(core_out)

        elif self.attn_type in [2, 4, 5, 6, 7, 10, 14, 16]:  # absolute
            if self.no_pos:
                core_out = self.drop(word_emb)
            else:
                pos_seq = torch.arange(klen-1, -1, -1.0,
                                       device=word_emb.device,
                                       dtype=word_emb.dtype)
                # pos_seq = torch.arange(0, klen, device=word_emb.device,
                #                        dtype=word_emb.dtype)
                if self.clamp_len > 0:
                    pos_seq.clamp_(max=self.clamp_len)
                pos_emb = self.pos_emb(pos_seq)
                core_out = self.drop(word_emb + pos_emb[-qlen:])

            hids.append(core_out)
            for i, layer in enumerate(self.layers):
                mems_i = None if mems is None else mems[i]
                if mems_i is not None and i == 0:
                    mems_i += pos_emb[:mlen]
                if self.attn_type == 5:
                    core_out = layer(core_out, dec_attn_mask=dec_attn_mask,
                                     mems=mems_i, redraw=self.training)
                else:
                    core_out = layer(core_out, dec_attn_mask=dec_attn_mask,
                                     mems=mems_i)
                hids.append(core_out)

        elif self.attn_type in [24, 25, 26, 34, 35, 44, 45, 46, 54, 64,
                                74, 80, 90, 104, 114, 124, 134, 144,
                                204, 224, 234, 244, 254, 324,
                                924, 934]:  # absolute
            if self.no_pos:
                core_out = self.drop(word_emb)
            else:
                pos_seq = torch.arange(klen-1, -1, -1.0,
                                       device=word_emb.device,
                                       dtype=word_emb.dtype)
                # pos_seq = torch.arange(0, klen, device=word_emb.device,
                #                        dtype=word_emb.dtype)
                if self.clamp_len > 0:
                    pos_seq.clamp_(max=self.clamp_len)
                pos_emb = self.pos_emb(pos_seq)
                core_out = self.drop(word_emb + pos_emb[-qlen:])

            hids.append(core_out)
            if carry_over_fast_weight:
                new_mems = []
            for i, layer in enumerate(self.layers):
                mems_i = None if mems is None else mems[i]
                if self.attn_type in [25, 35, 45]:
                    out = layer(
                        core_out, dec_attn_mask=dec_attn_mask,
                        mems=mems_i, redraw=self.training,
                        carry_over_fast_weight=carry_over_fast_weight)
                else:
                    out = layer(
                        core_out, dec_attn_mask=dec_attn_mask, mems=mems_i,
                        carry_over_fast_weight=carry_over_fast_weight)
                if carry_over_fast_weight:
                    core_out, new_fast_weight = out
                    new_mems.append(new_fast_weight)
                else:
                    core_out = out
                hids.append(core_out)

        elif self.attn_type == 3:
            core_out = self.drop(word_emb)

            hids.append(core_out)
            for i, layer in enumerate(self.layers):
                mems_i = None if mems is None else mems[i]
                if mems_i is not None and mlen > 0:
                    cur_emb = self.r_emb[i][:-qlen]
                    cur_size = cur_emb.size(0)
                    if cur_size < mlen:
                        cur_emb_pad = cur_emb[0:1].expand(
                            mlen-cur_size, -1, -1)
                        cur_emb = torch.cat([cur_emb_pad, cur_emb], 0)
                    else:
                        cur_emb = cur_emb[-mlen:]
                    mems_i += cur_emb.view(mlen, 1, -1)
                core_out += self.r_emb[i][-qlen:].view(qlen, 1, -1)

                core_out = layer(
                    core_out, dec_attn_mask=dec_attn_mask, mems=mems_i)
                hids.append(core_out)

        core_out = self.drop(core_out)
        if not carry_over_fast_weight:
            new_mems = self._update_mems(hids, mems, mlen, qlen)

        return core_out, new_mems

    def forward(self, data, target, *mems, softmax_keep_order=False,
                carry_over_fast_weight=False):
        # nn.DataParallel does not allow size(0) tensors to be broadcasted.
        # So, have to initialize size(0) mems inside the model forward.
        # Moreover, have to return new_mems to allow nn.DataParallel to piece
        # them together.
        if not mems:
            mems = self.init_mems()

        tgt_len = target.size(0)
        hidden, new_mems = self._forward(
            data, mems=mems, carry_over_fast_weight=carry_over_fast_weight)

        pred_hid = hidden[-tgt_len:]
        if self.sample_softmax > 0 and self.training:
            assert self.tie_weight
            logit = sample_logits(self.word_emb,
                self.out_layer.bias, target, pred_hid, self.sampler)
            loss = -F.log_softmax(logit, -1)[:, :, 0]
        else:
            loss = self.crit(pred_hid.reshape(-1, pred_hid.size(-1)),
                             target.reshape(-1),
                             keep_order=softmax_keep_order)
            loss = loss.view(tgt_len, -1)

        if new_mems is None:
            return [loss]
        else:
            return [loss] + new_mems


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='unit test')

    parser.add_argument('--n_layer', type=int, default=4, help='')
    parser.add_argument('--n_rel_layer', type=int, default=4, help='')
    parser.add_argument('--n_head', type=int, default=2, help='')
    parser.add_argument('--d_head', type=int, default=2, help='')
    parser.add_argument('--d_model', type=int, default=200, help='')
    parser.add_argument('--d_embed', type=int, default=200, help='')
    parser.add_argument('--d_inner', type=int, default=200, help='')
    parser.add_argument('--dropout', type=float, default=0.0, help='')
    parser.add_argument('--cuda', action='store_true', help='')
    parser.add_argument('--seed', type=int, default=1111, help='')
    parser.add_argument('--multi_gpu', action='store_true', help='')

    args = parser.parse_args()

    device = torch.device("cuda" if args.cuda else "cpu")

    B = 4
    tgt_len, mem_len, ext_len = 36, 36, 0
    data_len = tgt_len * 20
    args.n_token = 10000

    import data_utils

    data = torch.LongTensor(data_len*B).random_(0, args.n_token).to(device)
    diter = data_utils.LMOrderedIterator(
        data, B, tgt_len, device=device, ext_len=ext_len)

    cutoffs = [args.n_token // 2]
    tie_projs = [False] + [True] * len(cutoffs)

    for div_val in [1, 2]:
        for d_embed in [200, 100]:
            model = MemTransformerLM(
                args.n_token, args.n_layer, args.n_head, args.d_model,
                args.d_head, args.d_inner, args.dropout, dropatt=args.dropout,
                tie_weight=True, d_embed=d_embed, div_val=div_val,
                tie_projs=tie_projs, pre_lnorm=True, tgt_len=tgt_len,
                ext_len=ext_len, mem_len=mem_len, cutoffs=cutoffs,
                attn_type=0).to(device)

            print(sum(p.numel() for p in model.parameters()))

            mems = tuple()
            for idx, (inp, tgt, seqlen) in enumerate(diter):
                print('batch {}'.format(idx))
                out = model(inp, tgt, *mems)
                mems = out[1:]
