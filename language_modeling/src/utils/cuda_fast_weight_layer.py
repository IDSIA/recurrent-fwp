# Fast weight layers using custom kernels.
# Many code duplications to be refactored!
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.fast_fast_weight import fast_weight_delta
from utils.fast_transformers import fast_weight_sum

# Delta RNN variants
from utils.fast_weight_rnn import fast_rnn
from utils.fast_weight_rnn_v2 import fast_rnn_v2

# Delta LSTM variants
from utils.fast_lstm import fast_lstm
from utils.fast_lstm_v2 import fast_lstm_v2
from utils.fast_lstm_v3 import fast_lstm_v3
from utils.fast_lstm_v4 import fast_lstm_v4

# Recurrent Delta Net
from utils.rec_update_fwm_tanh import rec_update_fwm_tanh

from utils.performer_helper import prime, draw_orthogonal_random_matrix


@torch.jit.script
def elu_p1(x):
    return F.elu(x, 1., False) + 1.


@torch.jit.script
def sum_norm(x):
    return x / x.sum(-1, keepdim=True)


@torch.jit.script
def sum_norm_eps(x):
    return x / (x.sum(-1, keepdim=True) + 1e-5)


@torch.jit.script
def elu_p1_sum_norm(x):
    y = F.elu(x, 1., False) + 1.
    return y / y.sum(-1, keepdim=True)


@torch.jit.script
def elu_p1_sum_norm_eps(x):
    y = F.elu(x, 1., False) + 1.
    return y / (y.sum(-1, keepdim=True) + 1e-5)


# Linear Transformer version
# our update rule + Katharopoulos et al's ELU based attention
class CudaFastWeightLinearTransformerLayer(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout, dropatt=0,
                 pre_lnorm=False, eps=1e-5, layer_id=None, num_layer=None,
                 skip_attn_normalization=True):
        # skip_attn_normalization is now set to True by default, thus it can
        # be removed.
        # Originally, with skip_attn_normalization set to False,
        # we had a version of the model which applies attention normalization
        # to the output (but not when we retrieve with the key for removal).
        super(CudaFastWeightLinearTransformerLayer, self).__init__()
        print(f"Using CudaFastWeightLinearTransformerLayer {layer_id} -")

        assert layer_id is not None
        assert num_layer is not None
        self.layer_id = layer_id
        self.num_layer = num_layer

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout

        # (3 * d_head * n_head) for qkv and (1 * n_head) for beta.
        self.qkvb_net = nn.Linear(
            d_model, n_head * (3 * d_head + 1), bias=False)

        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)

        self.layer_norm = nn.LayerNorm(d_model)

        self.scale = 1 / (d_head ** 0.5)

        self.pre_lnorm = pre_lnorm
        self.normalize_attn_scores = (not skip_attn_normalization)
        self.eps = eps

    def forward(self, h, attn_mask=None, mems=None,
                carry_over_fast_weight=False):
        # multihead attention
        # shape h: (len, B, n_head * d_head)

        if self.pre_lnorm:
            # layer normalization
            h = self.layer_norm(h)

        slen, bsz, _ = h.size()

        qkvb = self.qkvb_net(h)
        qkvb = qkvb.view(slen, bsz, self.n_head, 3 * self.d_head + 1)
        head_q, head_k, head_v, head_beta = torch.split(
            qkvb, (self.d_head,) * 3 + (1,), -1)
        head_beta = torch.sigmoid(head_beta)

        # reshape to (B, heads, len, dim)
        head_q = head_q.permute(1, 2, 0, 3)
        head_k = head_k.permute(1, 2, 0, 3)
        head_v = head_v.permute(1, 2, 0, 3)
        head_beta = head_beta.permute(1, 2, 0, 3)

        # TODO add dropout here?
        # transform q and k
        head_q = F.elu(head_q, 1., False) + 1.
        head_k = F.elu(head_k, 1., False) + 1.

        # normalize k and q, crucial for stable training.
        head_k = head_k / head_k.sum(-1, keepdim=True)
        head_q = head_q / head_q.sum(-1, keepdim=True)

        if self.normalize_attn_scores:
            # another version would be:
            # head_k_beta = head_k * head_beta
            # denominator_acc = torch.cumsum(head_k_beta, dim=2)
            denominator_acc = torch.cumsum(head_k, dim=2)

        if mems is None:
            mem_fast_weights = torch.zeros(
                bsz, self.n_head, self.d_head, self.d_head,
                device=head_k.device)
        else:
            assert carry_over_fast_weight
            mem_fast_weights, fast_denom = mems
            # bsz can be smaller for the last batch
            mem_fast_weights = mem_fast_weights[:bsz]
            if self.normalize_attn_scores:
                denominator_acc = denominator_acc + fast_denom[:bsz]

        if self.normalize_attn_scores:
            denominator = torch.einsum(
                'lbij,lbij->lbi', denominator_acc, head_q).unsqueeze(-1)

        layer_out = fast_weight_delta(
            head_q, head_k, head_v, head_beta, mem_fast_weights)

        # shape (B, n_head, len, d_head)
        if self.normalize_attn_scores:
            layer_out = self.scale * layer_out / (denominator + self.eps)
        else:
            layer_out = self.scale * layer_out

        layer_out = layer_out.transpose(1, 2)

        layer_out = layer_out.reshape(
            bsz, slen, self.n_head * self.d_head)

        layer_out = layer_out.transpose(0, 1)

        # expect [qlen, B, n_head * d_head]

        # linear projection
        attn_out = self.o_net(layer_out)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            # residual connection
            output = h + attn_out
        else:
            # residual connection + layer normalization
            output = self.layer_norm(h + attn_out)

        if carry_over_fast_weight:
            # last values of accumulator should be carried over.
            # clone is needed as backward modifies the data of fast weight
            if self.normalize_attn_scores:
                new_k_acc = denominator_acc[:, :, -1, :].unsqueeze(2).detach()
            else:
                new_k_acc = None
            new_mem = (mem_fast_weights.clone().detach(), new_k_acc)
            return output, new_mem

        return output


# Linear Transformer version
# our update rule + Katharopoulos et al's ELU based attention
# with attention normalization
class CudaNormFastWeightLinearTransformerLayer(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout, dropatt=0,
                 pre_lnorm=False, eps=1e-5, layer_id=None, num_layer=None,
                 skip_attn_normalization=False):
        super(CudaNormFastWeightLinearTransformerLayer, self).__init__()
        print(f"Using CudaNormFastWeightLinearTransformerLayer {layer_id} -")

        assert layer_id is not None
        assert num_layer is not None
        self.layer_id = layer_id
        self.num_layer = num_layer

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout

        # (3 * d_head * n_head) for qkv and (1 * n_head) for beta.
        self.qkvb_net = nn.Linear(
            d_model, n_head * (3 * d_head + 1), bias=False)

        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)

        self.layer_norm = nn.LayerNorm(d_model)

        self.scale = 1 / (d_head ** 0.5)

        self.pre_lnorm = pre_lnorm
        self.normalize_attn_scores = (not skip_attn_normalization)
        self.eps = eps

    def forward(self, h, attn_mask=None, mems=None,
                carry_over_fast_weight=False):
        # multihead attention
        # shape h: (len, B, n_head * d_head)

        if self.pre_lnorm:
            # layer normalization
            h = self.layer_norm(h)

        slen, bsz, _ = h.size()

        qkvb = self.qkvb_net(h)
        qkvb = qkvb.view(slen, bsz, self.n_head, 3 * self.d_head + 1)
        head_q, head_k, head_v, head_beta = torch.split(
            qkvb, (self.d_head,) * 3 + (1,), -1)
        head_beta = torch.sigmoid(head_beta)

        # reshape to (B, heads, len, dim)
        head_q = head_q.permute(1, 2, 0, 3)
        head_k = head_k.permute(1, 2, 0, 3)
        head_v = head_v.permute(1, 2, 0, 3)
        head_beta = head_beta.permute(1, 2, 0, 3)

        # TODO add dropout here?
        # transform q and k
        head_q = F.elu(head_q, 1., False) + 1.
        head_k = F.elu(head_k, 1., False) + 1.

        # normalize k and q, crucial for stable training.
        head_k = head_k / head_k.sum(-1, keepdim=True)
        head_q = head_q / head_q.sum(-1, keepdim=True)

        if self.normalize_attn_scores:
            # another version would be:
            # head_k_beta = head_k * head_beta
            # denominator_acc = torch.cumsum(head_k_beta, dim=2)
            denominator_acc = torch.cumsum(head_k, dim=2)

        if mems is None:
            mem_fast_weights = torch.zeros(
                bsz, self.n_head, self.d_head, self.d_head,
                device=head_k.device)
            if self.normalize_attn_scores:
                # key_denom = z(i-1) * key(i) and 1 if i=1
                # z(i) = denominator_acc
                key_denom = torch.cat(
                    [torch.zeros([bsz, self.n_head, 1, self.d_head],
                                 device=head_q.device),
                     denominator_acc[:, :, :-1, :].clone()], dim=2)
                key_denom = torch.einsum('lbij,lbij->lbi', key_denom, head_k)
                key_denom = torch.cat(
                    [torch.ones([bsz, self.n_head, 1], device=head_q.device),
                     key_denom[:, :, 1:].clone()], dim=2).unsqueeze(-1)
                head_beta = head_beta * key_denom 
                head_k = head_k / (key_denom + self.eps)
        else:
            assert carry_over_fast_weight
            mem_fast_weights, fast_denom = mems
            # bsz can be smaller for the last batch
            mem_fast_weights = mem_fast_weights[:bsz]
            if self.normalize_attn_scores:
                key_denom = torch.cat(
                    [torch.zeros([bsz, self.n_head, 1, self.d_head],
                                 device=head_q.device),
                     denominator_acc[:, :, :-1, :].clone()], dim=2)
                key_denom = key_denom + fast_denom[:bsz]
                denominator_acc = denominator_acc + fast_denom[:bsz]

                key_denom = torch.einsum(
                    'lbij,lbij->lbi', key_denom, head_k).unsqueeze(-1)
                head_beta = head_beta * key_denom
                head_k = head_k / (key_denom + self.eps)

        if self.normalize_attn_scores:
            denominator = torch.einsum(
                'lbij,lbij->lbi', denominator_acc, head_q).unsqueeze(-1)

        layer_out = fast_weight_delta(
            head_q, head_k, head_v, head_beta, mem_fast_weights)

        # shape (B, n_head, len, d_head)
        if self.normalize_attn_scores:
            layer_out = self.scale * layer_out / (denominator + self.eps)
        else:
            layer_out = self.scale * layer_out

        layer_out = layer_out.transpose(1, 2)

        layer_out = layer_out.reshape(
            bsz, slen, self.n_head * self.d_head)

        layer_out = layer_out.transpose(0, 1)

        # expect [qlen, B, n_head * d_head]

        # linear projection
        attn_out = self.o_net(layer_out)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            # residual connection
            output = h + attn_out
        else:
            # residual connection + layer normalization
            output = self.layer_norm(h + attn_out)

        if carry_over_fast_weight:
            # last values of accumulator should be carried over.
            # clone is needed as backward modifies the data of fast weight
            if self.normalize_attn_scores:
                new_k_acc = denominator_acc[:, :, -1, :].unsqueeze(2).detach()
            else:
                new_k_acc = None
            new_mem = (mem_fast_weights.clone().detach(), new_k_acc)
            return output, new_mem

        return output


# Performer version, our update rule + FAVOR+
class CudaFastWeightPerformerLayer(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout, dropatt=0,
                 pre_lnorm=False, eps=1e-5, skip_attn_normalization=True,
                 proj_dim=256, device='cuda'):
        super(CudaFastWeightPerformerLayer, self).__init__()
        print(f"Using CudaFastWeightPerformerLayer - "
              f"proj_dim: {proj_dim}")

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout

        # (3 * d_head * n_head) for qkv and (1 * n_head) for beta.
        self.qkvb_net = nn.Linear(
            d_model, n_head * (3 * d_head + 1), bias=False)

        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)

        self.layer_norm = nn.LayerNorm(d_model)

        self.scale = 1 / (d_head ** 0.5)

        self.pre_lnorm = pre_lnorm
        self.normalize_attn_scores = (not skip_attn_normalization)
        self.eps = eps

        self.proj_dim = proj_dim
        self.proj_matrix = draw_orthogonal_random_matrix(
            d_head, proj_dim, device=device)  # TODO store this as param?

    def forward(self, h, attn_mask=None, mems=None, redraw=True,
                carry_over_fast_weight=False):
        # multihead attention
        # shape h: (len, B, n_head * d_head)

        if self.pre_lnorm:
            # layer normalization
            h = self.layer_norm(h)

        slen, bsz, _ = h.size()

        qkvb = self.qkvb_net(h)
        qkvb = qkvb.view(slen, bsz, self.n_head, 3 * self.d_head + 1)
        head_q, head_k, head_v, head_beta = torch.split(
            qkvb, (self.d_head,) * 3 + (1,), -1)
        head_beta = torch.sigmoid(head_beta)

        # reshape to (B, heads, len, dim)
        head_q = head_q.permute(1, 2, 0, 3)
        head_k = head_k.permute(1, 2, 0, 3)
        head_v = head_v.permute(1, 2, 0, 3)
        head_beta = head_beta.permute(1, 2, 0, 3)

        if redraw:
            self.proj_matrix = draw_orthogonal_random_matrix(
                self.d_head, self.proj_dim, device=h.device)

        head_q = prime(head_q, self.proj_matrix)  # (B, n_head, len, proj_dim)
        head_k = prime(head_k, self.proj_matrix)

        # normalize k and q, crucial for stable training.
        head_k = head_k / head_k.sum(-1, keepdim=True)
        head_q = head_q / head_q.sum(-1, keepdim=True)

        if self.normalize_attn_scores:
            # another version would be:
            # head_k_beta = head_k * head_beta
            # denominator_acc = torch.cumsum(head_k_beta, dim=2)
            denominator_acc = torch.cumsum(head_k, dim=2)

        if mems is None:
            mem_fast_weights = torch.zeros(
                bsz, self.n_head, 2 * self.proj_dim, self.d_head,
                device=head_k.device)
        else:
            assert carry_over_fast_weight
            mem_fast_weights, fast_denom = mems
            # bsz can be smaller for the last batch
            mem_fast_weights = mem_fast_weights[:bsz]
            if self.normalize_attn_scores:
                denominator_acc = denominator_acc + fast_denom[:bsz]

        if self.normalize_attn_scores:
            denominator = torch.einsum(
                'lbij,lbij->lbi', denominator_acc, head_q).unsqueeze(-1)

        layer_out = fast_weight_delta(
            head_q, head_k, head_v, head_beta, mem_fast_weights)

        # shape (B, n_head, len, d_head)
        if self.normalize_attn_scores:
            layer_out = self.scale * layer_out / (denominator + self.eps)
        else:
            layer_out = self.scale * layer_out

        layer_out = layer_out.transpose(1, 2)

        layer_out = layer_out.reshape(
            bsz, slen, self.n_head * self.d_head)

        layer_out = layer_out.transpose(0, 1)

        # expect [qlen, B, n_head * d_head]

        # linear projection
        attn_out = self.o_net(layer_out)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            # residual connection
            output = h + attn_out
        else:
            # residual connection + layer normalization
            output = self.layer_norm(h + attn_out)

        if carry_over_fast_weight:
            # last values of accumulator should be carried over.
            # clone is needed as backward modifies the data of fast weight
            if self.normalize_attn_scores:
                new_k_acc = denominator_acc[:, :, -1, :].unsqueeze(2).detach()
            else:
                new_k_acc = None
            new_mem = (mem_fast_weights.clone().detach(), new_k_acc)
            return output, new_mem

        return output


# Performer version, our update rule + FAVOR+
# with attention normalization
class CudaNormFastWeightPerformerLayer(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout, dropatt=0,
                 pre_lnorm=False, eps=1e-5, skip_attn_normalization=False,
                 proj_dim=256, device='cuda'):
        super(CudaNormFastWeightPerformerLayer, self).__init__()
        print(f"Using CudaNormFastWeightPerformerLayer - "
              f"proj_dim: {proj_dim}")

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout

        # (3 * d_head * n_head) for qkv and (1 * n_head) for beta.
        self.qkvb_net = nn.Linear(
            d_model, n_head * (3 * d_head + 1), bias=False)

        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)

        self.layer_norm = nn.LayerNorm(d_model)

        self.scale = 1 / (d_head ** 0.5)

        self.pre_lnorm = pre_lnorm
        self.normalize_attn_scores = (not skip_attn_normalization)
        self.eps = eps

        self.proj_dim = proj_dim
        self.proj_matrix = draw_orthogonal_random_matrix(
            d_head, proj_dim, device=device)  # TODO store this as param?

    def forward(self, h, attn_mask=None, mems=None, redraw=True,
                carry_over_fast_weight=False):
        # multihead attention
        # shape h: (len, B, n_head * d_head)

        if self.pre_lnorm:
            # layer normalization
            h = self.layer_norm(h)

        slen, bsz, _ = h.size()

        qkvb = self.qkvb_net(h)
        qkvb = qkvb.view(slen, bsz, self.n_head, 3 * self.d_head + 1)
        head_q, head_k, head_v, head_beta = torch.split(
            qkvb, (self.d_head,) * 3 + (1,), -1)
        head_beta = torch.sigmoid(head_beta)

        # reshape to (B, heads, len, dim)
        head_q = head_q.permute(1, 2, 0, 3)
        head_k = head_k.permute(1, 2, 0, 3)
        head_v = head_v.permute(1, 2, 0, 3)
        head_beta = head_beta.permute(1, 2, 0, 3)

        if redraw:
            self.proj_matrix = draw_orthogonal_random_matrix(
                self.d_head, self.proj_dim, device=h.device)

        head_q = prime(head_q, self.proj_matrix)  # (B, n_head, len, proj_dim)
        head_k = prime(head_k, self.proj_matrix)

        # normalize k and q, crucial for stable training.
        head_k = head_k / head_k.sum(-1, keepdim=True)
        head_q = head_q / head_q.sum(-1, keepdim=True)

        if self.normalize_attn_scores:
            # another version would be:
            # head_k_beta = head_k * head_beta
            # denominator_acc = torch.cumsum(head_k_beta, dim=2)
            denominator_acc = torch.cumsum(head_k, dim=2)

        if mems is None:
            mem_fast_weights = torch.zeros(
                bsz, self.n_head, 2 * self.proj_dim, self.d_head,
                device=head_k.device)
            if self.normalize_attn_scores:
                # key_denom = z(i-1) * key(i) and 1 if i=1
                # z(i) = denominator_acc
                key_denom = torch.cat(
                    [torch.zeros([bsz, self.n_head, 1, self.proj_dim * 2],
                                 device=head_q.device),
                     denominator_acc[:, :, :-1, :].clone()], dim=2)
                key_denom = torch.einsum('lbij,lbij->lbi', key_denom, head_k)
                key_denom = torch.cat(
                    [torch.ones([bsz, self.n_head, 1], device=head_q.device),
                     key_denom[:, :, 1:].clone()], dim=2).unsqueeze(-1)
                head_beta = head_beta * key_denom
                head_k = head_k / (key_denom + self.eps)

        else:
            assert carry_over_fast_weight
            mem_fast_weights, fast_denom = mems
            # bsz can be smaller for the last batch
            mem_fast_weights = mem_fast_weights[:bsz]
            if self.normalize_attn_scores:
                key_denom = torch.cat(
                    [torch.zeros([bsz, self.n_head, 1, self.proj_dim * 2],
                                 device=head_q.device),
                     denominator_acc[:, :, :-1, :].clone()], dim=2)
                key_denom = key_denom + fast_denom[:bsz]
                denominator_acc = denominator_acc + fast_denom[:bsz]

                key_denom = torch.einsum(
                    'lbij,lbij->lbi', key_denom, head_k).unsqueeze(-1)
                head_beta = head_beta * key_denom
                head_k = head_k / (key_denom + self.eps)

        if self.normalize_attn_scores:
            denominator = torch.einsum(
                'lbij,lbij->lbi', denominator_acc, head_q).unsqueeze(-1)

        layer_out = fast_weight_delta(
            head_q, head_k, head_v, head_beta, mem_fast_weights)

        # shape (B, n_head, len, d_head)
        if self.normalize_attn_scores:
            layer_out = self.scale * layer_out / (denominator + self.eps)
        else:
            layer_out = self.scale * layer_out

        layer_out = layer_out.transpose(1, 2)

        layer_out = layer_out.reshape(
            bsz, slen, self.n_head * self.d_head)

        layer_out = layer_out.transpose(0, 1)

        # expect [qlen, B, n_head * d_head]

        # linear projection
        attn_out = self.o_net(layer_out)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            # residual connection
            output = h + attn_out
        else:
            # residual connection + layer normalization
            output = self.layer_norm(h + attn_out)

        if carry_over_fast_weight:
            # last values of accumulator should be carried over.
            # clone is needed as backward modifies the data of fast weight
            if self.normalize_attn_scores:
                new_k_acc = denominator_acc[:, :, -1, :].unsqueeze(2).detach()
            else:
                new_k_acc = None
            new_mem = (mem_fast_weights.clone().detach(), new_k_acc)
            return output, new_mem

        return output


# Katharopoulos et al's Linear Transformer https://arxiv.org/abs/2006.16236
# = Sum update rule + ELU based attention function
class CudaFastWeightSumLinearTransformerLayer(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout, dropatt=0,
                 pre_lnorm=False, eps=1e-5, layer_id=None, num_layer=None,
                 skip_attn_normalization=False, use_sum_norm=False):
        super(CudaFastWeightSumLinearTransformerLayer, self).__init__()
        print(f"Using CudaFastWeightSumLinearTransformerLayer {layer_id} -")

        assert layer_id is not None
        assert num_layer is not None
        self.layer_id = layer_id
        self.num_layer = num_layer

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout

        # (3 * d_head * n_head) for qkv and (1 * n_head) for beta.
        self.qkv_net = nn.Linear(
            d_model, n_head * 3 * d_head, bias=False)

        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)

        self.layer_norm = nn.LayerNorm(d_model)

        self.scale = 1 / (d_head ** 0.5)

        self.pre_lnorm = pre_lnorm
        self.normalize_attn_scores = (not skip_attn_normalization)
        self.use_sum_norm = use_sum_norm
        self.eps = eps

    def forward(self, h, attn_mask=None, mems=None,
                carry_over_fast_weight=False):
        # multihead attention
        # shape h: (len, B, n_head * d_head)

        if self.pre_lnorm:
            # layer normalization
            h = self.layer_norm(h)

        slen, bsz, _ = h.size()

        qkv = self.qkv_net(h)
        qkv = qkv.view(slen, bsz, self.n_head, 3 * self.d_head)
        head_q, head_k, head_v = torch.split(
            qkv, (self.d_head,) * 3, -1)

        # reshape to (B, heads, len, dim)
        head_q = head_q.permute(1, 2, 0, 3)
        head_k = head_k.permute(1, 2, 0, 3)
        head_v = head_v.permute(1, 2, 0, 3)

        # TODO add dropout here?
        # transform q and k
        head_q = F.elu(head_q, 1., False) + 1.
        head_k = F.elu(head_k, 1., False) + 1.

        # normalize k and q, crucial for stable training.
        if self.use_sum_norm:
            head_k = head_k / head_k.sum(-1, keepdim=True)
            head_q = head_q / head_q.sum(-1, keepdim=True)

        if self.normalize_attn_scores:
            denominator_acc = torch.cumsum(head_k, dim=2)

        if mems is None:
            mem_fast_weights = torch.zeros(
                bsz, self.n_head, self.d_head, self.d_head,
                device=head_k.device)
        else:
            assert carry_over_fast_weight
            mem_fast_weights, fast_denom = mems
            # bsz can be smaller for the last batch
            mem_fast_weights = mem_fast_weights[:bsz]
            if self.normalize_attn_scores:
                denominator_acc = denominator_acc + fast_denom[:bsz]

        if self.normalize_attn_scores:
            denominator = torch.einsum(
                'lbij,lbij->lbi', denominator_acc, head_q).unsqueeze(-1)

        layer_out = fast_weight_sum(
            head_q, head_k, head_v, mem_fast_weights)

        # shape (B, n_head, len, d_head)
        if self.normalize_attn_scores:
            layer_out = self.scale * layer_out / (denominator + self.eps)
        else:
            layer_out = self.scale * layer_out

        layer_out = layer_out.transpose(1, 2)

        layer_out = layer_out.reshape(
            bsz, slen, self.n_head * self.d_head)

        layer_out = layer_out.transpose(0, 1)

        # expect [qlen, B, n_head * d_head]

        # linear projection
        attn_out = self.o_net(layer_out)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            # residual connection
            output = h + attn_out
        else:
            # residual connection + layer normalization
            output = self.layer_norm(h + attn_out)

        if carry_over_fast_weight:
            # last values of accumulator should be carried over.
            # clone is needed as backward modifies the data of fast weight
            if self.normalize_attn_scores:
                new_k_acc = denominator_acc[:, :, -1, :].unsqueeze(2).detach()
            else:
                new_k_acc = None
            new_mem = (mem_fast_weights.clone().detach(), new_k_acc)
            return output, new_mem

        return output


# Performer https://arxiv.org/abs/2009.14794
# = Sum update rule + FAVOR+
class CudaFastWeightSumPerformerLayer(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout, dropatt=0,
                 pre_lnorm=False, eps=1e-5, skip_attn_normalization=False,
                 proj_dim=256, device='cuda', use_sum_norm=False):
        super(CudaFastWeightSumPerformerLayer, self).__init__()
        print(f"Using CudaFastWeightSumPerformerLayer - "
              f"proj_dim: {proj_dim}")

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout

        # (3 * d_head * n_head) for qkv and (1 * n_head) for beta.
        self.qkv_net = nn.Linear(
            d_model, n_head * 3 * d_head, bias=False)

        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)

        self.layer_norm = nn.LayerNorm(d_model)

        self.scale = 1 / (d_head ** 0.5)

        self.pre_lnorm = pre_lnorm
        self.normalize_attn_scores = (not skip_attn_normalization)
        self.use_sum_norm = use_sum_norm
        self.eps = eps
        print(f"normalize_attn_scores: {self.normalize_attn_scores}")
        print(f"use_sum_norm: {self.use_sum_norm}")

        self.proj_dim = proj_dim
        self.proj_matrix = draw_orthogonal_random_matrix(
            d_head, proj_dim, device=device)  # TODO store this as param?

    def forward(self, h, attn_mask=None, mems=None, redraw=True,
                carry_over_fast_weight=False):
        # shape h: (len, B, n_head * d_head)

        if self.pre_lnorm:
            # layer normalization
            h = self.layer_norm(h)

        slen, bsz, _ = h.size()

        qkv = self.qkv_net(h)
        qkv = qkv.view(slen, bsz, self.n_head, 3 * self.d_head)
        head_q, head_k, head_v = torch.split(
            qkv, (self.d_head,) * 3, -1)

        # reshape to (B, heads, len, dim)
        head_q = head_q.permute(1, 2, 0, 3)
        head_k = head_k.permute(1, 2, 0, 3)
        head_v = head_v.permute(1, 2, 0, 3)

        if redraw:
            self.proj_matrix = draw_orthogonal_random_matrix(
                self.d_head, self.proj_dim, device=h.device)

        head_q = prime(head_q, self.proj_matrix)  # (B, n_head, len, proj_dim)
        head_k = prime(head_k, self.proj_matrix)

        if self.use_sum_norm:
            head_k = head_k / head_k.sum(-1, keepdim=True)
            head_q = head_q / head_q.sum(-1, keepdim=True)

        if self.normalize_attn_scores:
            denominator_acc = torch.cumsum(head_k, dim=2)

        if mems is None:
            mem_fast_weights = torch.zeros(
                bsz, self.n_head, 2 * self.proj_dim, self.d_head,
                device=head_k.device)
        else:
            assert carry_over_fast_weight
            mem_fast_weights, fast_denom = mems
            # bsz can be smaller for the last batch
            mem_fast_weights = mem_fast_weights[:bsz]
            if self.normalize_attn_scores:
                denominator_acc = denominator_acc + fast_denom[:bsz]

        if self.normalize_attn_scores:
            denominator = torch.einsum(
                'lbij,lbij->lbi', denominator_acc, head_q).unsqueeze(-1)

        layer_out = fast_weight_sum(
            head_q, head_k, head_v, mem_fast_weights)

        # shape (B, n_head, len, d_head)
        if self.normalize_attn_scores:
            layer_out = self.scale * layer_out / (denominator + self.eps)
        else:
            layer_out = self.scale * layer_out

        layer_out = layer_out.transpose(1, 2)

        layer_out = layer_out.reshape(
            bsz, slen, self.n_head * self.d_head)

        layer_out = layer_out.transpose(0, 1)

        # expect [qlen, B, n_head * d_head]

        # linear projection
        attn_out = self.o_net(layer_out)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            # residual connection
            output = h + attn_out
        else:
            # residual connection + layer normalization
            output = self.layer_norm(h + attn_out)

        if carry_over_fast_weight:
            # last values of accumulator should be carried over.
            # clone is needed as backward modifies the data of fast weight
            if self.normalize_attn_scores:
                new_k_acc = denominator_acc[:, :, -1, :].unsqueeze(2).detach()
            else:
                new_k_acc = None
            new_mem = (mem_fast_weights.clone().detach(), new_k_acc)
            return output, new_mem

        return output


# Our update rule + DPFP
class CudaFastWeightDPFPTransformerLayer(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout, dropatt=0,
                 pre_lnorm=False, eps=1e-5, layer_id=None, num_layer=None,
                 skip_attn_normalization=False, n_roll=2):
        super(CudaFastWeightDPFPTransformerLayer, self).__init__()
        print(f"Using CudaFastWeightDPFPTransformerLayer roll {n_roll} -")

        assert layer_id is not None
        assert num_layer is not None
        self.layer_id = layer_id
        self.num_layer = num_layer

        self.n_roll = n_roll

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout

        # (3 * d_head * n_head) for qkv and (1 * n_head) for beta.
        self.qkvb_net = nn.Linear(
            d_model, n_head * (3 * d_head + 1), bias=False)

        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)

        self.layer_norm = nn.LayerNorm(d_model)

        self.scale = 1 / (d_head ** 0.5)

        self.pre_lnorm = pre_lnorm
        self.normalize_attn_scores = (not skip_attn_normalization)
        self.eps = eps

    def mul_roll_repeat(self, x):
        rolls = []
        for i in range(1, self.n_roll + 1):
            rolls.append(x * x.roll(shifts=i, dims=-1))
        return torch.cat(rolls, dim=-1)

    def forward(self, h, attn_mask=None, mems=None,
                carry_over_fast_weight=False):
        # multihead attention
        # shape h: (len, B, n_head * d_head)

        if self.pre_lnorm:
            # layer normalization
            h = self.layer_norm(h)

        slen, bsz, _ = h.size()

        qkvb = self.qkvb_net(h)
        qkvb = qkvb.view(slen, bsz, self.n_head, 3 * self.d_head + 1)
        head_q, head_k, head_v, head_beta = torch.split(
            qkvb, (self.d_head,) * 3 + (1,), -1)
        head_beta = torch.sigmoid(head_beta)

        # reshape to (B, heads, len, dim)
        head_q = head_q.permute(1, 2, 0, 3)
        head_k = head_k.permute(1, 2, 0, 3)
        head_v = head_v.permute(1, 2, 0, 3)
        head_beta = head_beta.permute(1, 2, 0, 3)

        act = lambda x: F.relu(x)  # relu or exp
        head_k = torch.cat([act(head_k), act(-head_k)], dim=-1)
        head_q = torch.cat([act(head_q), act(-head_q)], dim=-1)

        head_k = self.mul_roll_repeat(head_k)
        head_q = self.mul_roll_repeat(head_q)

        # normalize k and q, crucial for stable training.
        head_k = head_k / head_k.sum(-1, keepdim=True)
        head_q = head_q / head_q.sum(-1, keepdim=True)

        if self.normalize_attn_scores:
            denominator_acc = torch.cumsum(head_k, dim=2)

        if mems is None:
            mem_fast_weights = torch.zeros(
                bsz, self.n_head, 2 * self.n_roll * self.d_head, self.d_head,
                device=head_k.device)
        else:
            assert carry_over_fast_weight
            mem_fast_weights, fast_denom = mems
            # bsz can be smaller for the last batch
            mem_fast_weights = mem_fast_weights[:bsz]
            if self.normalize_attn_scores:
                denominator_acc = denominator_acc + fast_denom[:bsz]

        if self.normalize_attn_scores:
            denominator = torch.einsum(
                'lbij,lbij->lbi', denominator_acc, head_q).unsqueeze(-1)

        layer_out = fast_weight_delta(
            head_q, head_k, head_v, head_beta, mem_fast_weights)

        # shape (B, n_head, len, d_head)
        if self.normalize_attn_scores:
            layer_out = self.scale * layer_out / (denominator + self.eps)
        else:
            layer_out = self.scale * layer_out

        layer_out = layer_out.transpose(1, 2)

        layer_out = layer_out.reshape(
            bsz, slen, self.n_head * self.d_head)

        layer_out = layer_out.transpose(0, 1)

        # expect [qlen, B, n_head * d_head]

        # linear projection
        attn_out = self.o_net(layer_out)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            # residual connection
            output = h + attn_out
        else:
            # residual connection + layer normalization
            output = self.layer_norm(h + attn_out)

        if carry_over_fast_weight:
            # last values of accumulator should be carried over.
            # clone is needed as backward modifies the data of fast weight
            if self.normalize_attn_scores:
                new_k_acc = denominator_acc[:, :, -1, :].unsqueeze(2).detach()
            else:
                new_k_acc = None
            new_mem = (mem_fast_weights.clone().detach(), new_k_acc)
            return output, new_mem

        return output


# Our update rule + DPFP, with attention normalization
class CudaNormFastWeightDPFPTransformerLayer(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout, dropatt=0,
                 pre_lnorm=False, eps=1e-5, layer_id=None, num_layer=None,
                 skip_attn_normalization=False, n_roll=2):
        super(CudaNormFastWeightDPFPTransformerLayer, self).__init__()
        print(f"Using CudaNormFastWeightDPFPTransformerLayer roll {n_roll} -")

        assert layer_id is not None
        assert num_layer is not None
        self.layer_id = layer_id
        self.num_layer = num_layer

        self.n_roll = n_roll

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout

        # (3 * d_head * n_head) for qkv and (1 * n_head) for beta.
        self.qkvb_net = nn.Linear(
            d_model, n_head * (3 * d_head + 1), bias=False)

        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)

        self.layer_norm = nn.LayerNorm(d_model)

        self.scale = 1 / (d_head ** 0.5)

        self.pre_lnorm = pre_lnorm
        self.normalize_attn_scores = (not skip_attn_normalization)
        self.eps = eps

    def mul_roll_repeat(self, x):
        rolls = []
        for i in range(1, self.n_roll + 1):
            rolls.append(x * x.roll(shifts=i, dims=-1))
        return torch.cat(rolls, dim=-1)

    def forward(self, h, attn_mask=None, mems=None,
                carry_over_fast_weight=False):
        # multihead attention
        # shape h: (len, B, n_head * d_head)

        if self.pre_lnorm:
            # layer normalization
            h = self.layer_norm(h)

        slen, bsz, _ = h.size()

        qkvb = self.qkvb_net(h)
        qkvb = qkvb.view(slen, bsz, self.n_head, 3 * self.d_head + 1)
        head_q, head_k, head_v, head_beta = torch.split(
            qkvb, (self.d_head,) * 3 + (1,), -1)
        head_beta = torch.sigmoid(head_beta)

        # reshape to (B, heads, len, dim)
        head_q = head_q.permute(1, 2, 0, 3)
        head_k = head_k.permute(1, 2, 0, 3)
        head_v = head_v.permute(1, 2, 0, 3)
        head_beta = head_beta.permute(1, 2, 0, 3)

        act = lambda x: F.relu(x)  # relu or exp
        head_k = torch.cat([act(head_k), act(-head_k)], dim=-1)
        head_q = torch.cat([act(head_q), act(-head_q)], dim=-1)

        head_k = self.mul_roll_repeat(head_k)
        head_q = self.mul_roll_repeat(head_q)

        # normalize k and q, crucial for stable training.
        head_k = head_k / head_k.sum(-1, keepdim=True)
        head_q = head_q / head_q.sum(-1, keepdim=True)

        if self.normalize_attn_scores:
            # another version would be:
            # head_k_beta = head_k * head_beta
            # denominator_acc = torch.cumsum(head_k_beta, dim=2)
            denominator_acc = torch.cumsum(head_k, dim=2)

        if mems is None:
            mem_fast_weights = torch.zeros(
                bsz, self.n_head, 2 * self.n_roll * self.d_head, self.d_head,
                device=head_k.device)
            if self.normalize_attn_scores:
                # key_denom = z(i-1) * key(i) and 1 if i=1
                # z(i) = denominator_acc
                key_denom = torch.cat(
                    [torch.zeros(
                        [bsz, self.n_head, 1, 2 * self.n_roll * self.d_head],
                        device=head_q.device),
                     denominator_acc[:, :, :-1, :].clone()], dim=2)
                key_denom = torch.einsum('lbij,lbij->lbi', key_denom, head_k)
                key_denom = torch.cat(
                    [torch.ones([bsz, self.n_head, 1], device=head_q.device),
                     key_denom[:, :, 1:].clone()], dim=2).unsqueeze(-1)
                head_beta = head_beta * key_denom 
                head_k = head_k / (key_denom + self.eps)
        else:
            assert carry_over_fast_weight
            mem_fast_weights, fast_denom = mems
            # bsz can be smaller for the last batch
            mem_fast_weights = mem_fast_weights[:bsz]
            if self.normalize_attn_scores:
                key_denom = torch.cat(
                    [torch.zeros(
                        [bsz, self.n_head, 1, 2 * self.n_roll * self.d_head],
                        device=head_q.device),
                        denominator_acc[:, :, :-1, :].clone()], dim=2)
                key_denom = key_denom + fast_denom[:bsz]
                denominator_acc = denominator_acc + fast_denom[:bsz]

                key_denom = torch.einsum(
                    'lbij,lbij->lbi', key_denom, head_k).unsqueeze(-1)
                head_beta = head_beta * key_denom
                head_k = head_k / (key_denom + self.eps)

        if self.normalize_attn_scores:
            denominator = torch.einsum(
                'lbij,lbij->lbi', denominator_acc, head_q).unsqueeze(-1)

        layer_out = fast_weight_delta(
            head_q, head_k, head_v, head_beta, mem_fast_weights)

        # print(mem_fast_weights.norm())
        # print(denominator_acc[:, :, -1, :].norm())

        # shape (B, n_head, len, d_head)
        if self.normalize_attn_scores:
            layer_out = self.scale * layer_out / (denominator + self.eps)
        else:
            layer_out = self.scale * layer_out

        layer_out = layer_out.transpose(1, 2)

        layer_out = layer_out.reshape(
            bsz, slen, self.n_head * self.d_head)

        layer_out = layer_out.transpose(0, 1)

        # expect [qlen, B, n_head * d_head]

        # linear projection
        attn_out = self.o_net(layer_out)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            # residual connection
            output = h + attn_out
        else:
            # residual connection + layer normalization
            output = self.layer_norm(h + attn_out)

        if carry_over_fast_weight:
            # last values of accumulator should be carried over.
            # clone is needed as backward modifies the data of fast weight
            if self.normalize_attn_scores:
                new_k_acc = denominator_acc[:, :, -1, :].unsqueeze(2).detach()
            else:
                new_k_acc = None
            new_mem = (mem_fast_weights.clone().detach(), new_k_acc)
            return output, new_mem

        return output


# Katharopoulos et al's Linear Transformer https://arxiv.org/abs/2006.16236
# = Sum update rule + ELU based attention function
# Deep fast network.
class CudaDeepFastNetSumLinearTransformerLayer(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout, dropatt=0,
                 pre_lnorm=False, eps=1e-5, layer_id=None, num_layer=None,
                 skip_attn_normalization=False, fast_net_depth=1):
        super(CudaDeepFastNetSumLinearTransformerLayer, self).__init__()
        print(f"Using CudaDeepFastNetSumLinearTransformerLayer {layer_id} -")

        assert layer_id is not None
        assert num_layer is not None
        self.layer_id = layer_id
        self.num_layer = num_layer
        self.fast_net_depth = fast_net_depth
        self.fast_ff_dim = 4 * d_model
        self.fast_ff_d_head = self.fast_ff_dim // n_head

        # layer norm for feed-forward blocks in the fast net.
        # NB: the weights in the layer norm are slow weights.
        self.fast_net_layer_norms = nn.ModuleList(
            [nn.LayerNorm(d_head) for _ in range(fast_net_depth)])

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout

        # (3 * d_head * n_head) for qkv.
        # self.qkv_net = nn.Linear(d_model, n_head * 3 * d_head, bias=False)
        # for each feed-forward block we have 2 layers:
        # - d_model to fast_ff_dim
        # - fast_ff_dim to d_model
        # this requires 2 * (d_model * d_ff) dims for each of key and value.
        # The total is therefore 4 * (d_model * d_ff) * fast_net_depth
        # plus d_model for query.
        self.total_qkv_dim = (
            4 * fast_net_depth * (d_model + self.fast_ff_dim) + d_model)
        self.qkv_net = nn.Linear(d_model, self.total_qkv_dim, bias=False)

        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)

        self.layer_norm = nn.LayerNorm(d_model)

        self.scale = 1 / (d_head ** 0.5)

        self.pre_lnorm = pre_lnorm
        self.normalize_attn_scores = (not skip_attn_normalization)
        self.eps = eps

    def forward(self, h, attn_mask=None, mems=None,
                carry_over_fast_weight=False):
        # multihead attention
        # shape h: (len, B, n_head * d_head)

        if self.pre_lnorm:
            # layer normalization
            h = self.layer_norm(h)

        slen, bsz, _ = h.size()

        qkv = self.qkv_net(h)
        qkv = qkv.view(slen, bsz, self.total_qkv_dim)
        head_q, head_kv = torch.split(
            qkv,
            [self.d_model,
             4 * self.fast_net_depth * (self.d_model + self.fast_ff_dim)], -1)
        head_q = head_q.view(slen, bsz, self.n_head, self.d_head)  # q ready

        head_kv = head_kv.reshape(
            slen, bsz, self.n_head,
            4 * self.fast_net_depth * (self.d_head + self.fast_ff_d_head))

        head_k, head_v = torch.split(
            head_kv,
            ((self.d_head + self.fast_ff_d_head) * 2 * self.fast_net_depth,
             ) * 2, -1)

        # reshape to (B, heads, len, dim)
        head_q = head_q.permute(1, 2, 0, 3)
        head_k = head_k.permute(1, 2, 0, 3)
        head_v = head_v.permute(1, 2, 0, 3)

        # TODO add dropout here?
        head_k = F.elu(head_k, 1., False) + 1.

        # normalize k and q, crucial for stable training.
        # key normalization is to be done for each layer.
        # head_k = head_k / head_k.sum(-1, keepdim=True)
        # sum normalization of the initial query
        head_q = head_q / (head_q.sum(-1, keepdim=True) + self.eps)

        head_v = torch.split(
            head_v,
            (self.fast_ff_d_head, self.d_head) * self.fast_net_depth * 2, -1)
        # head_v is now a tuple of size fast_net_depth * 2

        head_k = torch.split(
            head_k,
            (self.d_head, self.fast_ff_d_head) * self.fast_net_depth * 2, -1)
        # head_k is now a tuple of size fast_net_depth * 2

        for layer_id in range(self.fast_net_depth):
            layer_out = self.fast_net_layer_norms[layer_id](head_q)
            # Do we really elu to be here? TODO think
            layer_out = F.elu(layer_out, 1., False) + 1.
            # Do we really want layer norm here? TODO think
            layer_out = layer_out / (layer_out.sum(-1, keepdim=True) + self.eps)
            # sum normalization of keys
            cur_key = head_k[2 * layer_id]
            cur_key = cur_key / (cur_key.sum(-1, keepdim=True) + self.eps)
            mem_fast_weights = torch.zeros(
                bsz, self.n_head, self.d_head, self.fast_ff_d_head, device=head_q.device)
            layer_out = fast_weight_sum(
                layer_out, cur_key, head_v[2 * layer_id], mem_fast_weights)
            layer_out = F.elu(layer_out, 1., False) + 1.

            # sum normalization for key and query
            layer_out = layer_out / (layer_out.sum(-1, keepdim=True) + self.eps)
            cur_key = head_k[2 * layer_id + 1]
            cur_key = cur_key / (cur_key.sum(-1, keepdim=True) + self.eps)
            mem_fast_weights = torch.zeros(
                bsz, self.n_head, self.fast_ff_d_head, self.d_head, device=head_q.device)
            layer_out = fast_weight_sum(
                layer_out, cur_key, head_v[2 * layer_id + 1], mem_fast_weights)
            head_q = layer_out + head_q  # self.scale * head_q  # or not?

        layer_out = head_q.transpose(1, 2)

        layer_out = layer_out.reshape(bsz, slen, self.n_head * self.d_head)

        layer_out = layer_out.transpose(0, 1)

        # expect [qlen, B, n_head * d_head]

        # linear projection
        attn_out = self.o_net(layer_out)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            # residual connection
            output = h + attn_out
        else:
            # residual connection + layer normalization
            output = self.layer_norm(h + attn_out)

        if carry_over_fast_weight:
            # last values of accumulator should be carried over.
            # clone is needed as backward modifies the data of fast weight
            # if self.normalize_attn_scores:
            #     new_k_acc = denominator_acc[:, :, -1, :].unsqueeze(2).detach()
            # else:
            #     new_k_acc = None
            new_k_acc = None
            new_mem = (mem_fast_weights.clone().detach(), new_k_acc)
            return output, new_mem

        return output


# Katharopoulos et al's Linear Transformer https://arxiv.org/abs/2006.16236
# = Sum update rule + ELU based attention function
# Deep fast network.
class CudaDeepFastNetLayer(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout, dropatt=0,
                 pre_lnorm=False, eps=1e-5, layer_id=None, num_layer=None,
                 skip_attn_normalization=False, fast_net_depth=1):
        super(CudaDeepFastNetLayer, self).__init__()
        print(f"Using CudaDeepFastNetLayer {layer_id} -")

        assert layer_id is not None
        assert num_layer is not None
        self.layer_id = layer_id
        self.num_layer = num_layer
        self.fast_net_depth = fast_net_depth
        self.fast_ff_dim = 4 * d_model
        self.fast_ff_d_head = self.fast_ff_dim // n_head

        # layer norm for feed-forward blocks in the fast net.
        # NB: the weights in the layer norm are slow weights.
        self.fast_net_layer_norms = nn.ModuleList(
            [nn.LayerNorm(d_head) for _ in range(fast_net_depth)])

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout

        # for each feed-forward block we have 2 layers:
        # - d_model to fast_ff_dim
        # - fast_ff_dim to d_model
        # this requires 2 * (d_model * d_ff) dims for each of key and value.
        # The total is therefore 4 * (d_model * d_ff) * fast_net_depth
        # plus d_model for query, + 2 * fast_net_depth for beta
        self.total_qkv_dim = (
            4 * fast_net_depth * (d_model + self.fast_ff_dim) + d_model
            + 2 * fast_net_depth * n_head)
        self.qkv_net = nn.Linear(d_model, self.total_qkv_dim, bias=False)

        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)

        self.layer_norm = nn.LayerNorm(d_model)

        self.scale = 1 / (d_head ** 0.5)

        self.pre_lnorm = pre_lnorm
        self.normalize_attn_scores = (not skip_attn_normalization)
        self.eps = eps

    def forward(self, h, attn_mask=None, mems=None,
                carry_over_fast_weight=False):
        # multihead attention
        # shape h: (len, B, n_head * d_head)

        if self.pre_lnorm:
            # layer normalization
            h = self.layer_norm(h)

        slen, bsz, _ = h.size()

        qkv = self.qkv_net(h)
        qkv = qkv.view(slen, bsz, self.total_qkv_dim)
        head_q, head_kv, head_beta = torch.split(
            qkv,
            [self.d_model,
             4 * self.fast_net_depth * (self.d_model + self.fast_ff_dim),
             2 * self.fast_net_depth * self.n_head], -1)
        head_q = head_q.view(slen, bsz, self.n_head, self.d_head)  # q ready

        head_beta = torch.sigmoid(head_beta)
        head_beta = head_beta.reshape(
            slen, bsz, self.n_head, 2 * self.fast_net_depth)

        head_kv = head_kv.reshape(
            slen, bsz, self.n_head,
            4 * self.fast_net_depth * (self.d_head + self.fast_ff_d_head))

        head_k, head_v = torch.split(
            head_kv,
            ((self.d_head + self.fast_ff_d_head) * 2 * self.fast_net_depth,
             ) * 2, -1)

        # reshape to (B, heads, len, dim)
        head_q = head_q.permute(1, 2, 0, 3)
        head_k = head_k.permute(1, 2, 0, 3)
        head_v = head_v.permute(1, 2, 0, 3)
        head_beta = head_beta.permute(1, 2, 0, 3)

        # TODO add dropout here?
        # transform q and k
        head_k = elu_p1(head_k)

        head_v = torch.split(
            head_v,
            (self.fast_ff_d_head, self.d_head) * self.fast_net_depth * 2, -1)
        # head_v is now a tuple of size fast_net_depth * 2

        head_k = torch.split(
            head_k,
            (self.d_head, self.fast_ff_d_head) * self.fast_net_depth * 2, -1)
        # head_k is now a tuple of size fast_net_depth * 2

        head_beta = torch.split(head_beta, (1,) * self.fast_net_depth * 2, -1)

        for layer_id in range(self.fast_net_depth):
            layer_out = self.fast_net_layer_norms[layer_id](head_q)
            layer_out = elu_p1(layer_out)
            layer_out = sum_norm_eps(layer_out)
            # sum normalization of keys
            cur_key = head_k[2 * layer_id]
            cur_key = sum_norm_eps(cur_key)
            cur_beta = head_beta[2 * layer_id]
            mem_fast_weights = torch.zeros(
                bsz, self.n_head, self.d_head, self.fast_ff_d_head,
                device=head_q.device)
            layer_out = fast_weight_delta(
                layer_out, cur_key, head_v[2 * layer_id], cur_beta,
                mem_fast_weights)
            layer_out = elu_p1(layer_out)

            # sum normalization for key and query
            layer_out = sum_norm_eps(layer_out)
            cur_key = head_k[2 * layer_id + 1]
            cur_key = sum_norm_eps(cur_key)
            cur_beta = head_beta[2 * layer_id + 1]
            mem_fast_weights = torch.zeros(
                bsz, self.n_head, self.fast_ff_d_head, self.d_head,
                device=head_q.device)
            layer_out = fast_weight_delta(
                layer_out, cur_key, head_v[2 * layer_id + 1], cur_beta,
                mem_fast_weights)
            head_q = layer_out + head_q  # self.scale * head_q  # or not?

        layer_out = head_q.transpose(1, 2)

        layer_out = layer_out.reshape(bsz, slen, self.n_head * self.d_head)

        layer_out = layer_out.transpose(0, 1)

        # expect [qlen, B, n_head * d_head]

        # linear projection
        attn_out = self.o_net(layer_out)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            # residual connection
            output = h + attn_out
        else:
            # residual connection + layer normalization
            output = self.layer_norm(h + attn_out)

        if carry_over_fast_weight:
            # last values of accumulator should be carried over.
            # clone is needed as backward modifies the data of fast weight
            new_k_acc = None
            new_mem = (mem_fast_weights.clone().detach(), new_k_acc)
            return output, new_mem

        return output


class CudaDeltaDeltaLayer(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout, dropatt=0,
                 pre_lnorm=False, eps=1e-5, layer_id=None, num_layer=None,
                 skip_attn_normalization=True, fast_net_depth=1,
                 use_slow_base_weights=False):
        super(CudaDeltaDeltaLayer, self).__init__()
        print(f"Using CudaDeltaDeltaLayer {layer_id} -")
        print("skip_attn_normalization flag ignored in this layer.")
        print(f"use_slow_base_weights: {use_slow_base_weights}")

        assert layer_id is not None
        assert num_layer is not None
        self.layer_id = layer_id
        self.num_layer = num_layer

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout

        # value represents fast qkvq, so 3 times d_model + 1
        # + 2 d_model for q and k as usual, + 1 for beta
        self.total_qkv_dim = n_head * (5 * d_head + 2)
        self.slow_net = nn.Linear(d_model, self.total_qkv_dim, bias=False)

        # Base slow weights for the fast transformer.
        self.use_slow_base_weights = use_slow_base_weights
        if use_slow_base_weights:
            self.fast_net_slow_weights = torch.nn.Parameter(
                torch.rand(n_head, d_head, 3 * d_head + 1)
                + torch.ones(n_head, d_head, 3 * d_head + 1),
                requires_grad=True)
            # the init above works well.
            # bound = 1 / math.sqrt(3 * d_head * n_head)
            # nn_init.uniform_(self.fast_net_slow_weights, -bound, bound)

        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)

        self.layer_norm = nn.LayerNorm(d_model)

        self.scale = 1 / (d_head ** 0.5)

        self.pre_lnorm = pre_lnorm
        self.normalize_attn_scores = False
        self.eps = eps

    def forward(self, h, attn_mask=None, mems=None,
                carry_over_fast_weight=False):
        assert carry_over_fast_weight is False, "Not supported yet."
        # multihead attention
        # shape h: (len, B, n_head * d_head)

        if self.pre_lnorm:
            # layer normalization
            h = self.layer_norm(h)

        slen, bsz, _ = h.size()

        qkv = self.slow_net(h)
        qkv = qkv.view(slen, bsz, self.n_head, 5 * self.d_head + 2)
        head_q, head_k, head_v, head_beta = torch.split(
            qkv, (self.d_head, self.d_head, 3 * self.d_head + 1, 1), -1)
        head_beta = torch.sigmoid(head_beta)

        # reshape to (B, heads, len, dim)
        head_q = head_q.permute(1, 2, 0, 3)
        head_k = head_k.permute(1, 2, 0, 3)
        head_v = head_v.permute(1, 2, 0, 3)
        head_beta = head_beta.permute(1, 2, 0, 3)

        # TODO add dropout here?
        # transform q and k
        head_q = elu_p1_sum_norm_eps(head_q)
        head_k = elu_p1_sum_norm_eps(head_k)

        if self.use_slow_base_weights:
            mem_fast_weights = self.fast_net_slow_weights.unsqueeze(0).repeat(
                bsz, 1, 1, 1)
        else:
            mem_fast_weights = torch.zeros(
                bsz, self.n_head, self.d_head, 3 * self.d_head + 1,
                device=head_q.device)

        fast_qkvb = fast_weight_delta(
            head_q, head_k, head_v, head_beta, mem_fast_weights)

        fast_head_q, fast_head_k, fast_head_v, fast_beta = torch.split(
            fast_qkvb, (self.d_head,) * 3 + (1,), -1)

        fast_head_q = elu_p1_sum_norm_eps(fast_head_q)
        fast_head_k = elu_p1_sum_norm_eps(fast_head_k)
        fast_beta = torch.sigmoid(fast_beta)

        mem_fast_weights = torch.zeros(
            bsz, self.n_head, self.d_head, self.d_head, device=head_q.device)

        layer_out = fast_weight_delta(
            fast_head_q, fast_head_k, fast_head_v, fast_beta, mem_fast_weights)

        layer_out = layer_out.transpose(1, 2)

        layer_out = layer_out.reshape(bsz, slen, self.n_head * self.d_head)

        layer_out = layer_out.transpose(0, 1)

        # expect [qlen, B, n_head * d_head]

        # linear projection
        attn_out = self.o_net(layer_out)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            # residual connection
            output = h + attn_out
        else:
            # residual connection + layer normalization
            output = self.layer_norm(h + attn_out)

        return output


# 124, Fast RNN layer with FWM update rule
class CudaFastRNNLayer(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout, dropatt=0,
                 pre_lnorm=False, eps=1e-5, layer_id=None, num_layer=None,
                 use_lnorm=True, use_res=True, use_out_proj=False,
                 d_res=None, skip_attn_normalization=False, use_sum_norm=True):
        super(CudaFastRNNLayer, self).__init__()
        print(f"Using CudaFastRNNLayer {layer_id} -")
        print(f"skip_attn_normalization: {skip_attn_normalization}")
        print(f"use_sum_norm: {use_sum_norm}")

        assert layer_id is not None
        assert num_layer is not None
        self.layer_id = layer_id
        self.num_layer = num_layer

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout

        self.d_res = d_res
        if d_res is None:
            d_res = d_model

        self.use_lnorm = use_lnorm
        self.use_res = use_res
        self.use_out_proj = use_out_proj

        # (3 * d_head * n_head) for qkv and recurrent kv.
        self.qkvb_net = nn.Linear(
            d_res, n_head * (5 * d_head + 2), bias=False)

        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)

        if use_out_proj:
            self.out_proj = nn.Linear(n_head * d_head, d_res, bias=False)

        if use_lnorm:
            self.layer_norm = nn.LayerNorm(d_res)

        # self.scale = 1 / (d_head ** 0.5)

        self.pre_lnorm = pre_lnorm
        self.normalize_attn_scores = (not skip_attn_normalization)
        self.use_sum_norm = use_sum_norm
        self.eps = eps

    def forward(self, h, attn_mask=None, mems=None,
                carry_over_fast_weight=False):
        # multihead attention
        # shape h: (len, B, n_head * d_head)

        if self.use_lnorm and self.pre_lnorm:
            # layer normalization
            h = self.layer_norm(h)

        slen, bsz, _ = h.size()

        qkv = self.qkvb_net(h)
        qkv = qkv.view(slen, bsz, self.n_head, 5 * self.d_head + 2)
        (head_q, head_k, head_v, rec_head_k, rec_head_v, head_beta,
         rec_beta) = torch.split(qkv, (self.d_head,) * 5 + (1,) * 2, -1)

        head_beta = torch.sigmoid(head_beta)
        rec_beta = torch.sigmoid(rec_beta)

        # reshape to (B, heads, len, dim)
        head_q = head_q.permute(1, 2, 0, 3)
        head_k = head_k.permute(1, 2, 0, 3)
        head_v = head_v.permute(1, 2, 0, 3)
        head_beta = head_beta.permute(1, 2, 0, 3)

        rec_head_k = rec_head_k.permute(1, 2, 0, 3)
        rec_head_v = rec_head_v.permute(1, 2, 0, 3)
        rec_beta = rec_beta.permute(1, 2, 0, 3)

        # TODO add dropout here?
        # transform q and k
        head_q = F.softmax(head_q, dim=-1)
        head_k = F.softmax(head_k, dim=-1)
        # make recurrent key consistent with rec activation
        rec_head_k = F.softmax(rec_head_k, dim=-1)

        if self.normalize_attn_scores:
            denominator_acc = torch.cumsum(head_k, dim=2)

        if mems is None:
            mem_fast_weights = torch.zeros(
                bsz, self.n_head, self.d_head, self.d_head,
                device=head_k.device)
            mem_rec_fast_weights = torch.zeros(
                bsz, self.n_head, self.d_head, self.d_head,
                device=head_k.device)
            state0 = torch.zeros(
                bsz, self.n_head, 1, self.d_head, device=head_k.device)
        else:
            assert carry_over_fast_weight
            mem_fast_weights, mem_rec_fast_weights, state0 = mems
            # mem_fast_weights, fast_denom = mems
            # bsz can be smaller for the last batch
            mem_fast_weights = mem_fast_weights[:bsz]
            mem_rec_fast_weights = mem_rec_fast_weights[:bsz]
            state0 = state0[:bsz]
            if self.normalize_attn_scores:
                assert False, "Not implemented yet!"
                denominator_acc = denominator_acc + fast_denom[:bsz]

        if self.normalize_attn_scores:
            denominator = torch.einsum(
                'lbij,lbij->lbi', denominator_acc, head_q).unsqueeze(-1)

        # feed-forward part
        z_out = fast_weight_delta(
            head_q, head_k, head_v, head_beta, mem_fast_weights)

        # recurrent part
        layer_out = fast_rnn(
            z_out, rec_head_k, rec_head_v, mem_rec_fast_weights, rec_beta,
            state0)
        # shape (B, n_head, len, d_head)

        if carry_over_fast_weight:  # clone state
            # layer_out shape (B, n_head, len, d_head)
            state0_next = layer_out[:, :, -1, :].clone().detach()
            state0_next = state0_next.unsqueeze(2)

        if self.normalize_attn_scores:
            layer_out = layer_out / (denominator + self.eps)

        layer_out = layer_out.transpose(1, 2)

        layer_out = layer_out.reshape(bsz, slen, self.n_head * self.d_head)

        layer_out = layer_out.transpose(0, 1)

        # expect [qlen, B, n_head * d_head]

        # linear projection
        if self.use_out_proj:
            layer_out = self.drop(layer_out)
            layer_out = self.out_proj(layer_out)
        attn_out = self.drop(layer_out)

        if self.use_res:
            if self.use_lnorm:
                if self.pre_lnorm:
                    # residual connection
                    output = h + attn_out
                else:
                    # residual connection + layer normalization
                    output = self.layer_norm(h + attn_out)
            else:
                output = h + attn_out
        else:
            if self.use_lnorm:
                if self.pre_lnorm:
                    # residual connection
                    output = attn_out
                else:
                    # residual connection + layer normalization
                    output = self.layer_norm(attn_out)
            else:
                output = attn_out

        if carry_over_fast_weight:
            # last values of accumulator should be carried over.
            # clone is needed as backward modifies the data of fast weight
            if self.normalize_attn_scores:
                assert False, "Not implemented."
                new_k_acc = denominator_acc[:, :, -1, :].unsqueeze(2).detach()
            else:
                new_k_acc = None
            # new_mem = (mem_fast_weights.clone().detach(), new_k_acc)
            new_mem = (mem_fast_weights.clone().detach(),
                       mem_rec_fast_weights.clone().detach(), state0_next)
            return output, new_mem

        return output


# Fast RNN layer with FWM update rule
class CudaFastRNNv2Layer(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout, dropatt=0,
                 pre_lnorm=False, eps=1e-5, layer_id=None, num_layer=None,
                 skip_attn_normalization=False, use_sum_norm=True):
        super(CudaFastRNNv2Layer, self).__init__()
        print(f"Using CudaFastRNNv2Layer {layer_id} -")
        print(f"skip_attn_normalization: {skip_attn_normalization}")
        print(f"use_sum_norm: {use_sum_norm}")

        assert layer_id is not None
        assert num_layer is not None
        self.layer_id = layer_id
        self.num_layer = num_layer

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout

        # (3 * d_head * n_head) for qkv and recurrent kv.
        self.qkvb_net = nn.Linear(
            d_model, n_head * (5 * d_head + 2), bias=False)

        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)

        self.layer_norm = nn.LayerNorm(d_model)

        self.scale = 1 / (d_head ** 0.5)

        self.pre_lnorm = pre_lnorm
        self.normalize_attn_scores = (not skip_attn_normalization)
        self.use_sum_norm = use_sum_norm
        self.eps = eps

    def forward(self, h, attn_mask=None, mems=None,
                carry_over_fast_weight=False):
        # multihead attention
        # shape h: (len, B, n_head * d_head)

        if self.pre_lnorm:
            # layer normalization
            h = self.layer_norm(h)

        slen, bsz, _ = h.size()

        qkv = self.qkvb_net(h)
        qkv = qkv.view(slen, bsz, self.n_head, 5 * self.d_head + 2)
        (head_q, head_k, head_v, rec_head_k, rec_head_v, head_beta,
         rec_beta) = torch.split(qkv, (self.d_head,) * 5 + (1,) * 2, -1)

        head_beta = torch.sigmoid(head_beta)
        rec_beta = torch.sigmoid(rec_beta)

        # reshape to (B, heads, len, dim)
        head_q = head_q.permute(1, 2, 0, 3)
        head_k = head_k.permute(1, 2, 0, 3)
        head_v = head_v.permute(1, 2, 0, 3)
        head_beta = head_beta.permute(1, 2, 0, 3)

        rec_head_k = rec_head_k.permute(1, 2, 0, 3)
        rec_head_v = rec_head_v.permute(1, 2, 0, 3)
        rec_beta = rec_beta.permute(1, 2, 0, 3)

        # TODO add dropout here?
        # transform q and k
        head_q = F.elu(head_q, 1., False) + 1.
        head_k = F.elu(head_k, 1., False) + 1.
        # make recurrent key consistent with rec activation
        rec_head_k = F.softmax(rec_head_k, dim=-1)

        # normalize k and q, crucial for stable training.
        if self.use_sum_norm:
            head_k = head_k / head_k.sum(-1, keepdim=True)
            head_q = head_q / head_q.sum(-1, keepdim=True)

        if self.normalize_attn_scores:
            denominator_acc = torch.cumsum(head_k, dim=2)

        if mems is None:
            mem_fast_weights = torch.zeros(
                bsz, self.n_head, self.d_head, self.d_head,
                device=head_k.device)
            mem_rec_fast_weights = torch.zeros(
                bsz, self.n_head, self.d_head, self.d_head,
                device=head_k.device)
            state0 = torch.zeros(
                bsz, self.n_head, 1, self.d_head, device=head_k.device)
        else:
            assert carry_over_fast_weight
            mem_fast_weights, mem_rec_fast_weights, state0 = mems
            # mem_fast_weights, fast_denom = mems
            # bsz can be smaller for the last batch
            mem_fast_weights = mem_fast_weights[:bsz]
            mem_rec_fast_weights = mem_rec_fast_weights[:bsz]
            state0 = state0[:bsz]
            if self.normalize_attn_scores:
                assert False, "Not implemented yet!"
                denominator_acc = denominator_acc + fast_denom[:bsz]

        if self.normalize_attn_scores:
            denominator = torch.einsum(
                'lbij,lbij->lbi', denominator_acc, head_q).unsqueeze(-1)

        # feed-forward part
        z_out = fast_weight_delta(
            head_q, head_k, head_v, head_beta, mem_fast_weights)

        # recurrent part
        layer_out = fast_rnn_v2(
            z_out, rec_head_k, rec_head_v, mem_rec_fast_weights, rec_beta,
            state0)

        if carry_over_fast_weight:  # clone state
            # layer_out shape (B, n_head, len, d_head)
            state0_next = layer_out[:, :, -1, :].clone().detach()
            state0_next = state0_next.unsqueeze(2)

        # shape (B, n_head, len, d_head)
        if self.normalize_attn_scores:
            layer_out = self.scale * layer_out / (denominator + self.eps)
        else:
            layer_out = self.scale * layer_out

        layer_out = layer_out.transpose(1, 2)

        layer_out = layer_out.reshape(bsz, slen, self.n_head * self.d_head)

        layer_out = layer_out.transpose(0, 1)

        # expect [qlen, B, n_head * d_head]

        # linear projection
        attn_out = self.o_net(layer_out)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            # residual connection
            output = h + attn_out
        else:
            # residual connection + layer normalization
            output = self.layer_norm(h + attn_out)

        if carry_over_fast_weight:
            # last values of accumulator should be carried over.
            # clone is needed as backward modifies the data of fast weight
            if self.normalize_attn_scores:
                assert False, "Not implemented."
                new_k_acc = denominator_acc[:, :, -1, :].unsqueeze(2).detach()
            else:
                new_k_acc = None
            new_mem = (mem_fast_weights.clone().detach(),
                       mem_rec_fast_weights.clone().detach(), state0_next)
            return output, new_mem

        return output


# 224, Fast LSTM layer with FWM update rule
class CudaFastLSTMLayer(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout, dropatt=0,
                 pre_lnorm=False, eps=1e-5, layer_id=None, num_layer=None,
                 use_lnorm=True, use_res=True, use_out_proj=False,
                 d_res=None, skip_attn_normalization=True, use_sum_norm=True):
        super(CudaFastLSTMLayer, self).__init__()
        print(f"Using CudaFastLSTMLayer {layer_id} -")
        print(f"skip_attn_normalization: {skip_attn_normalization}")
        print(f"use_sum_norm: {use_sum_norm}")

        assert layer_id is not None
        assert num_layer is not None
        self.layer_id = layer_id
        self.num_layer = num_layer

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout

        self.d_res = d_res
        if d_res is None:
            d_res = d_model
        self.use_lnorm = use_lnorm
        self.use_res = use_res
        self.use_out_proj = use_out_proj

        self.qkv_net = nn.Linear(
            d_res, n_head * (15 * d_head + 2 * 3), bias=False)

        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        if use_out_proj:
            self.out_proj = nn.Linear(n_head * d_head, d_res, bias=False)

        if use_lnorm:
            self.layer_norm = nn.LayerNorm(d_res)

        self.scale = 1 / (d_head ** 0.5)

        self.pre_lnorm = pre_lnorm
        self.normalize_attn_scores = (not skip_attn_normalization)
        self.use_sum_norm = use_sum_norm
        self.eps = eps

    def forward(self, h, attn_mask=None, mems=None,
                carry_over_fast_weight=False):
        # multihead attention
        # shape h: (len, B, n_head * d_head)

        if self.use_lnorm and self.pre_lnorm:
            # layer normalization
            h = self.layer_norm(h)

        slen, bsz, _ = h.size()

        qkv = self.qkv_net(h)
        qkv = qkv.view(slen, bsz, self.n_head, 15 * self.d_head + 6)
        (head_qi, head_ki, head_vi, rec_head_ki, rec_head_vi,
         head_qu, head_ku, head_vu, rec_head_ku, rec_head_vu,
         head_qo, head_ko, head_vo, rec_head_ko, rec_head_vo,
         head_beta_i, rec_beta_i,
         head_beta_u, rec_beta_u,
         head_beta_o, rec_beta_o) = torch.split(
             qkv, (self.d_head,) * 15 + (1,) * 6, -1)

        head_beta_i = torch.sigmoid(head_beta_i)
        rec_beta_i = torch.sigmoid(rec_beta_i)
        head_beta_u = torch.sigmoid(head_beta_u)
        rec_beta_u = torch.sigmoid(rec_beta_u)
        head_beta_o = torch.sigmoid(head_beta_o)
        rec_beta_o = torch.sigmoid(rec_beta_o)

        # Reshape to (B, heads, len, dim)
        # input gate
        head_qi = head_qi.permute(1, 2, 0, 3)
        head_ki = head_ki.permute(1, 2, 0, 3)
        head_vi = head_vi.permute(1, 2, 0, 3)
        head_beta_i = head_beta_i.permute(1, 2, 0, 3)

        rec_head_ki = rec_head_ki.permute(1, 2, 0, 3)
        rec_head_vi = rec_head_vi.permute(1, 2, 0, 3)
        rec_beta_i = rec_beta_i.permute(1, 2, 0, 3)

        # update term
        head_qu = head_qu.permute(1, 2, 0, 3)
        head_ku = head_ku.permute(1, 2, 0, 3)
        head_vu = head_vu.permute(1, 2, 0, 3)
        head_beta_u = head_beta_u.permute(1, 2, 0, 3)

        rec_head_ku = rec_head_ku.permute(1, 2, 0, 3)
        rec_head_vu = rec_head_vu.permute(1, 2, 0, 3)
        rec_beta_u = rec_beta_u.permute(1, 2, 0, 3)

        # output gate
        head_qo = head_qo.permute(1, 2, 0, 3)
        head_ko = head_ko.permute(1, 2, 0, 3)
        head_vo = head_vo.permute(1, 2, 0, 3)
        head_beta_o = head_beta_o.permute(1, 2, 0, 3)

        rec_head_ko = rec_head_ko.permute(1, 2, 0, 3)
        rec_head_vo = rec_head_vo.permute(1, 2, 0, 3)
        rec_beta_o = rec_beta_o.permute(1, 2, 0, 3)

        # TODO add dropout here?
        # transform q and k for the forward part
        head_qi = elu_p1(head_qi)
        head_ki = elu_p1(head_ki)
        head_qu = elu_p1(head_qu)
        head_ku = elu_p1(head_ku)
        head_qo = elu_p1(head_qo)
        head_ko = elu_p1(head_ko)

        # make recurrent key consistent with rec activation
        rec_head_ki = F.softmax(rec_head_ki, dim=-1)
        rec_head_ku = F.softmax(rec_head_ku, dim=-1)
        rec_head_ko = F.softmax(rec_head_ko, dim=-1)
        # this performed better than elu+1

        # normalize k and q, crucial for stable training.
        # replaced by softmax
        if self.use_sum_norm:
            head_ki = sum_norm(head_ki)
            head_qi = sum_norm(head_qi)
            head_ku = sum_norm(head_ku)
            head_qu = sum_norm(head_qu)
            head_ko = sum_norm(head_ko)
            head_qo = sum_norm(head_qo)

        if self.normalize_attn_scores:
            assert False, "`normalize_attn_scores` not supported."
            # denominator_acc = torch.cumsum(head_k, dim=2)

        if mems is None:
            # lstm states
            state0 = torch.zeros(
                bsz, self.n_head, 1, self.d_head, device=head_ki.device)
            cell0 = torch.zeros(
                bsz, self.n_head, 1, self.d_head, device=head_ki.device)
            # input gate
            mem_fast_weights_i = torch.zeros(
                bsz, self.n_head, self.d_head, self.d_head,
                device=head_ki.device)
            mem_rec_fast_weights_i = torch.zeros(
                bsz, self.n_head, self.d_head, self.d_head,
                device=head_ki.device)
            # update term
            mem_fast_weights_u = torch.zeros(
                bsz, self.n_head, self.d_head, self.d_head,
                device=head_ki.device)
            mem_rec_fast_weights_u = torch.zeros(
                bsz, self.n_head, self.d_head, self.d_head,
                device=head_ki.device)
            # output gate
            mem_fast_weights_o = torch.zeros(
                bsz, self.n_head, self.d_head, self.d_head,
                device=head_ki.device)
            mem_rec_fast_weights_o = torch.zeros(
                bsz, self.n_head, self.d_head, self.d_head,
                device=head_ki.device)
        else:
            assert carry_over_fast_weight
            (mem_fast_weights_i, mem_fast_weights_u, mem_fast_weights_o,
             mem_rec_fast_weights_i, mem_rec_fast_weights_u,
             mem_rec_fast_weights_o, state0, cell0) = mems
            # mem_fast_weights, fast_denom = mems
            # bsz can be smaller for the last batch
            mem_fast_weights_i = mem_fast_weights_i[:bsz]
            mem_fast_weights_u = mem_fast_weights_u[:bsz]
            mem_fast_weights_o = mem_fast_weights_o[:bsz]

            mem_rec_fast_weights_i = mem_rec_fast_weights_i[:bsz]
            mem_rec_fast_weights_u = mem_rec_fast_weights_u[:bsz]
            mem_rec_fast_weights_o = mem_rec_fast_weights_o[:bsz]

            state0 = state0[:bsz]
            cell0 = cell0[:bsz]

            if self.normalize_attn_scores:
                assert False, "Not implemented yet!"
                denominator_acc = denominator_acc + fast_denom[:bsz]

        if self.normalize_attn_scores:
            assert False, "Not supported."
            # denominator = torch.einsum(
            #     'lbij,lbij->lbi', denominator_acc, head_q).unsqueeze(-1)

        # feed-forward part
        zi = fast_weight_delta(
            head_qi, head_ki, head_vi, head_beta_i, mem_fast_weights_i)
        zu = fast_weight_delta(
            head_qu, head_ku, head_vu, head_beta_u, mem_fast_weights_u)
        zo = fast_weight_delta(
            head_qo, head_ko, head_vo, head_beta_o, mem_fast_weights_o)

        # recurrent part
        layer_out, cell_out = fast_lstm(zi,
                                        rec_head_ki, rec_head_vi, rec_beta_i,
                                        mem_rec_fast_weights_i,
                                        zu,
                                        rec_head_ku, rec_head_vu, rec_beta_u,
                                        mem_rec_fast_weights_u,
                                        zo,
                                        rec_head_ko, rec_head_vo, rec_beta_o,
                                        mem_rec_fast_weights_o, state0, cell0)

        if carry_over_fast_weight:  # clone state
            # layer_out shape (B, n_head, len, d_head)
            state0_next = layer_out[:, :, -1, :].clone().detach()
            state0_next = state0_next.unsqueeze(2)
            cell0_next = cell_out[:, :, -1, :].clone().detach()
            cell0_next = cell0_next.unsqueeze(2)

        # shape (B, n_head, len, d_head)
        if self.normalize_attn_scores:
            assert False, "Not supported."
            layer_out = self.scale * layer_out / (denominator + self.eps)
        # else:
        #     layer_out = self.scale * layer_out

        layer_out = layer_out.transpose(1, 2)
        layer_out = layer_out.reshape(bsz, slen, self.n_head * self.d_head)
        layer_out = layer_out.transpose(0, 1)

        # expect [qlen, B, n_head * d_head]
        # linear projection
        if self.use_out_proj:
            layer_out = self.drop(layer_out)
            layer_out = self.out_proj(layer_out)
        attn_out = self.drop(layer_out)

        if self.use_res:
            if self.use_lnorm:
                if self.pre_lnorm:
                    # residual connection
                    output = h + attn_out
                else:
                    # residual connection + layer normalization
                    output = self.layer_norm(h + attn_out)
            else:
                output = h + attn_out
        else:
            if self.use_lnorm:
                if self.pre_lnorm:
                    # residual connection
                    output = attn_out
                else:
                    # residual connection + layer normalization
                    output = self.layer_norm(attn_out)
            else:
                output = attn_out

        if carry_over_fast_weight:
            # last values of accumulator should be carried over.
            # clone is needed as backward modifies the data of fast weight
            if self.normalize_attn_scores:
                assert False, "Not implemented."
                new_k_acc = denominator_acc[:, :, -1, :].unsqueeze(2).detach()
            else:
                new_k_acc = None
            # new_mem = (mem_fast_weights.clone().detach(), new_k_acc)
            new_mem = (mem_fast_weights_i.clone().detach(),
                       mem_fast_weights_u.clone().detach(),
                       mem_fast_weights_o.clone().detach(),
                       mem_rec_fast_weights_i.clone().detach(),
                       mem_rec_fast_weights_u.clone().detach(),
                       mem_rec_fast_weights_o.clone().detach(),
                       state0_next, cell0_next)
            return output, new_mem

        return output


# Fast LSTM layer with FWM update rule
class CudaFastLSTMv2Layer(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout, dropatt=0,
                 pre_lnorm=False, eps=1e-5, layer_id=None, num_layer=None,
                 skip_attn_normalization=True, use_sum_norm=True):
        super(CudaFastLSTMv2Layer, self).__init__()
        print(f"Using CudaFastLSTMv2Layer {layer_id} -")
        print(f"skip_attn_normalization: {skip_attn_normalization}")
        print(f"use_sum_norm: {use_sum_norm}")

        assert layer_id is not None
        assert num_layer is not None
        self.layer_id = layer_id
        self.num_layer = num_layer

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout

        self.qkv_net = nn.Linear(
            d_model, n_head * (15 * d_head + 2 * 3), bias=False)

        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)

        self.layer_norm = nn.LayerNorm(d_model)

        self.scale = 1 / (d_head ** 0.5)

        self.pre_lnorm = pre_lnorm
        self.normalize_attn_scores = (not skip_attn_normalization)
        self.use_sum_norm = use_sum_norm
        self.eps = eps

    def forward(self, h, attn_mask=None, mems=None,
                carry_over_fast_weight=False):
        # multihead attention
        # shape h: (len, B, n_head * d_head)

        if self.pre_lnorm:
            # layer normalization
            h = self.layer_norm(h)

        slen, bsz, _ = h.size()

        qkv = self.qkv_net(h)
        qkv = qkv.view(slen, bsz, self.n_head, 15 * self.d_head + 6)
        (head_qi, head_ki, head_vi, rec_head_ki, rec_head_vi,
         head_qu, head_ku, head_vu, rec_head_ku, rec_head_vu,
         head_qo, head_ko, head_vo, rec_head_ko, rec_head_vo,
         head_beta_i, rec_beta_i,
         head_beta_u, rec_beta_u,
         head_beta_o, rec_beta_o) = torch.split(
             qkv, (self.d_head,) * 15 + (1,) * 6, -1)

        head_beta_i = torch.sigmoid(head_beta_i)
        rec_beta_i = torch.sigmoid(rec_beta_i)
        head_beta_u = torch.sigmoid(head_beta_u)
        rec_beta_u = torch.sigmoid(rec_beta_u)
        head_beta_o = torch.sigmoid(head_beta_o)
        rec_beta_o = torch.sigmoid(rec_beta_o)

        # Reshape to (B, heads, len, dim)
        # input gate
        head_qi = head_qi.permute(1, 2, 0, 3)
        head_ki = head_ki.permute(1, 2, 0, 3)
        head_vi = head_vi.permute(1, 2, 0, 3)
        head_beta_i = head_beta_i.permute(1, 2, 0, 3)

        rec_head_ki = rec_head_ki.permute(1, 2, 0, 3)
        rec_head_vi = rec_head_vi.permute(1, 2, 0, 3)
        rec_beta_i = rec_beta_i.permute(1, 2, 0, 3)

        # update term
        head_qu = head_qu.permute(1, 2, 0, 3)
        head_ku = head_ku.permute(1, 2, 0, 3)
        head_vu = head_vu.permute(1, 2, 0, 3)
        head_beta_u = head_beta_u.permute(1, 2, 0, 3)

        rec_head_ku = rec_head_ku.permute(1, 2, 0, 3)
        rec_head_vu = rec_head_vu.permute(1, 2, 0, 3)
        rec_beta_u = rec_beta_u.permute(1, 2, 0, 3)

        # output gate
        head_qo = head_qo.permute(1, 2, 0, 3)
        head_ko = head_ko.permute(1, 2, 0, 3)
        head_vo = head_vo.permute(1, 2, 0, 3)
        head_beta_o = head_beta_o.permute(1, 2, 0, 3)

        rec_head_ko = rec_head_ko.permute(1, 2, 0, 3)
        rec_head_vo = rec_head_vo.permute(1, 2, 0, 3)
        rec_beta_o = rec_beta_o.permute(1, 2, 0, 3)

        # TODO add dropout here?
        # transform q and k for the forward part
        head_qi = F.elu(head_qi, 1., False) + 1.
        head_ki = F.elu(head_ki, 1., False) + 1.
        head_qu = F.elu(head_qu, 1., False) + 1.
        head_ku = F.elu(head_ku, 1., False) + 1.
        head_qo = F.elu(head_qo, 1., False) + 1.
        head_ko = F.elu(head_ko, 1., False) + 1.

        # make recurrent key consistent with rec activation
        rec_head_ki = F.softmax(rec_head_ki, dim=-1)
        rec_head_ku = F.softmax(rec_head_ku, dim=-1)
        rec_head_ko = F.softmax(rec_head_ko, dim=-1)
        # this performed better than elu+1

        # normalize k and q, crucial for stable training.
        if self.use_sum_norm:
            head_ki = head_ki / head_ki.sum(-1, keepdim=True)
            head_qi = head_qi / head_qi.sum(-1, keepdim=True)
            head_ku = head_ku / head_ku.sum(-1, keepdim=True)
            head_qu = head_qu / head_qu.sum(-1, keepdim=True)
            head_ko = head_ko / head_ko.sum(-1, keepdim=True)
            head_qo = head_qo / head_qo.sum(-1, keepdim=True)

        if self.normalize_attn_scores:
            assert False, "`normalize_attn_scores` not supported."
            # denominator_acc = torch.cumsum(head_k, dim=2)

        if mems is None:
            # lstm states
            state0 = torch.zeros(
                bsz, self.n_head, 1, self.d_head, device=head_ki.device)
            cell0 = torch.zeros(
                bsz, self.n_head, 1, self.d_head, device=head_ki.device)
            # input gate
            mem_fast_weights_i = torch.zeros(
                bsz, self.n_head, self.d_head, self.d_head,
                device=head_ki.device)
            mem_rec_fast_weights_i = torch.zeros(
                bsz, self.n_head, self.d_head, self.d_head,
                device=head_ki.device)
            # update term
            mem_fast_weights_u = torch.zeros(
                bsz, self.n_head, self.d_head, self.d_head,
                device=head_ki.device)
            mem_rec_fast_weights_u = torch.zeros(
                bsz, self.n_head, self.d_head, self.d_head,
                device=head_ki.device)
            # output gate
            mem_fast_weights_o = torch.zeros(
                bsz, self.n_head, self.d_head, self.d_head,
                device=head_ki.device)
            mem_rec_fast_weights_o = torch.zeros(
                bsz, self.n_head, self.d_head, self.d_head,
                device=head_ki.device)
        else:
            assert carry_over_fast_weight
            (mem_fast_weights_i, mem_fast_weights_u, mem_fast_weights_o,
             mem_rec_fast_weights_i, mem_rec_fast_weights_u,
             mem_rec_fast_weights_o, state0, cell0) = mems
            # mem_fast_weights, fast_denom = mems
            # bsz can be smaller for the last batch
            mem_fast_weights_i = mem_fast_weights_i[:bsz]
            mem_fast_weights_u = mem_fast_weights_u[:bsz]
            mem_fast_weights_o = mem_fast_weights_o[:bsz]

            mem_rec_fast_weights_i = mem_rec_fast_weights_i[:bsz]
            mem_rec_fast_weights_u = mem_rec_fast_weights_u[:bsz]
            mem_rec_fast_weights_o = mem_rec_fast_weights_o[:bsz]

            state0 = state0[:bsz]
            cell0 = cell0[:bsz]

            if self.normalize_attn_scores:
                assert False, "Not implemented yet!"
                denominator_acc = denominator_acc + fast_denom[:bsz]

        if self.normalize_attn_scores:
            assert False, "Not supported."
            # denominator = torch.einsum(
            #     'lbij,lbij->lbi', denominator_acc, head_q).unsqueeze(-1)

        # feed-forward part
        zi = fast_weight_delta(
            head_qi, head_ki, head_vi, head_beta_i, mem_fast_weights_i)
        zu = fast_weight_delta(
            head_qu, head_ku, head_vu, head_beta_u, mem_fast_weights_u)
        zo = fast_weight_delta(
            head_qo, head_ko, head_vo, head_beta_o, mem_fast_weights_o)

        # recurrent part
        layer_out, cell_out = fast_lstm_v2(zi, rec_head_ki, rec_head_vi,
                                           rec_beta_i,
                                           mem_rec_fast_weights_i,
                                           zu, rec_head_ku, rec_head_vu,
                                           rec_beta_u,
                                           mem_rec_fast_weights_u,
                                           zo, rec_head_ko, rec_head_vo,
                                           rec_beta_o,
                                           mem_rec_fast_weights_o,
                                           state0, cell0)

        if carry_over_fast_weight:  # clone state
            # layer_out shape (B, n_head, len, d_head)
            state0_next = layer_out[:, :, -1, :].clone().detach()
            state0_next = state0_next.unsqueeze(2)
            cell0_next = cell_out[:, :, -1, :].clone().detach()
            cell0_next = cell0_next.unsqueeze(2)

        # shape (B, n_head, len, d_head)
        if self.normalize_attn_scores:
            assert False, "Not supported."
            layer_out = self.scale * layer_out / (denominator + self.eps)
        else:
            layer_out = self.scale * layer_out

        layer_out = layer_out.transpose(1, 2)
        layer_out = layer_out.reshape(bsz, slen, self.n_head * self.d_head)
        layer_out = layer_out.transpose(0, 1)

        # expect [qlen, B, n_head * d_head]

        # linear projection
        attn_out = self.o_net(layer_out)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            # residual connection
            output = h + attn_out
        else:
            # residual connection + layer normalization
            output = self.layer_norm(h + attn_out)

        if carry_over_fast_weight:
            # last values of accumulator should be carried over.
            # clone is needed as backward modifies the data of fast weight
            if self.normalize_attn_scores:
                assert False, "Not implemented."
                new_k_acc = denominator_acc[:, :, -1, :].unsqueeze(2).detach()
            else:
                new_k_acc = None
            # new_mem = (mem_fast_weights.clone().detach(), new_k_acc)
            new_mem = (mem_fast_weights_i.clone().detach(),
                       mem_fast_weights_u.clone().detach(),
                       mem_fast_weights_o.clone().detach(),
                       mem_rec_fast_weights_i.clone().detach(),
                       mem_rec_fast_weights_u.clone().detach(),
                       mem_rec_fast_weights_o.clone().detach(),
                       state0_next, cell0_next)
            return output, new_mem

        return output


# Fast LSTM layer with FWM update rule
class CudaFastLSTMv3Layer(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout, dropatt=0,
                 pre_lnorm=False, eps=1e-5, layer_id=None, num_layer=None,
                 skip_attn_normalization=True, use_sum_norm=True):
        super(CudaFastLSTMv3Layer, self).__init__()
        print(f"Using CudaFastLSTMv3Layer {layer_id} -")
        print(f"skip_attn_normalization: {skip_attn_normalization}")
        print(f"use_sum_norm: {use_sum_norm}")

        assert layer_id is not None
        assert num_layer is not None
        self.layer_id = layer_id
        self.num_layer = num_layer

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout

        self.qkv_net = nn.Linear(
            d_model, n_head * (15 * d_head + 2 * 3), bias=False)

        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)

        self.layer_norm = nn.LayerNorm(d_model)

        self.scale = 1 / (d_head ** 0.5)

        self.pre_lnorm = pre_lnorm
        self.normalize_attn_scores = (not skip_attn_normalization)
        self.use_sum_norm = use_sum_norm
        self.eps = eps

    def forward(self, h, attn_mask=None, mems=None,
                carry_over_fast_weight=False):
        # multihead attention
        # shape h: (len, B, n_head * d_head)

        if self.pre_lnorm:
            # layer normalization
            h = self.layer_norm(h)

        slen, bsz, _ = h.size()

        qkv = self.qkv_net(h)
        qkv = qkv.view(slen, bsz, self.n_head, 15 * self.d_head + 6)
        (head_qi, head_ki, head_vi, rec_head_ki, rec_head_vi,
         head_qu, head_ku, head_vu, rec_head_ku, rec_head_vu,
         head_qo, head_ko, head_vo, rec_head_ko, rec_head_vo,
         head_beta_i, rec_beta_i,
         head_beta_u, rec_beta_u,
         head_beta_o, rec_beta_o) = torch.split(
             qkv, (self.d_head,) * 15 + (1,) * 6, -1)

        head_beta_i = torch.sigmoid(head_beta_i)
        rec_beta_i = torch.sigmoid(rec_beta_i)
        head_beta_u = torch.sigmoid(head_beta_u)
        rec_beta_u = torch.sigmoid(rec_beta_u)
        head_beta_o = torch.sigmoid(head_beta_o)
        rec_beta_o = torch.sigmoid(rec_beta_o)

        # Reshape to (B, heads, len, dim)
        # input gate
        head_qi = head_qi.permute(1, 2, 0, 3)
        head_ki = head_ki.permute(1, 2, 0, 3)
        head_vi = head_vi.permute(1, 2, 0, 3)
        head_beta_i = head_beta_i.permute(1, 2, 0, 3)

        rec_head_ki = rec_head_ki.permute(1, 2, 0, 3)
        rec_head_vi = rec_head_vi.permute(1, 2, 0, 3)
        rec_beta_i = rec_beta_i.permute(1, 2, 0, 3)

        # update term
        head_qu = head_qu.permute(1, 2, 0, 3)
        head_ku = head_ku.permute(1, 2, 0, 3)
        head_vu = head_vu.permute(1, 2, 0, 3)
        head_beta_u = head_beta_u.permute(1, 2, 0, 3)

        rec_head_ku = rec_head_ku.permute(1, 2, 0, 3)
        rec_head_vu = rec_head_vu.permute(1, 2, 0, 3)
        rec_beta_u = rec_beta_u.permute(1, 2, 0, 3)

        # output gate
        head_qo = head_qo.permute(1, 2, 0, 3)
        head_ko = head_ko.permute(1, 2, 0, 3)
        head_vo = head_vo.permute(1, 2, 0, 3)
        head_beta_o = head_beta_o.permute(1, 2, 0, 3)

        rec_head_ko = rec_head_ko.permute(1, 2, 0, 3)
        rec_head_vo = rec_head_vo.permute(1, 2, 0, 3)
        rec_beta_o = rec_beta_o.permute(1, 2, 0, 3)

        # TODO add dropout here?
        # transform q and k for the forward part
        head_qi = F.elu(head_qi, 1., False) + 1.
        head_ki = F.elu(head_ki, 1., False) + 1.
        head_qu = F.elu(head_qu, 1., False) + 1.
        head_ku = F.elu(head_ku, 1., False) + 1.
        head_qo = F.elu(head_qo, 1., False) + 1.
        head_ko = F.elu(head_ko, 1., False) + 1.

        # make recurrent key consistent with rec activation
        rec_head_ki = F.softmax(rec_head_ki, dim=-1)
        rec_head_ku = F.softmax(rec_head_ku, dim=-1)
        rec_head_ko = F.softmax(rec_head_ko, dim=-1)
        # this performed better than elu+1
        # rec_head_k = F.elu(rec_head_k, 1., False) + 1.

        # normalize k and q, crucial for stable training.
        if self.use_sum_norm:
            head_ki = head_ki / head_ki.sum(-1, keepdim=True)
            head_qi = head_qi / head_qi.sum(-1, keepdim=True)
            head_ku = head_ku / head_ku.sum(-1, keepdim=True)
            head_qu = head_qu / head_qu.sum(-1, keepdim=True)
            head_ko = head_ko / head_ko.sum(-1, keepdim=True)
            head_qo = head_qo / head_qo.sum(-1, keepdim=True)

        if self.normalize_attn_scores:
            assert False, "`normalize_attn_scores` not supported."
            # denominator_acc = torch.cumsum(head_k, dim=2)

        if mems is None:
            # lstm states
            state0 = torch.zeros(
                bsz, self.n_head, 1, self.d_head, device=head_ki.device)
            cell0 = torch.zeros(
                bsz, self.n_head, 1, self.d_head, device=head_ki.device)
            # input gate
            mem_fast_weights_i = torch.zeros(
                bsz, self.n_head, self.d_head, self.d_head,
                device=head_ki.device)
            mem_rec_fast_weights_i = torch.zeros(
                bsz, self.n_head, self.d_head, self.d_head,
                device=head_ki.device)
            # update term
            mem_fast_weights_u = torch.zeros(
                bsz, self.n_head, self.d_head, self.d_head,
                device=head_ki.device)
            mem_rec_fast_weights_u = torch.zeros(
                bsz, self.n_head, self.d_head, self.d_head,
                device=head_ki.device)
            # output gate
            mem_fast_weights_o = torch.zeros(
                bsz, self.n_head, self.d_head, self.d_head,
                device=head_ki.device)
            mem_rec_fast_weights_o = torch.zeros(
                bsz, self.n_head, self.d_head, self.d_head,
                device=head_ki.device)
        else:
            assert carry_over_fast_weight
            (mem_fast_weights_i, mem_fast_weights_u, mem_fast_weights_o,
             mem_rec_fast_weights_i, mem_rec_fast_weights_u,
             mem_rec_fast_weights_o, state0, cell0) = mems
            # mem_fast_weights, fast_denom = mems
            # bsz can be smaller for the last batch
            mem_fast_weights_i = mem_fast_weights_i[:bsz]
            mem_fast_weights_u = mem_fast_weights_u[:bsz]
            mem_fast_weights_o = mem_fast_weights_o[:bsz]

            mem_rec_fast_weights_i = mem_rec_fast_weights_i[:bsz]
            mem_rec_fast_weights_u = mem_rec_fast_weights_u[:bsz]
            mem_rec_fast_weights_o = mem_rec_fast_weights_o[:bsz]

            state0 = state0[:bsz]
            cell0 = cell0[:bsz]

            if self.normalize_attn_scores:
                assert False, "Not implemented yet!"
                denominator_acc = denominator_acc + fast_denom[:bsz]

        if self.normalize_attn_scores:
            assert False, "Not supported."
            # denominator = torch.einsum(
            #     'lbij,lbij->lbi', denominator_acc, head_q).unsqueeze(-1)

        # feed-forward part
        zi = fast_weight_delta(
            head_qi, head_ki, head_vi, head_beta_i, mem_fast_weights_i)
        zu = fast_weight_delta(
            head_qu, head_ku, head_vu, head_beta_u, mem_fast_weights_u)
        zo = fast_weight_delta(
            head_qo, head_ko, head_vo, head_beta_o, mem_fast_weights_o)

        # recurrent part
        layer_out, cell_out = fast_lstm_v3(zi, rec_head_ki, rec_head_vi,
                                           rec_beta_i,
                                           mem_rec_fast_weights_i,
                                           zu, rec_head_ku, rec_head_vu,
                                           rec_beta_u,
                                           mem_rec_fast_weights_u,
                                           zo, rec_head_ko, rec_head_vo,
                                           rec_beta_o,
                                           mem_rec_fast_weights_o,
                                           state0, cell0)

        if carry_over_fast_weight:  # clone state
            # layer_out shape (B, n_head, len, d_head)
            state0_next = layer_out[:, :, -1, :].clone().detach()
            state0_next = state0_next.unsqueeze(2)
            cell0_next = cell_out[:, :, -1, :].clone().detach()
            cell0_next = cell0_next.unsqueeze(2)

        # shape (B, n_head, len, d_head)
        if self.normalize_attn_scores:
            assert False, "Not supported."
            layer_out = self.scale * layer_out / (denominator + self.eps)
        else:
            layer_out = self.scale * layer_out

        layer_out = layer_out.transpose(1, 2)
        layer_out = layer_out.reshape(bsz, slen, self.n_head * self.d_head)
        layer_out = layer_out.transpose(0, 1)

        # expect [qlen, B, n_head * d_head]

        # linear projection
        attn_out = self.o_net(layer_out)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            # residual connection
            output = h + attn_out
        else:
            # residual connection + layer normalization
            output = self.layer_norm(h + attn_out)

        if carry_over_fast_weight:
            # last values of accumulator should be carried over.
            # clone is needed as backward modifies the data of fast weight
            if self.normalize_attn_scores:
                assert False, "Not implemented."
                new_k_acc = denominator_acc[:, :, -1, :].unsqueeze(2).detach()
            else:
                new_k_acc = None
            # new_mem = (mem_fast_weights.clone().detach(), new_k_acc)
            new_mem = (mem_fast_weights_i.clone().detach(),
                       mem_fast_weights_u.clone().detach(),
                       mem_fast_weights_o.clone().detach(),
                       mem_rec_fast_weights_i.clone().detach(),
                       mem_rec_fast_weights_u.clone().detach(),
                       mem_rec_fast_weights_o.clone().detach(),
                       state0_next, cell0_next)
            return output, new_mem

        return output


# Fast LSTM layer with FWM update rule
class CudaFastLSTMv4Layer(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout, dropatt=0,
                 pre_lnorm=False, eps=1e-5, layer_id=None, num_layer=None,
                 skip_attn_normalization=True, use_sum_norm=True):
        super(CudaFastLSTMv4Layer, self).__init__()
        print(f"Using CudaFastLSTMv4Layer {layer_id} -")
        print(f"skip_attn_normalization: {skip_attn_normalization}")
        print(f"use_sum_norm: {use_sum_norm}")

        assert layer_id is not None
        assert num_layer is not None
        self.layer_id = layer_id
        self.num_layer = num_layer

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout

        self.qkv_net = nn.Linear(
            d_model, n_head * (15 * d_head + 2 * 3), bias=False)

        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)

        self.layer_norm = nn.LayerNorm(d_model)

        self.scale = 1 / (d_head ** 0.5)

        self.pre_lnorm = pre_lnorm
        self.normalize_attn_scores = (not skip_attn_normalization)
        self.use_sum_norm = use_sum_norm
        self.eps = eps

    def forward(self, h, attn_mask=None, mems=None,
                carry_over_fast_weight=False):
        # multihead attention
        # shape h: (len, B, n_head * d_head)

        if self.pre_lnorm:
            # layer normalization
            h = self.layer_norm(h)

        slen, bsz, _ = h.size()

        qkv = self.qkv_net(h)
        qkv = qkv.view(slen, bsz, self.n_head, 15 * self.d_head + 6)
        (head_qi, head_ki, head_vi, rec_head_ki, rec_head_vi,
         head_qu, head_ku, head_vu, rec_head_ku, rec_head_vu,
         head_qo, head_ko, head_vo, rec_head_ko, rec_head_vo,
         head_beta_i, rec_beta_i,
         head_beta_u, rec_beta_u,
         head_beta_o, rec_beta_o) = torch.split(
             qkv, (self.d_head,) * 15 + (1,) * 6, -1)

        head_beta_i = torch.sigmoid(head_beta_i)
        rec_beta_i = torch.sigmoid(rec_beta_i)
        head_beta_u = torch.sigmoid(head_beta_u)
        rec_beta_u = torch.sigmoid(rec_beta_u)
        head_beta_o = torch.sigmoid(head_beta_o)
        rec_beta_o = torch.sigmoid(rec_beta_o)

        # Reshape to (B, heads, len, dim)
        # input gate
        head_qi = head_qi.permute(1, 2, 0, 3)
        head_ki = head_ki.permute(1, 2, 0, 3)
        head_vi = head_vi.permute(1, 2, 0, 3)
        head_beta_i = head_beta_i.permute(1, 2, 0, 3)

        rec_head_ki = rec_head_ki.permute(1, 2, 0, 3)
        rec_head_vi = rec_head_vi.permute(1, 2, 0, 3)
        rec_beta_i = rec_beta_i.permute(1, 2, 0, 3)

        # update term
        head_qu = head_qu.permute(1, 2, 0, 3)
        head_ku = head_ku.permute(1, 2, 0, 3)
        head_vu = head_vu.permute(1, 2, 0, 3)
        head_beta_u = head_beta_u.permute(1, 2, 0, 3)

        rec_head_ku = rec_head_ku.permute(1, 2, 0, 3)
        rec_head_vu = rec_head_vu.permute(1, 2, 0, 3)
        rec_beta_u = rec_beta_u.permute(1, 2, 0, 3)

        # output gate
        head_qo = head_qo.permute(1, 2, 0, 3)
        head_ko = head_ko.permute(1, 2, 0, 3)
        head_vo = head_vo.permute(1, 2, 0, 3)
        head_beta_o = head_beta_o.permute(1, 2, 0, 3)

        rec_head_ko = rec_head_ko.permute(1, 2, 0, 3)
        rec_head_vo = rec_head_vo.permute(1, 2, 0, 3)
        rec_beta_o = rec_beta_o.permute(1, 2, 0, 3)

        # TODO add dropout here?
        # transform q and k for the forward part
        head_qi = F.elu(head_qi, 1., False) + 1.
        head_ki = F.elu(head_ki, 1., False) + 1.
        head_qu = F.elu(head_qu, 1., False) + 1.
        head_ku = F.elu(head_ku, 1., False) + 1.
        head_qo = F.elu(head_qo, 1., False) + 1.
        head_ko = F.elu(head_ko, 1., False) + 1.

        # make recurrent key consistent with rec activation
        rec_head_ki = F.softmax(rec_head_ki, dim=-1)
        rec_head_ku = F.softmax(rec_head_ku, dim=-1)
        rec_head_ko = F.softmax(rec_head_ko, dim=-1)
        # this performed better than elu+1
        # rec_head_k = F.elu(rec_head_k, 1., False) + 1.

        # normalize k and q, crucial for stable training.
        if self.use_sum_norm:
            head_ki = head_ki / head_ki.sum(-1, keepdim=True)
            head_qi = head_qi / head_qi.sum(-1, keepdim=True)
            head_ku = head_ku / head_ku.sum(-1, keepdim=True)
            head_qu = head_qu / head_qu.sum(-1, keepdim=True)
            head_ko = head_ko / head_ko.sum(-1, keepdim=True)
            head_qo = head_qo / head_qo.sum(-1, keepdim=True)

        if self.normalize_attn_scores:
            assert False, "`normalize_attn_scores` not supported."
            # denominator_acc = torch.cumsum(head_k, dim=2)

        if mems is None:
            # lstm states
            state0 = torch.zeros(
                bsz, self.n_head, 1, self.d_head, device=head_ki.device)
            cell0 = torch.zeros(
                bsz, self.n_head, 1, self.d_head, device=head_ki.device)
            # input gate
            mem_fast_weights_i = torch.zeros(
                bsz, self.n_head, self.d_head, self.d_head,
                device=head_ki.device)
            mem_rec_fast_weights_i = torch.zeros(
                bsz, self.n_head, self.d_head, self.d_head,
                device=head_ki.device)
            # update term
            mem_fast_weights_u = torch.zeros(
                bsz, self.n_head, self.d_head, self.d_head,
                device=head_ki.device)
            mem_rec_fast_weights_u = torch.zeros(
                bsz, self.n_head, self.d_head, self.d_head,
                device=head_ki.device)
            # output gate
            mem_fast_weights_o = torch.zeros(
                bsz, self.n_head, self.d_head, self.d_head,
                device=head_ki.device)
            mem_rec_fast_weights_o = torch.zeros(
                bsz, self.n_head, self.d_head, self.d_head,
                device=head_ki.device)
        else:
            assert carry_over_fast_weight
            (mem_fast_weights_i, mem_fast_weights_u, mem_fast_weights_o,
             mem_rec_fast_weights_i, mem_rec_fast_weights_u,
             mem_rec_fast_weights_o, state0, cell0) = mems
            # mem_fast_weights, fast_denom = mems
            # bsz can be smaller for the last batch
            mem_fast_weights_i = mem_fast_weights_i[:bsz]
            mem_fast_weights_u = mem_fast_weights_u[:bsz]
            mem_fast_weights_o = mem_fast_weights_o[:bsz]

            mem_rec_fast_weights_i = mem_rec_fast_weights_i[:bsz]
            mem_rec_fast_weights_u = mem_rec_fast_weights_u[:bsz]
            mem_rec_fast_weights_o = mem_rec_fast_weights_o[:bsz]

            state0 = state0[:bsz]
            cell0 = cell0[:bsz]

            if self.normalize_attn_scores:
                assert False, "Not implemented yet!"
                denominator_acc = denominator_acc + fast_denom[:bsz]

        if self.normalize_attn_scores:
            assert False, "Not supported."
            # denominator = torch.einsum(
            #     'lbij,lbij->lbi', denominator_acc, head_q).unsqueeze(-1)

        # feed-forward part
        zi = fast_weight_delta(
            head_qi, head_ki, head_vi, head_beta_i, mem_fast_weights_i)
        zu = fast_weight_delta(
            head_qu, head_ku, head_vu, head_beta_u, mem_fast_weights_u)
        zo = fast_weight_delta(
            head_qo, head_ko, head_vo, head_beta_o, mem_fast_weights_o)

        # recurrent part
        layer_out, cell_out = fast_lstm_v4(zi, rec_head_ki, rec_head_vi,
                                           rec_beta_i,
                                           mem_rec_fast_weights_i,
                                           zu, rec_head_ku, rec_head_vu,
                                           rec_beta_u,
                                           mem_rec_fast_weights_u,
                                           zo, rec_head_ko, rec_head_vo,
                                           rec_beta_o,
                                           mem_rec_fast_weights_o,
                                           state0, cell0)

        if carry_over_fast_weight:  # clone state
            # layer_out shape (B, n_head, len, d_head)
            state0_next = layer_out[:, :, -1, :].clone().detach()
            state0_next = state0_next.unsqueeze(2)
            cell0_next = cell_out[:, :, -1, :].clone().detach()
            cell0_next = cell0_next.unsqueeze(2)

        # shape (B, n_head, len, d_head)
        if self.normalize_attn_scores:
            assert False, "Not supported."
            layer_out = self.scale * layer_out / (denominator + self.eps)
        else:
            layer_out = self.scale * layer_out

        layer_out = layer_out.transpose(1, 2)
        layer_out = layer_out.reshape(bsz, slen, self.n_head * self.d_head)
        layer_out = layer_out.transpose(0, 1)

        # expect [qlen, B, n_head * d_head]

        # linear projection
        attn_out = self.o_net(layer_out)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            # residual connection
            output = h + attn_out
        else:
            # residual connection + layer normalization
            output = self.layer_norm(h + attn_out)

        if carry_over_fast_weight:
            # last values of accumulator should be carried over.
            # clone is needed as backward modifies the data of fast weight
            if self.normalize_attn_scores:
                assert False, "Not implemented."
                new_k_acc = denominator_acc[:, :, -1, :].unsqueeze(2).detach()
            else:
                new_k_acc = None
            # new_mem = (mem_fast_weights.clone().detach(), new_k_acc)
            new_mem = (mem_fast_weights_i.clone().detach(),
                       mem_fast_weights_u.clone().detach(),
                       mem_fast_weights_o.clone().detach(),
                       mem_rec_fast_weights_i.clone().detach(),
                       mem_rec_fast_weights_u.clone().detach(),
                       mem_rec_fast_weights_o.clone().detach(),
                       state0_next, cell0_next)
            return output, new_mem

        return output


# Parameterize qkv as an LSTM.
# We did not report final performance of this model in the paper
# as we saw this one clearly does not perform well.
class CudaFastWeightSlowRNNLayer(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout, dropatt=0,
                 pre_lnorm=False, eps=1e-5, layer_id=None, num_layer=None,
                 skip_attn_normalization=True):
        # skip_attn_normalization is now set to True by default, thus it can
        # be removed.
        # Originally, with skip_attn_normalization set to False,
        # we had a version of the model which applies attention normalization
        # to the output (but not when we retrieve with the key for removal).
        super(CudaFastWeightSlowRNNLayer, self).__init__()
        print(f"Using CudaFastWeightSlowRNNLayer {layer_id} -")

        assert layer_id is not None
        assert num_layer is not None
        self.layer_id = layer_id
        self.num_layer = num_layer

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout

        slow_net_out_dim = n_head * (3 * d_head + 1)
        self.slow_net = nn.LSTM(
            input_size=d_model, hidden_size=slow_net_out_dim, num_layers=1)

        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)

        self.layer_norm = nn.LayerNorm(d_model)

        self.scale = 1 / (d_head ** 0.5)

        self.pre_lnorm = pre_lnorm
        self.normalize_attn_scores = (not skip_attn_normalization)
        self.eps = eps

    def forward(self, h, attn_mask=None, mems=None,
                carry_over_fast_weight=False):
        # multihead attention
        # shape h: (len, B, n_head * d_head)

        if self.pre_lnorm:
            # layer normalization
            h = self.layer_norm(h)

        slen, bsz, _ = h.size()

        self.slow_net.flatten_parameters()  # Not needed in recent PyT version?
        if mems is None:
            mem_fast_weights = torch.zeros(
                bsz, self.n_head, self.d_head, self.d_head,
                device=h.device)
            # If (h_0, c_0) is not provided, both h_0 and c_0 default to zero.
            qkvb, (state0_next, cell0_next) = self.slow_net(h)
        else:
            assert carry_over_fast_weight
            mem_fast_weights, fast_denom, state0, cell0 = mems
            # bsz can be smaller for the last batch
            mem_fast_weights = mem_fast_weights[:bsz]
            state0 = state0[:bsz]
            cell0 = cell0[:bsz]
            state_tuple = (state0, cell0)
            qkvb, (state0_next, cell0_next) = self.slow_net(h, state_tuple)

        if carry_over_fast_weight:
            state0_next = state0_next[-1].clone().detach().unsqueeze(0)
            cell0_next = cell0_next[-1].clone().detach().unsqueeze(0)

        qkvb = qkvb.view(slen, bsz, self.n_head, 3 * self.d_head + 1)
        head_q, head_k, head_v, head_beta = torch.split(
            qkvb, (self.d_head,) * 3 + (1,), -1)
        head_beta = torch.sigmoid(head_beta)

        # reshape to (B, heads, len, dim)
        head_q = head_q.permute(1, 2, 0, 3)
        head_k = head_k.permute(1, 2, 0, 3)
        head_v = head_v.permute(1, 2, 0, 3)
        head_beta = head_beta.permute(1, 2, 0, 3)

        # TODO add dropout here?
        # transform q and k
        head_q = elu_p1_sum_norm(head_q)
        head_k = elu_p1_sum_norm(head_k)

        if self.normalize_attn_scores:
            # another version would be:
            # head_k_beta = head_k * head_beta
            # denominator_acc = torch.cumsum(head_k_beta, dim=2)
            denominator_acc = torch.cumsum(head_k, dim=2)
            if carry_over_fast_weight:
                denominator_acc = denominator_acc + fast_denom[:bsz]

        if self.normalize_attn_scores:
            denominator = torch.einsum(
                'lbij,lbij->lbi', denominator_acc, head_q).unsqueeze(-1)

        layer_out = fast_weight_delta(
            head_q, head_k, head_v, head_beta, mem_fast_weights)

        # shape (B, n_head, len, d_head)
        if self.normalize_attn_scores:
            layer_out = self.scale * layer_out / (denominator + self.eps)
        else:
            layer_out = self.scale * layer_out

        layer_out = layer_out.transpose(1, 2)
        layer_out = layer_out.reshape(bsz, slen, self.n_head * self.d_head)
        layer_out = layer_out.transpose(0, 1)

        # expect [qlen, B, n_head * d_head]

        # linear projection
        attn_out = self.o_net(layer_out)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            # residual connection
            output = h + attn_out
        else:
            # residual connection + layer normalization
            output = self.layer_norm(h + attn_out)

        if carry_over_fast_weight:
            # last values of accumulator should be carried over.
            # clone is needed as backward modifies the data of fast weight
            if self.normalize_attn_scores:
                new_k_acc = denominator_acc[:, :, -1, :].unsqueeze(2).detach()
            else:
                new_k_acc = None

            new_mem = (mem_fast_weights.clone().detach(), new_k_acc,
                       state0_next, cell0_next)
            return output, new_mem

        return output


class CudaFastWeightRecUpdateTanhLayer(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout, dropatt=0,
                 pre_lnorm=False, eps=1e-5, layer_id=None, num_layer=None,
                 skip_attn_normalization=True):
        # skip_attn_normalization is now set to True by default, thus it can
        # be removed.
        # Originally, with skip_attn_normalization set to False,
        # we had a version of the model which applies attention normalization
        # to the output (but not when we retrieve with the key for removal).
        super(CudaFastWeightRecUpdateTanhLayer, self).__init__()
        print(f"Using CudaFastWeightRecUpdateTanhLayer {layer_id} -")

        assert layer_id is not None
        assert num_layer is not None
        self.layer_id = layer_id
        self.num_layer = num_layer

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout

        # (3 * d_head * n_head) for qkv and (1 * n_head) for beta.
        self.slow_net = nn.Linear(
            d_model, n_head * (3 * d_head + 1), bias=False)

        self.R_q = nn.Parameter(torch.Tensor(1, n_head, d_head, d_head),
                                requires_grad=True)
        self.R_k = nn.Parameter(torch.Tensor(1, n_head, d_head, d_head),
                                requires_grad=True)
        self.R_v = nn.Parameter(torch.Tensor(1, n_head, d_head, d_head),
                                requires_grad=True)
        self.r_b = nn.Parameter(torch.Tensor(1, n_head, 1, d_head),
                                requires_grad=True)

        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)

        self.layer_norm = nn.LayerNorm(d_model)

        self.scale = 1 / (d_head ** 0.5)

        self.pre_lnorm = pre_lnorm
        self.normalize_attn_scores = (not skip_attn_normalization)
        self.eps = eps

        self.reset_rec_parameters()

    def reset_rec_parameters(self):
        std = 1.0 / math.sqrt(self.d_head)
        nn.init.normal_(self.R_q, mean=0., std=std)
        nn.init.normal_(self.R_k, mean=0., std=std)
        nn.init.normal_(self.R_v, mean=0., std=std)
        nn.init.normal_(self.r_b, mean=0., std=std)

    def forward(self, h, attn_mask=None, mems=None,
                carry_over_fast_weight=False):
        # multihead attention
        # shape h: (len, B, n_head * d_head)

        if self.pre_lnorm:
            # layer normalization
            h = self.layer_norm(h)

        slen, bsz, _ = h.size()

        qkvb = self.slow_net(h)
        qkvb = qkvb.view(slen, bsz, self.n_head, 3 * self.d_head + 1)
        head_q, head_k, head_v, head_beta = torch.split(
            qkvb, (self.d_head,) * 3 + (1,), -1)
        # sigmoid on beta is applied inside the kernel.
        # elu+1 and sum norm is replaced by softmax inside the kernel

        # reshape to (B, heads, len, dim)
        head_q = head_q.permute(1, 2, 0, 3)
        head_k = head_k.permute(1, 2, 0, 3)
        head_v = head_v.permute(1, 2, 0, 3)
        head_beta = head_beta.permute(1, 2, 0, 3)

        # TODO add dropout here?

        if self.normalize_attn_scores:
            # another version would be:
            # head_k_beta = head_k * head_beta
            denominator_acc = torch.cumsum(head_k, dim=2)

        if mems is None:
            mem_fast_weights = torch.zeros(
                bsz, self.n_head, self.d_head, self.d_head,
                device=head_k.device)
            state0 = torch.zeros(
                bsz, self.n_head, 1, self.d_head, device=head_k.device)
        else:
            assert carry_over_fast_weight
            mem_fast_weights, fast_denom, state0 = mems
            # bsz can be smaller for the last batch
            mem_fast_weights = mem_fast_weights[:bsz]
            if self.normalize_attn_scores:
                denominator_acc = denominator_acc + fast_denom[:bsz]

        if self.normalize_attn_scores:
            denominator = torch.einsum(
                'lbij,lbij->lbi', denominator_acc, head_q).unsqueeze(-1)

        layer_out = rec_update_fwm_tanh(
            head_q, head_k, head_v, head_beta,
            self.R_q, self.R_k, self.R_v, self.r_b,
            mem_fast_weights, state0)

        if carry_over_fast_weight:  # clone state
            # layer_out shape (B, n_head, len, d_head)
            state0_next = layer_out[:, :, -1, :].clone().detach()
            state0_next = state0_next.unsqueeze(2)

        # shape (B, n_head, len, d_head)
        if self.normalize_attn_scores:
            layer_out = self.scale * layer_out / (denominator + self.eps)
        else:
            layer_out = self.scale * layer_out

        layer_out = layer_out.transpose(1, 2)

        layer_out = layer_out.reshape(
            bsz, slen, self.n_head * self.d_head)

        layer_out = layer_out.transpose(0, 1)

        # expect [qlen, B, n_head * d_head]

        # linear projection
        attn_out = self.o_net(layer_out)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            # residual connection
            output = h + attn_out
        else:
            # residual connection + layer normalization
            output = self.layer_norm(h + attn_out)

        if carry_over_fast_weight:
            # last values of accumulator should be carried over.
            # clone is needed as backward modifies the data of fast weight
            if self.normalize_attn_scores:
                new_k_acc = denominator_acc[:, :, -1, :].unsqueeze(2).detach()
            else:
                new_k_acc = None
            new_mem = (mem_fast_weights.clone().detach(), new_k_acc,
                       state0_next)
            return output, new_mem

        return output
