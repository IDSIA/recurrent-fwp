# Include code for model components such as
# - positional encoding,
# - Transformer feed-forward block
# - ...
#
# Many lines originally from https://github.com/kimiyoung/transformer-xl/blob/master/pytorch/mem_transformer.py

import sys

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEmbedding(nn.Module):
    def __init__(self, demb):
        super(PositionalEmbedding, self).__init__()

        self.demb = demb

        inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos_seq, bsz=None):
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)

        if bsz is not None:
            return pos_emb[:,None,:].expand(-1, bsz, -1)
        else:
            return pos_emb[:,None,:]


class PositionwiseFF(nn.Module):
    def __init__(self, d_model, d_inner, dropout, pre_lnorm=False):
        super(PositionwiseFF, self).__init__()

        self.d_model = d_model
        self.d_inner = d_inner
        self.dropout = dropout

        self.CoreNet = nn.Sequential(
            nn.Linear(d_model, d_inner), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_inner, d_model),
            nn.Dropout(dropout),
        )

        self.layer_norm = nn.LayerNorm(d_model)

        self.pre_lnorm = pre_lnorm

    def forward(self, inp):
        if self.pre_lnorm:
            # layer normalization + positionwise feed-forward
            core_out = self.CoreNet(self.layer_norm(inp))

            # residual connection
            output = core_out + inp
        else:
            # positionwise feed-forward
            core_out = self.CoreNet(inp)

            # residual connection + layer normalization
            output = self.layer_norm(inp + core_out)

        return output


# Standard PyTorch LSTM layer.
# input dimension is the dim_res
# recurrent state can be of arbitrary size
# output projection put the dim back to dim_res for res connection.
class LSTMLayer(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout, dropatt=0,
                 pre_lnorm=False, eps=1e-5, layer_id=None, num_layer=None,
                 use_lnorm=True, use_res=True, d_res=None, use_out_proj=True,
                 skip_attn_normalization=False, use_sum_norm=True):
        super(LSTMLayer, self).__init__()
        print(f"Using LSTMLayer {layer_id} -")
        print("skip_attn_normalization and use_sum_norm are not used")
        print(f"use_sum_norm: {use_sum_norm}")

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

        self.lstm_func = nn.LSTM(input_size=d_res,
                                 hidden_size=d_model,
                                 num_layers=1,
                                 bias=True,  # default
                                 batch_first=False,  # default
                                 dropout=dropout,
                                 bidirectional=False)
        if use_out_proj:
            self.out_proj = nn.Linear(d_model, d_res, bias=False)

        self.drop = nn.Dropout(dropout)

        if use_lnorm:
            self.layer_norm = nn.LayerNorm(d_res)
        self.pre_lnorm = pre_lnorm

    def forward(self, x, attn_mask=None, mems=None,
                carry_over_fast_weight=False):
        # multihead attention
        # [hlen x bsz x n_head x d_head]
        _, bsz, _ = x.size()

        if mems is not None:
            assert carry_over_fast_weight
            state0, cell0 = mems
            state0 = state0.clone().detach()
            cell0 = cell0.clone().detach()
        else:
            state0 = torch.zeros(1, bsz, self.d_model, device=x.device)
            cell0 = torch.zeros(1, bsz, self.d_model, device=x.device)

        if self.use_lnorm and self.pre_lnorm:
            # layer normalization
            x = self.layer_norm(x)

        self.lstm_func.flatten_parameters()
        attn_out, state_tuple = self.lstm_func(x, (state0, cell0))

        # linear projection
        if self.use_out_proj:
            attn_out = self.drop(attn_out)
            attn_out = self.out_proj(attn_out)
        attn_out = self.drop(attn_out)

        if self.use_res:
            if self.use_lnorm:
                if self.pre_lnorm:
                    # residual connection
                    output = x + attn_out
                else:
                    # residual connection + layer normalization
                    output = self.layer_norm(x + attn_out)
            else:
                output = x + attn_out
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
            return output, state_tuple
        else:
            return output


# Standard PyTorch RNN layer.
class RNNLayer(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout, dropatt=0,
                 pre_lnorm=False, eps=1e-5, layer_id=None, num_layer=None,
                 use_lnorm=True, use_res=True, d_res=None, use_out_proj=True,
                 skip_attn_normalization=False, use_sum_norm=True):
        super(RNNLayer, self).__init__()
        print(f"Using RNNLayer {layer_id} -")
        print("skip_attn_normalization and use_sum_norm are not used")
        print(f"use_sum_norm: {use_sum_norm}")

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

        self.rnn_func = nn.RNN(input_size=d_res,
                               hidden_size=d_model,
                               num_layers=1,
                               nonlinearity='tanh',  # default
                               bias=True,  # default
                               batch_first=False,  # default
                               dropout=dropout,
                               bidirectional=False)
        if use_out_proj:
            self.out_proj = nn.Linear(d_model, d_res, bias=False)
        self.drop = nn.Dropout(dropout)

        if use_lnorm:
            self.layer_norm = nn.LayerNorm(d_res)
        self.pre_lnorm = pre_lnorm

    def forward(self, x, attn_mask=None, mems=None,
                carry_over_fast_weight=False):
        # multihead attention
        # [hlen x bsz x n_head x d_head]
        _, bsz, _ = x.size()

        if mems is not None:
            assert carry_over_fast_weight
            state0 = mems
            state0 = state0.clone().detach()
        else:
            state0 = torch.zeros(1, bsz, self.d_model, device=x.device)

        if self.use_lnorm and self.pre_lnorm:
            # layer normalization
            x = self.layer_norm(x)

        self.rnn_func.flatten_parameters()
        attn_out, state_next = self.rnn_func(x, state0)

        # linear projection
        if self.use_out_proj:
            attn_out = self.drop(attn_out)
            attn_out = self.out_proj(attn_out)
        attn_out = self.drop(attn_out)

        if self.use_res:
            if self.use_lnorm:
                if self.pre_lnorm:
                    # residual connection
                    output = x + attn_out
                else:
                    # residual connection + layer normalization
                    output = self.layer_norm(x + attn_out)
            else:
                output = x + attn_out
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
            return output, state_next
        else:
            return output
