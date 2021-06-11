import math

import torch
from torch import nn
from torch.nn import functional as F

from torchbeast.fast_weight import fast_weight_memory
from torchbeast.fast_transformers import fast_weight_sum
from torchbeast.rec_update_fwm_tanh import rec_update_fwm_tanh
from torchbeast.fast_weight_rnn_v2 import fast_rnn_v2


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
def elu_p1_sum_norm_eps(x):
    y = F.elu(x, 1., False) + 1.
    return y / (y.sum(-1, keepdim=True) + 1e-5)


# A block of residual feed-forward layers in Transformer
class TransformerFFlayers(nn.Module):
    def __init__(self, ff_dim, res_dim, dropout, use_layernorm=True):
        super(TransformerFFlayers, self).__init__()

        self.res_dim = res_dim
        self.ff_dim = ff_dim
        self.dropout = dropout
        self.use_layernorm = use_layernorm

        self.ff_layers = nn.Sequential(
            nn.Linear(res_dim, ff_dim), nn.ReLU(inplace=False),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, res_dim),
            nn.Dropout(dropout),
        )

        if use_layernorm:
            self.layer_norm = nn.LayerNorm(res_dim)

    def forward(self, x):
        out = self.layer_norm(x) if self.use_layernorm else x
        out = self.ff_layers(out) + x
        return out


# Fast weight layer with feed-forward fast net
# linear tranformer
class AdditiveFastFFlayer(nn.Module):
    def __init__(self, num_head, dim_head, in_dim, dropout):
        super(AdditiveFastFFlayer, self).__init__()

        self.num_head = num_head
        self.dim_head = dim_head
        self.in_dim = in_dim

        self.fw_layer = fast_weight_sum

        self.slow_net = nn.Linear(
            in_dim, num_head * (3 * dim_head), bias=False)

        self.layer_norm = nn.LayerNorm(in_dim)
        self.out_linear = nn.Linear(num_head * dim_head, in_dim, bias=False)

        self.drop = nn.Dropout(dropout)

    def forward(self, x, state=None):
        # x shape: (len, B, n_head * d_head)
        slen, bsz, _ = x.size()

        out = self.layer_norm(x)

        qkv = self.slow_net(out)
        qkv = qkv.view(slen, bsz, self.num_head, 3 * self.dim_head)
        head_q, head_k, head_v = torch.split(qkv, (self.dim_head,) * 3, -1)

        # reshape to (B, heads, len, dim)
        head_q = head_q.permute(1, 2, 0, 3)
        head_k = head_k.permute(1, 2, 0, 3)
        head_v = head_v.permute(1, 2, 0, 3)

        head_q = elu_p1_sum_norm_eps(head_q)
        head_k = elu_p1_sum_norm_eps(head_k)

        if state is not None:
            fast_weights = state
        else:
            assert False
            fast_weights = torch.zeros(
                bsz, self.num_head, self.dim_head, self.dim_head,
                device=head_k.device)
        assert torch.isnan(
            fast_weights).sum().item() == 0, f"Before NaN: fast weights"

        out = self.fw_layer(head_q, head_k, head_v, fast_weights)

        assert torch.isnan(
            fast_weights).sum().item() == 0, f"NaN: fast weights"
        out = out.transpose(1, 2)
        out = out.reshape(bsz, slen, self.num_head * self.dim_head)
        out = out.transpose(0, 1)
        # expect [qlen, B, n_head * d_head]

        # linear projection
        out = self.out_linear(out)
        out = self.drop(out)
        out = x + out

        return out, fast_weights.clone()

# Fast weight layer with feed-forward fast net
class FastFFlayer(nn.Module):
    def __init__(self, num_head, dim_head, in_dim, dropout):
        super(FastFFlayer, self).__init__()

        self.num_head = num_head
        self.dim_head = dim_head
        self.in_dim = in_dim

        self.fw_layer = fast_weight_memory

        self.slow_net = nn.Linear(
            in_dim, num_head * (3 * dim_head + 1), bias=False)

        self.layer_norm = nn.LayerNorm(in_dim)
        self.out_linear = nn.Linear(num_head * dim_head, in_dim, bias=False)

        self.drop = nn.Dropout(dropout)

    def forward(self, x, state=None):
        # x shape: (len, B, n_head * d_head)
        slen, bsz, _ = x.size()

        out = self.layer_norm(x)

        qkvb = self.slow_net(out)
        qkvb = qkvb.view(slen, bsz, self.num_head, 3 * self.dim_head + 1)
        head_q, head_k, head_v, head_beta = torch.split(
            qkvb, (self.dim_head,) * 3 + (1,), -1)
        head_beta = torch.sigmoid(head_beta)

        # reshape to (B, heads, len, dim)
        head_q = head_q.permute(1, 2, 0, 3)
        head_k = head_k.permute(1, 2, 0, 3)
        head_v = head_v.permute(1, 2, 0, 3)
        head_beta = head_beta.permute(1, 2, 0, 3)

        head_q = elu_p1_sum_norm_eps(head_q)
        head_k = elu_p1_sum_norm_eps(head_k)

        if state is not None:
            fast_weights = state
        else:
            assert False
            fast_weights = torch.zeros(
                bsz, self.num_head, self.dim_head, self.dim_head,
                device=head_k.device)
        assert torch.isnan(
            fast_weights).sum().item() == 0, f"Before NaN: fast weights"
        out = self.fw_layer(head_q, head_k, head_v, head_beta, fast_weights)
        assert torch.isnan(
            fast_weights).sum().item() == 0, f"NaN: fast weights"
        out = out.transpose(1, 2)
        out = out.reshape(bsz, slen, self.num_head * self.dim_head)
        out = out.transpose(0, 1)
        # expect [qlen, B, n_head * d_head]

        # linear projection
        out = self.out_linear(out)
        out = self.drop(out)
        out = x + out

        return out, fast_weights.clone()


# Fast weight layer with RNN fast net
class FastRNNlayer(nn.Module):
    def __init__(self, num_head, dim_head, in_dim, dropout):
        super(FastRNNlayer, self).__init__()

        self.num_head = num_head
        self.dim_head = dim_head
        self.in_dim = in_dim
        self.fw_layer = fast_weight_memory
        self.rec_fw_layer = fast_rnn_v2

        self.slow_net = nn.Linear(
            in_dim, num_head * (5 * dim_head + 2), bias=False)

        self.layer_norm = nn.LayerNorm(in_dim)
        self.out_linear = nn.Linear(num_head * dim_head, in_dim, bias=False)

        self.drop = nn.Dropout(dropout)

    def forward(self, x, state=None):
        # x shape: (len, B, n_head * d_head)
        slen, bsz, _ = x.size()

        out = self.layer_norm(x)

        qkvb = self.slow_net(out)
        qkvb = qkvb.view(slen, bsz, self.num_head, 5 * self.dim_head + 2)
        (head_q, head_k, head_v, rec_head_k, rec_head_v, head_beta,
         rec_beta) = torch.split(qkvb, (self.dim_head,) * 5 + (1,) * 2, -1)

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

        head_q = F.softmax(head_q, dim=-1)
        head_k = F.softmax(head_k, dim=-1)
        # make recurrent key consistent with rec activation
        rec_head_k = F.softmax(rec_head_k, dim=-1)

        # # normalize k and q, crucial for stable training.
        # head_k = sum_norm(head_k)
        # head_q = sum_norm(head_q)

        if state is None:
            fast_weights = torch.zeros(
                bsz, self.num_head, self.dim_head, self.dim_head,
                device=head_k.device)

            rec_fast_weights = torch.zeros(
                bsz, self.num_head, self.dim_head, self.dim_head,
                device=head_k.device)

            state0 = torch.zeros(
                bsz, self.num_head, 1, self.dim_head, device=head_k.device)
        else:
            fast_weights, rec_fast_weights, state0 = state
        assert torch.isnan(
            fast_weights).sum().item() == 0, f"Before NaN: fast weights"
        z_out = self.fw_layer(
            head_q, head_k, head_v, head_beta, fast_weights)

        out = self.rec_fw_layer(
            z_out, rec_head_k, rec_head_v, rec_fast_weights, rec_beta, state0)

        state0_next = out[:, :, -1, :].clone()
        state0_next = state0_next.unsqueeze(2)

        out = out.transpose(1, 2)
        out = out.reshape(bsz, slen, self.num_head * self.dim_head)
        out = out.transpose(0, 1)
        # expect [qlen, B, n_head * d_head]

        # linear projection
        out = self.out_linear(out)
        out = self.drop(out)
        out = x + out

        return out, (
            fast_weights.clone(), rec_fast_weights.clone(), state0_next)


class RecUpdateTanhFastFFlayer(nn.Module):
    def __init__(self, num_head, dim_head, in_dim, dropout):
        super(RecUpdateTanhFastFFlayer, self).__init__()

        self.num_head = num_head
        self.dim_head = dim_head
        self.in_dim = in_dim

        self.fw_layer = rec_update_fwm_tanh

        self.slow_net = nn.Linear(
            in_dim, num_head * (3 * dim_head + 1), bias=False)

        self.R_q = nn.Parameter(torch.Tensor(1, num_head, dim_head, dim_head),
                                requires_grad=True)
        self.R_k = nn.Parameter(torch.Tensor(1, num_head, dim_head, dim_head),
                                requires_grad=True)
        self.R_v = nn.Parameter(torch.Tensor(1, num_head, dim_head, dim_head),
                                requires_grad=True)
        self.r_b = nn.Parameter(torch.Tensor(1, num_head, 1, dim_head),
                                requires_grad=True)

        self.layer_norm = nn.LayerNorm(in_dim)
        self.out_linear = nn.Linear(num_head * dim_head, in_dim, bias=False)

        self.drop = nn.Dropout(dropout)

        self.reset_rec_parameters()

    def reset_rec_parameters(self):
        std = 1.0 / math.sqrt(self.dim_head)
        nn.init.normal_(self.R_q, mean=0., std=std)
        nn.init.normal_(self.R_k, mean=0., std=std)
        nn.init.normal_(self.R_v, mean=0., std=std)
        nn.init.normal_(self.r_b, mean=0., std=std)

    def forward(self, x, state=None):
        # x shape: (len, B, n_head * d_head)
        slen, bsz, _ = x.size()

        out = self.layer_norm(x)

        qkvb = self.slow_net(out)
        qkvb = qkvb.view(slen, bsz, self.num_head, 3 * self.dim_head + 1)
        head_q, head_k, head_v, head_beta = torch.split(
            qkvb, (self.dim_head,) * 3 + (1,), -1)

        # reshape to (B, heads, len, dim)
        head_q = head_q.permute(1, 2, 0, 3)
        head_k = head_k.permute(1, 2, 0, 3)
        head_v = head_v.permute(1, 2, 0, 3)
        head_beta = head_beta.permute(1, 2, 0, 3)

        if state is None:
            fast_weights = torch.zeros(
                bsz, self.num_head, self.dim_head, self.dim_head,
                device=head_k.device)

            state0 = torch.zeros(
                bsz, self.num_head, 1, self.dim_head, device=head_k.device)
        else:
            fast_weights, state0 = state

        out = self.fw_layer(head_q, head_k, head_v, head_beta,
                            self.R_q, self.R_k, self.R_v, self.r_b,
                            fast_weights, state0)

        state0_next = out[:, :, -1, :].clone()
        state0_next = state0_next.unsqueeze(2)

        out = out.transpose(1, 2)
        out = out.reshape(bsz, slen, self.num_head * self.dim_head)
        out = out.transpose(0, 1)
        # expect [qlen, B, n_head * d_head]

        # linear projection
        out = self.out_linear(out)
        out = self.drop(out)
        out = x + out

        return out, (fast_weights.clone(), state0_next)


# Linear Transformer with Fast weight memory update rule.
class LinearTransformerLayer(nn.Module):
    def __init__(self, input_dim, hidden_size, num_layers, num_head, dim_head,
                 dim_ff, dropout):
        super(LinearTransformerLayer, self).__init__()
        assert num_head * dim_head == hidden_size

        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_head = num_head
        self.dim_head = dim_head
        self.dim_ff = dim_ff
        self.dropout = dropout

        self.input_proj = nn.Linear(input_dim, hidden_size)

        fwm_layers = []
        ff_layers = []

        for _ in range(num_layers):  # each "layer" consists of two sub-layers
            fwm_layers.append(
                AdditiveFastFFlayer(num_head, dim_head, hidden_size, dropout))
            ff_layers.append(
                TransformerFFlayers(dim_ff, hidden_size, dropout))

        self.fwm_layers = nn.ModuleList(fwm_layers)
        self.ff_layers = nn.ModuleList(ff_layers)

    def forward(self, x, core_state):
        # core_state is a tuple with self.num_layers elements
        state_list = []

        out = self.input_proj(x)  # shape (len, B, dim)
        for i in range(self.num_layers):
            out, out_state = self.fwm_layers[i](out, state=core_state[i].squeeze(0))
            state_list.append(out_state.unsqueeze(0).clone())
            out = self.ff_layers[i](out)

        state_tuple = tuple(state_list)
        return out, state_tuple


# Linear Transformer with Fast weight memory update rule.
class DeltaNetLayer(nn.Module):
    def __init__(self, input_dim, hidden_size, num_layers, num_head, dim_head,
                 dim_ff, dropout):
        super(DeltaNetLayer, self).__init__()
        assert num_head * dim_head == hidden_size

        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_head = num_head
        self.dim_head = dim_head
        self.dim_ff = dim_ff
        self.dropout = dropout

        self.input_proj = nn.Linear(input_dim, hidden_size)

        fwm_layers = []
        ff_layers = []

        for _ in range(num_layers):  # each "layer" consists of two sub-layers
            fwm_layers.append(
                FastFFlayer(num_head, dim_head, hidden_size, dropout))
            ff_layers.append(
                TransformerFFlayers(dim_ff, hidden_size, dropout))

        self.fwm_layers = nn.ModuleList(fwm_layers)
        self.ff_layers = nn.ModuleList(ff_layers)

    def forward(self, x, core_state):
        # core_state is a tuple with self.num_layers elements
        state_list = []

        out = self.input_proj(x)  # shape (len, B, dim)

        for i in range(self.num_layers):
            out, out_state = self.fwm_layers[i](out, state=core_state[i].squeeze(0))
            state_list.append(out_state.unsqueeze(0).clone())
            out = self.ff_layers[i](out)

        state_tuple = tuple(state_list)
        return out, state_tuple


class FastFFRecUpdateTanhLayer(nn.Module):
    def __init__(self, input_dim, hidden_size, num_layers, num_head, dim_head,
                 dim_ff, dropout):
        super(FastFFRecUpdateTanhLayer, self).__init__()
        assert num_head * dim_head == hidden_size

        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_head = num_head
        self.dim_head = dim_head
        self.dim_ff = dim_ff
        self.dropout = dropout

        self.input_proj = nn.Linear(input_dim, hidden_size)

        fwm_layers = []
        ff_layers = []

        for _ in range(num_layers):  # each "layer" consists of two sub-layers
            fwm_layers.append(
                RecUpdateTanhFastFFlayer(
                    num_head, dim_head, hidden_size, dropout))
            ff_layers.append(
                TransformerFFlayers(dim_ff, hidden_size, dropout))

        self.fwm_layers = nn.ModuleList(fwm_layers)
        self.ff_layers = nn.ModuleList(ff_layers)

    def forward(self, x, core_state):
        fw_states, rnn_states = core_state
        # core_state is a tuple with self.num_layers elements
        fw_state_list = []
        rnn_state_list = []

        out = self.input_proj(x)  # shape (len, B, dim)

        for i in range(self.num_layers):
            out, out_state = self.fwm_layers[i](
                out, state=(fw_states[i].squeeze(0), rnn_states[i].squeeze(0)))
            fw_state_list.append(out_state[0].unsqueeze(0).clone())
            rnn_state_list.append(out_state[1].unsqueeze(0).clone())
            out = self.ff_layers[i](out)

        fw_state_tuple = tuple(fw_state_list)
        rnn_state_tuple = tuple(rnn_state_list)
        state_tuple = (fw_state_tuple, rnn_state_tuple)
        return out, state_tuple


class FastRNNModelLayer(nn.Module):
    def __init__(self, input_dim, hidden_size, num_layers, num_head, dim_head,
                 dim_ff, dropout):
        super(FastRNNModelLayer, self).__init__()
        assert num_head * dim_head == hidden_size

        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_head = num_head
        self.dim_head = dim_head
        self.dim_ff = dim_ff
        self.dropout = dropout

        self.input_proj = nn.Linear(input_dim, hidden_size)

        fwm_layers = []
        ff_layers = []

        for _ in range(num_layers):  # each "layer" consists of two sub-layers
            fwm_layers.append(
                FastRNNlayer(num_head, dim_head, hidden_size, dropout))
            ff_layers.append(
                TransformerFFlayers(dim_ff, hidden_size, dropout))

        self.fwm_layers = nn.ModuleList(fwm_layers)
        self.ff_layers = nn.ModuleList(ff_layers)

    def forward(self, x, core_state):
        fw_states, rec_fw_states, rnn_states = core_state
        # core_state is a tuple with self.num_layers elements
        fw_state_list = []
        rec_fw_state_list = []
        rnn_state_list = []

        out = self.input_proj(x)  # shape (len, B, dim)

        for i in range(self.num_layers):
            out, out_state = self.fwm_layers[i](
                out,
                state=(fw_states[i].squeeze(0), rec_fw_states[i].squeeze(0),
                       rnn_states[i].squeeze(0)))
            fw_state_list.append(out_state[0].unsqueeze(0).clone())
            rec_fw_state_list.append(out_state[1].unsqueeze(0).clone())
            rnn_state_list.append(out_state[2].unsqueeze(0).clone())
            out = self.ff_layers[i](out)

        fw_state_tuple = tuple(fw_state_list)
        rec_fw_state_tuple = tuple(rec_fw_state_list)
        rnn_state_tuple = tuple(rnn_state_list)
        state_tuple = (fw_state_tuple, rec_fw_state_tuple, rnn_state_tuple)
        return out, state_tuple
