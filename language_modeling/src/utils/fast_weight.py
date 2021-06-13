# Copyright (c) 2021 Kazuki Irie
# Custom ops for memory efficient fast weight implementations.
# All ops require `fast_weight` and `grad_fast_weight` as global variables
# (no another choice for PyTorch custom autograd).
# Only useful for prototyping. Otherwise the cuda implementations should be
# used instead.
# Many code duplications to be refactored!

import torch
import torch.nn as nn
import torch.nn.functional as F

# Initialize global variables with empty list.
# This is not nice, but we need them for memory efficient custom backprop
# for fast weight ops.
# The length of these lists will become as long as the number of layers
# times number of GPUs used.
# Each GPU will make use of the variable at the corresponding layer
# index.

fast_weight = []
grad_fast_weight = []


# Step-by-step fast weight memory with sum update rule, mainly for debugging.
# In general, use the cuda implementation instead.
class FastWeightSumLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, k, v, q, layer_id):
        # key, query: (B, dim)
        # value: (B, v_dim)
        # layer id: layer_id
        global fast_weight
        # fast_weight is a global variable. This is needed as we can not have
        # extra, non-gradient parameter in the backward pass.
        ctx.save_for_backward(k, v, q)
        ctx.layer_id = layer_id

        weight_update = torch.einsum('bi, bj->bij', v, k)
        fast_weight[layer_id] += weight_update
        output = torch.einsum('bij, bj->bi', fast_weight[layer_id], q)

        return output

    @staticmethod
    def backward(ctx, grad_out):
        global fast_weight, grad_fast_weight
        k, v, q = ctx.saved_tensors
        layer_id = ctx.layer_id

        # compute grad_q
        grad_q = torch.einsum("bij,bi->bj", fast_weight[layer_id], grad_out)

        # revert update fast weight
        fast_weight[layer_id] -= torch.einsum('bi, bj->bij', v, k)

        # update grad_W
        grad_fast_weight[layer_id] += torch.einsum('bi, bj->bij', grad_out, q)

        # compute grad_k and grad_v
        grad_k = torch.einsum('bij, bi->bj', grad_fast_weight[layer_id], v)
        grad_v = torch.einsum('bij, bj->bi', grad_fast_weight[layer_id], k)

        return grad_k, grad_v, grad_q, None


# Fast weight memory update rule with removal.
# about 25% faster with bmm instead of einsum?
# einsum code left in comments to help readability
# Still some room for speed improvement by grouping bmm in the backward pass.
class FastWeightLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, k, v, q, beta, layer_id):
        # key, value, query vectors of the current input, each with (B, dim).
        # beta: scalar to interpolate old and new value vectors, (B, 1)
        global fast_weight

        # v_old = torch.einsum('bij, bj->bi', fast_weight[layer_id], k)
        v_old = torch.bmm(fast_weight[layer_id], k.unsqueeze(2)).squeeze()
        v_insert = beta * (v - v_old)  # equation merging remove and insert.
        ctx.save_for_backward(k, v, q, beta, v_old, v_insert)
        ctx.layer_id = layer_id

        # weight_update = torch.einsum('bi, bj->bij', v_insert, k)
        weight_update = torch.bmm(v_insert.unsqueeze(2), k.unsqueeze(1))
        fast_weight[layer_id] += weight_update
        # fast_weight is a global variable.
        # this is needed as we can not have extra, non-gradient parameter
        # in the backward pass.
        # output = torch.einsum('bij, bj->bi', fast_weight[layer_id], q)
        output = torch.bmm(fast_weight[layer_id], q.unsqueeze(2)).squeeze()

        return output

    @staticmethod
    def backward(ctx, grad_out):
        global fast_weight, grad_fast_weight
        k, v, q, beta, v_old, v_insert = ctx.saved_tensors
        layer_id = ctx.layer_id

        # compute grad_q
        # grad_q = torch.einsum("bij,bi->bj", fast_weight[layer_id], grad_out)
        grad_q = torch.bmm(
            grad_out.unsqueeze(1), fast_weight[layer_id]).squeeze()

        # update grad_W
        # grad_fast_weight[layer_id] +=torch.einsum('bi, bj->bij', grad_out, q)
        grad_fast_weight[layer_id] += torch.bmm(
            grad_out.unsqueeze(2), q.unsqueeze(1))

        # Forward formula:
        # W_t = W_t-1 + beta * v_t (x) k_t - beta * v_old (x) k_t
        #     = ============================ beta * (W_t-1 * k_t) (x) k_t

        # compute grad_v
        # tmp_grad = torch.einsum('bij, bj->bi', grad_fast_weight[layer_id], k)
        tmp_grad = torch.bmm(
            grad_fast_weight[layer_id], k.unsqueeze(2)).squeeze()
        grad_v = beta * tmp_grad
        # first contribution to beta's gradient
        # grad_beta = torch.einsum('bi, bi->b', tmp_grad, v).unsqueeze(-1)
        grad_beta = torch.bmm(
            tmp_grad.unsqueeze(1), v.unsqueeze(2)).squeeze(-1)
        # first contribution to key's gradient
        # grad_k = beta * torch.einsum(
        #     'bij, bi->bj', grad_fast_weight[layer_id], v)
        grad_k = beta * torch.bmm(
            v.unsqueeze(1), grad_fast_weight[layer_id]).squeeze()

        # compute remainder for grad_beta
        # grad_beta -= torch.einsum('bi, bi->b', tmp_grad, v_old).unsqueeze(-1)
        grad_beta -= torch.bmm(
            tmp_grad.unsqueeze(1), v_old.unsqueeze(2)).squeeze(-1)

        # compute remainder for grad_key
        # grad for v_old (x) k_t
        tmp_grad = - beta.unsqueeze(-1) * grad_fast_weight[layer_id]
        # grad_k += torch.einsum('bij, bi->bj', tmp_grad, v_old)
        grad_k += torch.bmm(v_old.unsqueeze(1), tmp_grad).squeeze()
        # tmp_grad = torch.einsum('bij, bj->bi', tmp_grad, k)  # grad for v_old
        tmp_grad = torch.bmm(tmp_grad, k.unsqueeze(2)).squeeze()
        # revert update fast weight
        # fast_weight[layer_id] -= torch.einsum('bi, bj->bij', v_insert, k)
        fast_weight[layer_id] -= torch.bmm(
            v_insert.unsqueeze(2), k.unsqueeze(1))

        # grad_k += torch.einsum("bji,bj->bi", fast_weight[layer_id], tmp_grad)
        grad_k += torch.bmm(
            tmp_grad.unsqueeze(1), fast_weight[layer_id]).squeeze()

        # extra contribution to grad_fast_weight
        # grad_fast_weight[layer_id] +=torch.einsum('bi, bj->bij', tmp_grad, k)
        grad_fast_weight[layer_id] += torch.bmm(
            tmp_grad.unsqueeze(2), k.unsqueeze(1))

        return grad_k, grad_v, grad_q, grad_beta, None


# For Debugging!
# this is a regular Linear Transformer.
# (sum update rule + ELU based attention)
# In general, the cuda implementation should be used instead.
class DebugStepWiseLinearTransformerLayer(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout, dropatt=0,
                 pre_lnorm=False, eps=1e-5, layer_id=None, num_layer=None):
        super(DebugStepWiseLinearTransformerLayer, self).__init__()
        print(f"Using DebugStepWiseLinearTransformerLayer {layer_id} --")
        assert layer_id is not None
        self.layer_id = layer_id
        self.num_layer = num_layer
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
        self.eps = eps

        self.fast_weight_func = FastWeightSumLinear.apply

        global fast_weight, grad_fast_weight
        num_device = torch.cuda.device_count()  # TODO add cpu case
        # allocate size of global variable lists
        for _ in range(num_device * num_layer):
            fast_weight.append(0)
            grad_fast_weight.append(0)

    def forward(self, h, attn_mask=None, mems=None):
        # multihead attention
        # shape h: (len, B, d_model)

        if mems is not None:
            c = torch.cat([mems, h], 0)
        else:
            c = h

        if self.pre_lnorm:
            # layer normalization
            c = self.layer_norm(c)

        bsz = h.size(1)

        # reset fast weights and grad.
        device_id = torch.cuda.current_device()
        # When using multiple GPU, we need serapate variable slots
        # for each layer for each GPU.
        global fast_weight, grad_fast_weight

        fast_weight[self.layer_id + device_id * self.num_layer] = torch.zeros(
                [bsz * self.n_head, self.d_head, self.d_head], device=h.device)

        grad_fast_weight[
            self.layer_id + device_id * self.num_layer] = torch.zeros(
            [bsz * self.n_head, self.d_head, self.d_head], device=h.device)

        output_list = []
        denominator_acc = torch.zeros(
            [bsz * self.n_head, self.d_head], device=h.device)

        for x in torch.unbind(c, dim=0):
            # get k, v, and q.
            # shape of x: (B, d_model)
            head_q = self.q_net(x)  # TODO merge this with kv transform
            # shape of head_q: (B, n_head * d_head)
            head_k, head_v = torch.chunk(self.kv_net(x), 2, -1)

            head_q = head_q.view(bsz * self.n_head, self.d_head)
            head_k = head_k.reshape(bsz * self.n_head, self.d_head)
            head_v = head_v.reshape(bsz * self.n_head, self.d_head)

            # TODO add dropout here?
            # transform q and k
            head_q = F.elu(head_q, 1., False) + 1.
            head_k = F.elu(head_k, 1., False) + 1.

            # (B * n_head, d_head)
            out = self.fast_weight_func(
                head_k, head_v, head_q,
                self.layer_id + device_id * self.num_layer)

            denominator_acc = denominator_acc + head_k.clone()
            # one denominator for each head in each batch
            denominator = torch.einsum(
                'bi,bi->b', denominator_acc, head_q).unsqueeze(-1)

            # out = self.scale * out / (denominator + self.eps)
            out = self.scale * out
            out = out.reshape(bsz, self.n_head * self.d_head)
            output_list.append(out.clone())

        layer_out = torch.stack(output_list)  # (len, B, n_head * d_head)
        # expect [qlen, bsz, n_head * d_head]

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


# Fast weight memory layer with our update rule
# and Katharopoulos et al's ELU based attention function.
class StepWiseLinearTransformerLayer(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout, dropatt=0,
                 pre_lnorm=False, eps=1e-5, layer_id=None, num_layer=None):
        super(StepWiseLinearTransformerLayer, self).__init__()
        print(f"Using StepWiseLinearTransformerLayer {layer_id} --")

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
        self.eps = eps

        self.fast_weight_func = FastWeightLinear.apply
        print("Update rule: FastWeightLinear")

        global fast_weight, grad_fast_weight
        num_device = torch.cuda.device_count()  # TODO add cpu case
        # allocate size of global variable lists
        for _ in range(num_device * num_layer):
            fast_weight.append(0)
            grad_fast_weight.append(0)

    def forward(self, h, attn_mask=None, mems=None):
        # multihead attention
        # shape h: (len, B, n_head * d_head)

        if mems is not None:
            assert False, "Not supported."
            c = torch.cat([mems, h], 0)
        else:
            c = h

        if self.pre_lnorm:
            # layer normalization
            c = self.layer_norm(c)

        slen, bsz, _ = h.size()

        # reset fast weights and grad.
        device_id = torch.cuda.current_device()
        # When using multiple GPU, we need serapate variable slots
        # for each layer for each GPU.
        global fast_weight, grad_fast_weight

        fast_weight[self.layer_id + device_id * self.num_layer] = torch.zeros(
                [bsz * self.n_head, self.d_head, self.d_head], device=h.device)

        grad_fast_weight[
            self.layer_id + device_id * self.num_layer] = torch.zeros(
            [bsz * self.n_head, self.d_head, self.d_head], device=h.device)

        qkvb = self.qkvb_net(c)
        qkvb = qkvb.view(slen, bsz * self.n_head, 3 * self.d_head + 1)
        head_q, head_k, head_v, head_beta = torch.split(
            qkvb, (self.d_head,) * 3 + (1,), -1)
        head_beta = torch.sigmoid(head_beta)

        # TODO add dropout here?
        # transform q and k
        head_q = F.elu(head_q, 1., False) + 1.
        head_k = F.elu(head_k, 1., False) + 1.

        # normalize k and q, crucial for stable training.
        head_k = head_k / head_k.sum(-1, keepdim=True)
        head_q = head_q / head_q.sum(-1, keepdim=True)

        # denominator_acc = torch.cumsum(head_k, dim=0)
        # # (len, bsz * self.n_head, self.d_head)
        # denominator = torch.einsum(
        #     'lbi,lbi->lb', denominator_acc, head_q).unsqueeze(-1)
        # (len, bsz * n_head, 1)

        output_list = []

        for pos in range(slen):
            # (B * n_head, d_head)
            out = self.fast_weight_func(
                head_k[pos], head_v[pos], head_q[pos], head_beta[pos],
                self.layer_id + device_id * self.num_layer)
            output_list.append(out.clone())

        layer_out = torch.stack(output_list)  # (len, B * n_head, d_head)
        layer_out = self.scale * layer_out / (denominator + self.eps)
        layer_out = layer_out.view(
            layer_out.size(0), bsz, self.n_head * self.d_head)
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


# Fast weight memory layer with our update rule
# and DPFP attention function.
class StepWiseDPFPLinearTransformerLayer(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout, dropatt=0,
                 pre_lnorm=False, eps=1e-5, layer_id=None, num_layer=None,
                 n_roll=3):
        super(StepWiseDPFPLinearTransformerLayer, self).__init__()
        print(f"Using StepWiseDPFPLinearTransformerLayer {layer_id} --")

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
        self.eps = eps

        self.fast_weight_func = FastWeightLinear.apply
        print("Update rule: FastWeightLinear")

        global fast_weight, grad_fast_weight
        num_device = torch.cuda.device_count()  # TODO add cpu case
        # allocate size of global variable lists
        for _ in range(num_device * num_layer):
            fast_weight.append(0)
            grad_fast_weight.append(0)

    def mul_roll_repeat(self, x):
        rolls = []
        for i in range(1, self.n_roll + 1):
            rolls.append(x * x.roll(shifts=i, dims=-1))
        return torch.cat(rolls, dim=-1)

    def forward(self, h, attn_mask=None, mems=None):
        # multihead attention
        # shape h: (len, B, n_head * d_head)

        if mems is not None:
            assert False, "Not supported."
            c = torch.cat([mems, h], 0)
        else:
            c = h

        if self.pre_lnorm:
            # layer normalization
            c = self.layer_norm(c)

        slen, bsz, _ = h.size()

        # reset fast weights and grad.
        device_id = torch.cuda.current_device()
        # When using multiple GPU, we need serapate variable slots
        # for each layer for each GPU.
        global fast_weight, grad_fast_weight

        # key/query dims:
        kq_head_dim = self.d_head * 2 * self.n_roll

        fast_weight[self.layer_id + device_id * self.num_layer] = torch.zeros(
                [bsz * self.n_head, self.d_head, kq_head_dim], device=h.device)

        grad_fast_weight[
            self.layer_id + device_id * self.num_layer] = torch.zeros(
            [bsz * self.n_head, self.d_head, kq_head_dim], device=h.device)

        qkvb = self.qkvb_net(c)
        qkvb = qkvb.view(slen, bsz * self.n_head, 3 * self.d_head + 1)
        head_q, head_k, head_v, head_beta = torch.split(
            qkvb, (self.d_head,) * 3 + (1,), -1)
        head_beta = torch.sigmoid(head_beta)

        act = lambda x: F.relu(x)  # relu or exp
        head_k = torch.cat([act(head_k), act(-head_k)], dim=-1)
        head_q = torch.cat([act(head_q), act(-head_q)], dim=-1)

        head_k = self.mul_roll_repeat(head_k)
        head_q = self.mul_roll_repeat(head_q)

        # normalize k and q, crucial for stable training.
        head_k = head_k / head_k.sum(-1, keepdim=True)
        head_q = head_q / head_q.sum(-1, keepdim=True)

        output_list = []

        for pos in range(slen):
            # (B * n_head, d_head)
            out = self.fast_weight_func(
                head_k[pos], head_v[pos], head_q[pos], head_beta[pos],
                self.layer_id + device_id * self.num_layer)
            output_list.append(out.clone())

        layer_out = torch.stack(output_list)  # (len, B * n_head, d_head)
        layer_out = self.scale * layer_out / (denominator + self.eps)
        layer_out = layer_out.view(
            layer_out.size(0), bsz, self.n_head * self.d_head)
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


if __name__ == '__main__':
    # Gradient test
    import torch

    eps = 1e-5
    signigicant_digit = 5

    # Compute (f(x+eps) - f(x-eps)) / (2 * eps) and compare to the gradient
    # computed by the op.
    # Tests pass if `signigicant_digit` first digits (after period) match.

    bsz = 3
    dim = 5
    v_dim = 2
    steps = 10

    fast_weight = [torch.zeros([bsz, v_dim, dim], dtype=torch.double)]
    grad_fast_weight = [torch.zeros([bsz, v_dim, dim], dtype=torch.double)]

    k = torch.rand([steps, bsz, dim], requires_grad=True, dtype=torch.double)
    v = torch.rand([steps, bsz, v_dim], requires_grad=True, dtype=torch.double)
    q = torch.rand([steps, bsz, dim], requires_grad=True, dtype=torch.double)
    beta = torch.rand([steps, bsz, 1], requires_grad=True, dtype=torch.double)

    fx_func = FastWeightLinear.apply

    # compute gradient
    output = torch.zeros([bsz, v_dim], dtype=torch.double)

    for i in range(steps):
        output += fx_func(k[i], v[i], q[i], beta[i], 0)

    loss = output.sum()
    loss.backward()

    # compute finite difference
    print("###################################")
    print("# Gradient test for key vectors...")
    print("###################################")

    for s in range(steps):
        for b in range(bsz):
            for d in range(dim):
                k.data[s][b][d] += eps

                fast_weight = [
                    torch.zeros([bsz, v_dim, dim], dtype=torch.double)]
                grad_fast_weight = [
                    torch.zeros([bsz, v_dim, dim], dtype=torch.double)]
                output = torch.zeros([bsz, v_dim], dtype=torch.double)

                for i in range(steps):
                    output += fx_func(k[i], v[i], q[i], beta[i], 0)

                f_plus = output.sum()

                k.data[s][b][d] -= 2 * eps

                fast_weight = [torch.zeros(
                    [bsz, v_dim, dim], dtype=torch.double)]
                grad_fast_weight = [torch.zeros(
                    [bsz, v_dim, dim], dtype=torch.double)]
                output = torch.zeros([bsz, v_dim], dtype=torch.double)

                for i in range(steps):
                    output += fx_func(k[i], v[i], q[i], beta[i], 0)

                f_minus = output.sum()

                fd_grad = (f_plus - f_minus) / (2 * eps)
                fd = fd_grad.item()
                print(fd)
                grad = k.grad[s, b, d].item()
                print(grad)
                fd_sig = str(fd)[:signigicant_digit + 2]
                grad_sig = str(grad)[:signigicant_digit + 2]
                assert fd_sig == grad_sig, f"FAILURE {s, b, d} !"
                print(f"key {s, b, d} pass!")
                # revert change
                k.data[s][b][d] += eps

    print("#####################################")
    print("# Gradient test for value vectors...")
    print("#####################################")

    for s in range(steps):
        for b in range(bsz):
            for d in range(v_dim):
                v.data[s][b][d] += eps

                fast_weight = [
                    torch.zeros([bsz, v_dim, dim], dtype=torch.double)]
                grad_fast_weight = [
                    torch.zeros([bsz, v_dim, dim], dtype=torch.double)]
                output = torch.zeros([bsz, v_dim], dtype=torch.double)
                for i in range(steps):
                    output += fx_func(k[i], v[i], q[i], beta[i], 0)

                f_plus = output.sum()

                v.data[s][b][d] -= 2 * eps

                fast_weight = [
                    torch.zeros([bsz, v_dim, dim], dtype=torch.double)]
                grad_fast_weight = [
                    torch.zeros([bsz, v_dim, dim], dtype=torch.double)]
                output = torch.zeros([bsz, v_dim], dtype=torch.double)
                for i in range(steps):
                    output += fx_func(k[i], v[i], q[i], beta[i], 0)

                f_minus = output.sum()

                fd_grad = (f_plus - f_minus) / (2 * eps)
                fd = fd_grad.item()
                print(fd)
                grad = v.grad[s, b, d].item()
                print(grad)
                fd_sig = str(fd)[:signigicant_digit + 2]
                grad_sig = str(grad)[:signigicant_digit + 2]
                assert fd_sig == grad_sig, f"FAILURE {s, b, d} !"
                print(f"value {s, b, d} pass!")
                # revert change
                v.data[s][b][d] += eps

    print("#####################################")
    print("# Gradient test for query vectors...")
    print("#####################################")

    for s in range(steps):
        for b in range(bsz):
            for d in range(dim):
                q.data[s][b][d] += eps

                fast_weight = [
                    torch.zeros([bsz, v_dim, dim], dtype=torch.double)]
                grad_fast_weight = [
                    torch.zeros([bsz, v_dim, dim], dtype=torch.double)]
                output = torch.zeros([bsz, v_dim], dtype=torch.double)
                for i in range(steps):
                    output += fx_func(k[i], v[i], q[i], beta[i], 0)

                f_plus = output.sum()

                q.data[s][b][d] -= 2 * eps

                fast_weight = [
                    torch.zeros([bsz, v_dim, dim], dtype=torch.double)]
                grad_fast_weight = [
                    torch.zeros([bsz, v_dim, dim], dtype=torch.double)]
                output = torch.zeros([bsz, v_dim], dtype=torch.double)
                for i in range(steps):
                    output += fx_func(k[i], v[i], q[i], beta[i], 0)

                f_minus = output.sum()

                fd_grad = (f_plus - f_minus) / (2 * eps)
                fd = fd_grad.item()
                print(fd)
                grad = q.grad[s, b, d].item()
                print(grad)
                fd_sig = str(fd)[:signigicant_digit + 2]
                grad_sig = str(grad)[:signigicant_digit + 2]
                assert fd_sig == grad_sig, f"FAILURE {s, b, d} !"
                print(f"query {s, b, d} pass!")
                # revert change
                q.data[s][b][d] += eps

    print("#####################################")
    print("# Gradient test for beta vectors...")
    print("#####################################")

    for s in range(steps):
        for b in range(bsz):
            beta.data[s][b][0] += eps

            fast_weight = [
                torch.zeros([bsz, v_dim, dim], dtype=torch.double)]
            grad_fast_weight = [
                torch.zeros([bsz, v_dim, dim], dtype=torch.double)]
            output = torch.zeros([bsz, v_dim], dtype=torch.double)
            for i in range(steps):
                output += fx_func(k[i], v[i], q[i], beta[i], 0)

            f_plus = output.sum()

            beta.data[s][b][0] -= 2 * eps

            fast_weight = [
                torch.zeros([bsz, v_dim, dim], dtype=torch.double)]
            grad_fast_weight = [
                torch.zeros([bsz, v_dim, dim], dtype=torch.double)]
            output = torch.zeros([bsz, v_dim], dtype=torch.double)
            for i in range(steps):
                output += fx_func(k[i], v[i], q[i], beta[i], 0)

            f_minus = output.sum()

            fd_grad = (f_plus - f_minus) / (2 * eps)
            fd = fd_grad.item()
            print(fd)
            grad = beta.grad[s, b, 0].item()
            print(grad)
            fd_sig = str(fd)[:signigicant_digit + 2]
            grad_sig = str(grad)[:signigicant_digit + 2]
            assert fd_sig == grad_sig, f"FAILURE {s, b} !"
            print(f"beta {s, b} pass!")
            # revert change
            beta.data[s][b][0] += eps

    print("All tests pass.")
