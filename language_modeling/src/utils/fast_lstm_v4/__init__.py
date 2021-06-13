# Adaptation of the original code from
# https://github.com/idiap/fast-transformers/blob/master/fast_transformers/causal_product/__init__.py
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>,
# Apoorv Vyas <avyas@idiap.ch>
#
# Modifications Copyright (c) 2021 Kazuki Irie


import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load
# Just in time import
# https://pytorch.org/tutorials/advanced/cpp_extens

# The extra arg `extra_cuda_cflags=['--ftemplate-depth=1024']` solves:
# ```
# pybind11/detail/common.h(461):
# error: excessive recursion at instantiation of class
# ```
mod_causal_dot_product_cuda = load(
    extra_cuda_cflags=['--ftemplate-depth=1024'],
    name="fast_lstm_v4_forward",
    sources=["utils/fast_lstm_v4/fast_lstm_v4_cuda.cu"], verbose=True)

mod_causal_dot_backward_cuda = load(
    extra_cuda_cflags=['--ftemplate-depth=1024'],
    name="fast_lstm_v4_backward",
    sources=["utils/fast_lstm_v4/fast_lstm_v4_cuda.cu"], verbose=True)


causal_dot_product_cuda = mod_causal_dot_product_cuda.fast_lstm_v4_forward
causal_dot_backward_cuda = mod_causal_dot_backward_cuda.fast_lstm_v4_backward


class FastLSTMv4(torch.autograd.Function):
    """Fast LSTM with the FWM update rule."""
    dot = {
        # "cpu": causal_dot_product_cpu,
        "cuda": causal_dot_product_cuda
    }
    dot_backward = {
        # "cpu": causal_dot_backward_cpu,
        "cuda": causal_dot_backward_cuda
    }

    @staticmethod
    def forward(ctx,
                Zi, Ki, Vi, bi, Wi,
                Zu, Ku, Vu, bu, Wu,
                Zo, Ko, Vo, bo, Wo, h0, c0):
        # Computations:
        #   fast weights with sum update rule: R_t = R_t-1 + v_t (x) k_t
        #   output: h_t = tanh(R_t * h_t-1 + z_t)
        # z_t is the output of a feed-forward fast weight layer.
        # h0 is the initial RNN state.
        # E = M.

        # Create the output tensor
        device = Zi.device
        N, H, L, _ = Zi.shape
        _, _, _, M = Vi.shape

        assert Ki.shape == (N, H, L, M)
        assert Vi.shape == (N, H, L, M)
        assert h0.shape == (N, H, 1, M)
        assert Wi.shape == (N, H, M, M)

        rnn_out = torch.zeros((N, H, L, M), device=device, dtype=Zi.dtype)
        cell_out = torch.zeros((N, H, L, M), device=device, dtype=Zi.dtype)
        gate_i = torch.zeros((N, H, L, M), device=device, dtype=Zi.dtype)
        update_u = torch.zeros((N, H, L, M), device=device, dtype=Zi.dtype)
        gate_o = torch.zeros((N, H, L, M), device=device, dtype=Zi.dtype)

        out_del_nmz = torch.zeros((N, H, L, M), device=device, dtype=Zi.dtype)

        h_init = h0.detach().clone()
        c_init = c0.detach().clone()

        V_old_i = torch.zeros((N, H, L, M), device=device, dtype=Zi.dtype)
        V_old_u = torch.zeros((N, H, L, M), device=device, dtype=Zi.dtype)
        V_old_o = torch.zeros((N, H, L, M), device=device, dtype=Zi.dtype)

        # W = torch.zeros((N, H, E, M), device=device, dtype=Z.dtype)
        # h0 = torch.zeros((N, H, M), device=device, dtype=Z.dtype)

        # Actually perform the dot product
        FastLSTMv4.dot[device.type](
            Zi.data,  # input gate
            Ki.data,
            Vi.data,
            bi.data,
            Zu.data,  # update candidate
            Ku.data,
            Vu.data,
            bu.data,
            Zo.data,  # ouput gate
            Ko.data,
            Vo.data,
            bo.data,
            h0.data,  # init hidden states
            c0.data,  # init cell states
            Wi,
            Wu,
            Wo,
            rnn_out,
            out_del_nmz,
            cell_out,
            gate_i,
            update_u,
            gate_o,
            V_old_i,
            V_old_u,
            V_old_o
        )

        ctx.save_for_backward(rnn_out, out_del_nmz,
                              cell_out, gate_i, update_u, gate_o,
                              Zi, Ki, Vi, Wi, bi,
                              Zu, Ku, Vu, Wu, bu,
                              Zo, Ko, Vo, Wo, bo,
                              V_old_i, V_old_u, V_old_o,
                              h_init, c_init)

        return rnn_out, cell_out

    @staticmethod
    def backward(ctx, grad_out, grad_cell):
        # Extract the saved tensors
        (rnn_out, rnn_out_delayed, cell_out, gate_i, update_u, gate_o,
         Zi, Ki, Vi, Wi, bi,
         Zu, Ku, Vu, Wu, bu,
         Zo, Ko, Vo, Wo, bo,
         V_old_i, V_old_u, V_old_o,
         h0, c0) = ctx.saved_tensors

        # Allocate memory for the gradients
        grad_Zi = torch.zeros_like(Zi)
        grad_Ki = torch.zeros_like(Ki)
        grad_Vi = torch.zeros_like(Vi)
        grad_bi = torch.zeros_like(bi)

        grad_Zu = torch.zeros_like(Zu)
        grad_Ku = torch.zeros_like(Ku)
        grad_Vu = torch.zeros_like(Vu)
        grad_bu = torch.zeros_like(bu)

        grad_Zo = torch.zeros_like(Zo)
        grad_Ko = torch.zeros_like(Ko)
        grad_Vo = torch.zeros_like(Vo)
        grad_bo = torch.zeros_like(bo)

        # Prepare delayed RNN outputs
        # shape of rnn_out: N, H, L, M
        # dim2 is the time dim.
        # shape of h0: N, H, 1, M
        # rnn_out_delayed = torch.cat([h0, rnn_out[:, :, :-1]], dim=2)
        c_out_delayed = torch.cat([c0, cell_out[:, :, :-1]], dim=2)
        # In the backward pass, we need u_t - cell_{t-1} and not delayed cell.
        u_minus_c = update_u - c_out_delayed

        # Compute the gradients
        FastLSTMv4.dot_backward[Zi.device.type](
            grad_out,
            Ki.data,
            Vi.data,
            bi.data,
            Ku.data,
            Vu.data,
            bu.data,
            Ko.data,
            Vo.data,
            bo.data,
            V_old_i.data,
            V_old_u.data,
            V_old_o.data,
            rnn_out,
            rnn_out_delayed,
            cell_out,
            u_minus_c,
            gate_i,
            update_u,
            gate_o,
            Wi.data,
            Wu.data,
            Wo.data,
            grad_Zi,
            grad_Ki,
            grad_Vi,
            grad_bi,
            grad_Zu,
            grad_Ku,
            grad_Vu,
            grad_bu,
            grad_Zo,
            grad_Ko,
            grad_Vo,
            grad_bo,
        )

        return (grad_Zi, grad_Ki, grad_Vi, grad_bi, None,
                grad_Zu, grad_Ku, grad_Vu, grad_bu, None,
                grad_Zo, grad_Ko, grad_Vo, grad_bo, None,
                None, None)


# Alias the autograd functions to python style snake case naming
fast_lstm_v4 = FastLSTMv4.apply


if __name__ == '__main__':
    import torch
    torch.manual_seed(111)
    # Tests pass if the relative difference compared with
    # the corresponding torch autograd computation
    # is smaller than a threshold.

    # Ideally should be tested with double...
    rel_threshold = 1e-2

    # from https://github.com/idiap/fast-transformers/blob/master/tests/causal_product/test_causal_product_gpu.py
    def max_relative_error(a, b, eps=1e-8):
        return float(torch.abs((b - a) / (torch.abs(b) + eps)).max().item())

    print('##########################')
    print('# Test forward pass')
    print('##########################')

    bsz, n_head, slen, d_head = 3, 5, 11, 18
    v_dim = d_head

    h0 = torch.zeros(bsz, n_head, 1, v_dim, device='cuda')
    c0 = torch.zeros(bsz, n_head, 1, v_dim, device='cuda')

    # (B, H, len, dim)

    k0i = torch.rand(bsz, n_head, slen, d_head, device='cuda')
    v0i = torch.rand(bsz, n_head, slen, v_dim, device='cuda')
    z0i = torch.rand(bsz, n_head, slen, v_dim, device='cuda')
    b0i = torch.sigmoid(torch.rand(bsz, n_head, slen, 1, device='cuda'))

    k0u = torch.rand(bsz, n_head, slen, d_head, device='cuda')
    v0u = torch.rand(bsz, n_head, slen, v_dim, device='cuda')
    z0u = torch.rand(bsz, n_head, slen, v_dim, device='cuda')
    b0u = torch.sigmoid(torch.rand(bsz, n_head, slen, 1, device='cuda'))

    k0o = torch.rand(bsz, n_head, slen, d_head, device='cuda')
    v0o = torch.rand(bsz, n_head, slen, v_dim, device='cuda')
    z0o = torch.rand(bsz, n_head, slen, v_dim, device='cuda')
    b0o = torch.sigmoid(torch.rand(bsz, n_head, slen, 1, device='cuda'))

    # key sum norm
    k0i = F.softmax(k0i, dim=-1)
    k0u = F.softmax(k0u, dim=-1)
    k0o = F.softmax(k0o, dim=-1)

    k1i = torch.zeros(
        bsz, n_head, slen, d_head, requires_grad=True, device='cuda')
    v1i = torch.zeros(
        bsz, n_head, slen, v_dim, requires_grad=True, device='cuda')
    z1i = torch.zeros(
        bsz, n_head, slen, v_dim, requires_grad=True, device='cuda')
    b1i = torch.zeros(
        bsz, n_head, slen, 1, requires_grad=True, device='cuda')
    # update candidate
    k1u = torch.zeros(
        bsz, n_head, slen, d_head, requires_grad=True, device='cuda')
    v1u = torch.zeros(
        bsz, n_head, slen, v_dim, requires_grad=True, device='cuda')
    z1u = torch.zeros(
        bsz, n_head, slen, v_dim, requires_grad=True, device='cuda')
    b1u = torch.zeros(
        bsz, n_head, slen, 1, requires_grad=True, device='cuda')
    # output gate
    k1o = torch.zeros(
        bsz, n_head, slen, d_head, requires_grad=True, device='cuda')
    v1o = torch.zeros(
        bsz, n_head, slen, v_dim, requires_grad=True, device='cuda')
    z1o = torch.zeros(
        bsz, n_head, slen, v_dim, requires_grad=True, device='cuda')
    b1o = torch.zeros(
        bsz, n_head, slen, 1, requires_grad=True, device='cuda')

    # q1.data = q0.data
    k1i.data = k0i.data
    v1i.data = v0i.data
    b1i.data = b0i.data
    z1i.data = z0i.data

    k1u.data = k0u.data
    v1u.data = v0u.data
    b1u.data = b0u.data
    z1u.data = z0u.data

    k1o.data = k0o.data
    v1o.data = v0o.data
    b1o.data = b0o.data
    z1o.data = z0o.data

    W1i = torch.zeros(bsz, n_head, d_head, v_dim, device='cuda')
    W1u = torch.zeros(bsz, n_head, d_head, v_dim, device='cuda')
    W1o = torch.zeros(bsz, n_head, d_head, v_dim, device='cuda')

    # h0 = torch.zeros(n_head, d_head, v_dim, device='cuda')
    print("Forwarding custom kernel...")
    out1, _ = fast_lstm_v4(z1i, k1i, v1i, b1i, W1i,
                           z1u, k1u, v1u, b1u, W1u,
                           z1o, k1o, v1o, b1o, W1o,
                           h0, c0)
    print("done.")

    # compute using torch
    k2i = torch.zeros(
        bsz, n_head, slen, d_head, requires_grad=True, device='cuda')
    v2i = torch.zeros(
        bsz, n_head, slen, v_dim, requires_grad=True, device='cuda')
    z2i = torch.zeros(
        bsz, n_head, slen, v_dim, requires_grad=True, device='cuda')
    b2i = torch.zeros(
        bsz, n_head, slen, 1, requires_grad=True, device='cuda')
    # update candidate
    k2u = torch.zeros(
        bsz, n_head, slen, d_head, requires_grad=True, device='cuda')
    v2u = torch.zeros(
        bsz, n_head, slen, v_dim, requires_grad=True, device='cuda')
    z2u = torch.zeros(
        bsz, n_head, slen, v_dim, requires_grad=True, device='cuda')
    b2u = torch.zeros(
        bsz, n_head, slen, 1, requires_grad=True, device='cuda')
    # output gate
    k2o = torch.zeros(
        bsz, n_head, slen, d_head, requires_grad=True, device='cuda')
    v2o = torch.zeros(
        bsz, n_head, slen, v_dim, requires_grad=True, device='cuda')
    z2o = torch.zeros(
        bsz, n_head, slen, v_dim, requires_grad=True, device='cuda')
    b2o = torch.zeros(
        bsz, n_head, slen, 1, requires_grad=True, device='cuda')

    # q1.data = q0.data
    k2i.data = k0i.data
    v2i.data = v0i.data
    b2i.data = b0i.data
    z2i.data = z0i.data

    k2u.data = k0u.data
    v2u.data = v0u.data
    b2u.data = b0u.data
    z2u.data = z0u.data

    k2o.data = k0o.data
    v2o.data = v0o.data
    b2o.data = b0o.data
    z2o.data = z0o.data

    # (len, B, H, dim)
    # input
    z_2i = z2i.permute(2, 0, 1, 3)
    slen, bsz, n_head, d_head = z_2i.shape
    z_2i = z_2i.reshape(slen, bsz * n_head, d_head)

    k_2i = k2i.permute(2, 0, 1, 3)
    k_2i = k_2i.reshape(slen, bsz * n_head, d_head)

    v_2i = v2i.permute(2, 0, 1, 3)
    v_2i = v_2i.reshape(slen, bsz * n_head, v_dim)

    b_2i = b2i.permute(2, 0, 1, 3)
    b_2i = b_2i.reshape(slen, bsz * n_head, 1)

    # update
    z_2u = z2u.permute(2, 0, 1, 3)
    z_2u = z_2u.reshape(slen, bsz * n_head, d_head)

    k_2u = k2u.permute(2, 0, 1, 3)
    k_2u = k_2u.reshape(slen, bsz * n_head, d_head)

    v_2u = v2u.permute(2, 0, 1, 3)
    v_2u = v_2u.reshape(slen, bsz * n_head, v_dim)

    b_2u = b2u.permute(2, 0, 1, 3)
    b_2u = b_2u.reshape(slen, bsz * n_head, 1)

    # output gate
    z_2o = z2o.permute(2, 0, 1, 3)
    z_2o = z_2o.reshape(slen, bsz * n_head, d_head)

    k_2o = k2o.permute(2, 0, 1, 3)
    k_2o = k_2o.reshape(slen, bsz * n_head, d_head)

    v_2o = v2o.permute(2, 0, 1, 3)
    v_2o = v_2o.reshape(slen, bsz * n_head, v_dim)

    b_2o = b2o.permute(2, 0, 1, 3)
    b_2o = b_2o.reshape(slen, bsz * n_head, 1)

    Wi = torch.zeros(bsz * n_head, v_dim, d_head, device='cuda')
    Wu = torch.zeros(bsz * n_head, v_dim, d_head, device='cuda')
    Wo = torch.zeros(bsz * n_head, v_dim, d_head, device='cuda')

    h = torch.zeros(bsz * n_head, d_head, device='cuda')
    cell = torch.zeros(bsz * n_head, d_head, device='cuda')

    out_list = []
    print("Forwarding PyTorch code...")
    for pos in range(slen):
        # get old values
        v_old_i = torch.bmm(Wi, k_2i[pos].unsqueeze(2)).squeeze()
        v_old_u = torch.bmm(Wu, k_2u[pos].unsqueeze(2)).squeeze()
        v_old_o = torch.bmm(Wo, k_2o[pos].unsqueeze(2)).squeeze()

        v_insert_i = b_2i[pos] * (v_2i[pos] - v_old_i)
        v_insert_u = b_2u[pos] * (v_2u[pos] - v_old_u)
        v_insert_o = b_2o[pos] * (v_2o[pos] - v_old_o)

        # update fast weights
        Wi = Wi + torch.bmm(v_insert_i.unsqueeze(2), k_2i[pos].unsqueeze(1))
        Wu = Wu + torch.bmm(v_insert_u.unsqueeze(2), k_2u[pos].unsqueeze(1))
        Wo = Wo + torch.bmm(v_insert_o.unsqueeze(2), k_2o[pos].unsqueeze(1))

        h = F.softmax(h, dim=-1)
        gate_i = torch.sigmoid(
            torch.bmm(Wi, h.unsqueeze(2)).squeeze() + z_2i[pos])
        # v4
        rec_u = torch.bmm(Wu, h.unsqueeze(2)).squeeze() + z_2u[pos]
        gate_o = torch.sigmoid(
            torch.bmm(Wo, h.unsqueeze(2)).squeeze() + z_2o[pos])

        cell = gate_i * rec_u + (1. - gate_i) * cell
        h = cell * gate_o

        out_list.append(h.clone())
    print("done.")

    out2 = torch.stack(out_list)
    out2 = out2.view(slen, bsz, n_head, v_dim)

    out1 = out1.permute(2, 0, 1, 3)

    for s in range(slen):
        for b in range(bsz):
            for h in range(n_head):
                print(f"forward: s={s} b={b} h={h}")
                print(f"out: {out1[s][b][h]}")
                print(f"ref: {out2[s][b][h]}")
                assert max_relative_error(
                    out1[s][b][h], out2[s][b][h]) < rel_threshold
                print("pass!")

    print('##########################')
    print('# Test Backward pass')
    print('##########################')

    # grad
    loss1 = out1.sum()
    z1i.retain_grad()
    k1i.retain_grad()
    v1i.retain_grad()
    b1i.retain_grad()

    z1u.retain_grad()
    k1u.retain_grad()
    v1u.retain_grad()
    b1u.retain_grad()

    z1o.retain_grad()
    k1o.retain_grad()
    v1o.retain_grad()
    b1o.retain_grad()

    loss1.backward()

    loss2 = out2.sum()
    z2i.retain_grad()
    k2i.retain_grad()
    v2i.retain_grad()
    b2i.retain_grad()

    z2u.retain_grad()
    k2u.retain_grad()
    v2u.retain_grad()
    b2u.retain_grad()

    z2o.retain_grad()
    k2o.retain_grad()
    v2o.retain_grad()
    b2o.retain_grad()

    loss2.backward()

    thr = 1e-6
    for s in reversed(range(slen)):
        for b in reversed(range(bsz)):
            for h in range(n_head):
                print(f" === backward: s={s}, b={b}, h={h} ===")

                # Output gate
                print("Output gate ---")
                print(f"grad input out: {z1o.grad[b][h][s]}")
                print(f"grad input ref: {z2o.grad[b][h][s]}")
                assert max_relative_error(
                    z1o.grad[b][h][s], z2o.grad[b][h][s]) < rel_threshold
                print("pass!")

                print(f"grad key out: {k1o.grad[b][h][s]}")
                print(f"grad key ref: {k2o.grad[b][h][s]}")
                assert max_relative_error(
                    k1o.grad[b][h][s], k2o.grad[b][h][s]) < rel_threshold
                print("pass!")

                print(f"grad beta out: {b1o.grad[b][h][s]}")
                print(f"grad beta ref: {b2o.grad[b][h][s]}")
                assert max_relative_error(
                    b1o.grad[b][h][s], b2o.grad[b][h][s]) < rel_threshold
                print("pass!")

                print(f"grad value out: {v1o.grad[b][h][s]}")
                print(f"grad value ref: {v2o.grad[b][h][s]}")
                assert max_relative_error(
                    v1o.grad[b][h][s], v2o.grad[b][h][s]) < rel_threshold
                print("pass!")

                # Update term
                print("Update candidate ---")
                print(f"grad input out: {z1u.grad[b][h][s]}")
                print(f"grad input ref: {z2u.grad[b][h][s]}")
                assert max_relative_error(
                    z1u.grad[b][h][s], z2u.grad[b][h][s]) < rel_threshold
                print("pass!")

                print(f"grad key out: {k1u.grad[b][h][s]}")
                print(f"grad key ref: {k2u.grad[b][h][s]}")
                assert max_relative_error(
                    k1u.grad[b][h][s], k2u.grad[b][h][s]) < rel_threshold
                print("pass!")

                print(f"grad value out: {v1u.grad[b][h][s]}")
                print(f"grad value ref: {v2u.grad[b][h][s]}")
                assert max_relative_error(
                    v1u.grad[b][h][s], v2u.grad[b][h][s]) < rel_threshold
                print("pass!")

                print(f"grad beta out: {b1u.grad[b][h][s]}")
                print(f"grad beta ref: {b2u.grad[b][h][s]}")
                assert max_relative_error(
                    b1u.grad[b][h][s], b2u.grad[b][h][s]) < rel_threshold
                print("pass!")

                # Input gate
                print("Input gate ---")
                print(f"grad input out: {z1i.grad[b][h][s]}")
                print(f"grad input ref: {z2i.grad[b][h][s]}")
                assert max_relative_error(
                    z1i.grad[b][h][s], z2i.grad[b][h][s]) < rel_threshold
                print("pass!")

                print(f"grad key out: {k1i.grad[b][h][s]}")
                print(f"grad key ref: {k2i.grad[b][h][s]}")
                assert max_relative_error(
                    k1i.grad[b][h][s], k2i.grad[b][h][s]) < rel_threshold
                print("pass!")

                print(f"grad value out: {v1i.grad[b][h][s]}")
                print(f"grad value ref: {v2i.grad[b][h][s]}")
                assert max_relative_error(
                    v1i.grad[b][h][s], v2i.grad[b][h][s]) < rel_threshold
                print("pass!")

                print(f"grad beta out: {b1i.grad[b][h][s]}")
                print(f"grad beta ref: {b2i.grad[b][h][s]}")
                assert max_relative_error(
                    b1i.grad[b][h][s], b2i.grad[b][h][s]) < rel_threshold
                print("pass!")

    print("All tests pass.")
