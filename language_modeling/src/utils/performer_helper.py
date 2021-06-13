# Helper functions for Performer attention

import torch
import math


# Code from https://github.com/lucidrains/performer-pytorch/tree/main/performer_pytorch
def orthogonal_matrix_chunk(cols, qr_uniform_q=False, device=None, qr_on_cpu=True):
    unstructured_block = torch.randn((cols, cols), device=device)
    if qr_on_cpu:
        # apparently torch.qr is known to be faster on cpu in many cases?
        # https://discuss.pytorch.org/t/doing-qr-decomposition-on-gpu-is-much-slower-than-on-cpu/21213
        q, r = torch.qr(unstructured_block.cpu(), some=True)
        # but gives memory leak in some cases...
    else:
        q, r = torch.qr(unstructured_block, some=True)
    q, r = map(lambda t: t.to(device), (q, r))

    if qr_uniform_q:
        d = torch.diag(r, 0)
        q *= d.sign()
    return q.t()


# Code from https://github.com/lucidrains/performer-pytorch/tree/main/performer_pytorch
def draw_orthogonal_random_matrix(
        nb_rows, nb_columns, scaling=0, qr_uniform_q=False, device=None):
    nb_full_blocks = int(nb_rows / nb_columns)

    block_list = []

    for _ in range(nb_full_blocks):
        q = orthogonal_matrix_chunk(
            nb_columns, qr_uniform_q=qr_uniform_q, device=device)
        block_list.append(q)

    remaining_rows = nb_rows - nb_full_blocks * nb_columns
    if remaining_rows > 0:
        q = orthogonal_matrix_chunk(
            nb_columns, qr_uniform_q=qr_uniform_q, device=device)
        block_list.append(q[:remaining_rows])

    final_matrix = torch.cat(block_list)

    if scaling == 0:
        multiplier = torch.randn(
            (nb_rows, nb_columns), device=device).norm(dim=1)
    elif scaling == 1:
        multiplier = math.sqrt(
            (float(nb_columns))) * torch.ones((nb_rows,), device=device)
    else:
        raise ValueError(f'Invalid scaling {scaling}')

    return torch.diag(multiplier) @ final_matrix


# given x and projection matrix, compute x'
# TODO extra stabilization needed? tried but was not helpful...
def prime(x, proj_matrix, kernel_eps=1e-4, old_behavior=False):
    # x: shape (B, len, dim)
    # proj_matrix (dim, dim)
    _, m = proj_matrix.shape

    offset = -torch.sum(x ** 2, dim=-1, keepdim=True) / 2

    u = torch.matmul(x, proj_matrix)  # Do we need an extra stabilizer here?

    pos = torch.exp(offset + u)
    neg = torch.exp(offset - u)

    out = torch.cat([pos, neg], dim=-1) + kernel_eps

    if old_behavior:
        factor = m ** -0.5
    else:
        factor = (2 * m) ** -0.5

    return out * factor
