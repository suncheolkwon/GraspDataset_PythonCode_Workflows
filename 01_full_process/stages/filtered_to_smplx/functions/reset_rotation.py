"""Batched SO(3) exponential map for native SMPL-X axis-angle parameters."""

import torch


def aa2rot_torch(x):
    z = torch.zeros_like(x[..., 0])
    a, b, c = x.unbind(-1)
    h = torch.stack((z, -c, b, c, z, -a, -b, a, z), dim=-1).reshape(*x.shape[:-1], 3, 3)
    return torch.matrix_exp(h)
