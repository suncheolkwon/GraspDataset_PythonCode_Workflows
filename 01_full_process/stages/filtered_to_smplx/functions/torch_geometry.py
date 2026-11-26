"""Differentiable same geometric chart; undefined anatomy is never a target."""

import torch
import reset_evaluation as e


def unit(x):
    return x / torch.linalg.vector_norm(x, dim=-1, keepdim=True).clamp_min(1e-8)


def palm(w, c):
    y = unit(c[:, 1:4, 0].mean(1) - w)
    span = c[:, 4, 0] - c[:, 1, 0]
    x = unit(span - y * (span * y).sum(-1, keepdim=True))
    z = unit(torch.cross(x, y, dim=-1))
    return torch.stack([x, y, z], -1)


def safe_atan2(y, x):
    good = x.abs() + y.abs() > 1e-10
    return torch.atan2(
        torch.where(good, y, torch.zeros_like(y)),
        torch.where(good, x, torch.ones_like(x)),
    )


def angles(c, R, side):
    d = unit(c[:, :, 1:] - c[:, :, :-1])
    normal = R[:, :, 2] * (1 if side == "L" else -1)
    local = torch.einsum("tji,tdj->tdi", R, d[:, :, 0])
    f = safe_atan2(local[:, :, 2] * (1 if side == "L" else -1), local[:, :, 1])
    a = safe_atan2(local[:, :, 0], local[:, :, 1])
    cross = torch.cross(d[:, :, 0], normal[:, None].expand_as(d[:, :, 0]), dim=-1)
    axis = unit(cross)
    tn = torch.linalg.vector_norm(cross, dim=-1)
    projection = d - axis[:, :, None] * (d * axis[:, :, None]).sum(-1, keepdim=True)
    pn = torch.linalg.vector_norm(projection, dim=-1)
    u = unit(projection)
    vals = [f, a]
    for k in range(2):
        vals.append(
            safe_atan2(
                (axis * torch.cross(u[:, :, k], u[:, :, k + 1], dim=-1)).sum(-1),
                (u[:, :, k] * u[:, :, k + 1]).sum(-1),
            )
        )
    q = torch.stack(
        [
            torch.linalg.vector_norm(local[:, :, [1, 2]], dim=-1),
            torch.linalg.vector_norm(local[:, :, [0, 1]], dim=-1),
            torch.minimum(torch.minimum(pn[:, :, 0], pn[:, :, 1]), tn),
            torch.minimum(torch.minimum(pn[:, :, 1], pn[:, :, 2]), tn),
        ],
        -1,
    )
    return torch.stack(vals, -1), q


def chain(out, side):
    tips = out.vertices[:, list(e.TIPS[side])]
    c = torch.stack(
        [
            torch.cat([out.joints[:, b : b + 3], tips[:, k, None]], 1)
            for k, b in enumerate(e.BASE[side])
        ],
        1,
    )
    return c, palm(out.joints[:, e.WRIST[side]], c)
