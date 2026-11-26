"""Differentiable port of frozen CORE_CAPSULE_V1 distances, not skin clearance."""

import torch


def segment_distance(a, b, c, d):
    u = b - a
    v = d - c
    w = a - c
    A = (u * u).sum(-1)
    B = (u * v).sum(-1)
    C = (v * v).sum(-1)
    D = (u * w).sum(-1)
    E = (v * w).sum(-1)
    den = A * C - B * B

    def endpoint(p, x, y):
        yy = y - x
        t = ((p - x) * yy).sum(-1) / (yy * yy).sum(-1).clamp_min(1e-30)
        return torch.linalg.vector_norm(p - x - t.clamp(0, 1)[..., None] * yy, dim=-1)

    dist = torch.stack(
        [endpoint(a, c, d), endpoint(b, c, d), endpoint(c, a, b), endpoint(d, a, b)]
    ).amin(0)
    good = den.abs() > 1e-24
    safe = torch.where(good, den, torch.ones_like(den))
    s = (B * E - C * D) / safe
    t = (A * E - B * D) / safe
    interior = torch.linalg.vector_norm(w + s[..., None] * u - t[..., None] * v, dim=-1)
    return torch.where(
        good & (s >= 0) & (s <= 1) & (t >= 0) & (t <= 1),
        torch.minimum(dist, interior),
        dist,
    )


def evaluate(chains, config):
    pair = torch.as_tensor(config["pairs"], device=chains.device, dtype=torch.long)
    radii = torch.as_tensor(
        config["radius_m"], device=chains.device, dtype=chains.dtype
    )
    if not torch.isfinite(radii).all():
        raise ValueError("UNKNOWN_PROXY_RADII")
    a = chains[:, :, :-1].reshape(-1, 15, 3)
    b = chains[:, :, 1:].reshape(-1, 15, 3)
    u = b - a
    aa = a + 0.15 * u
    bb = b - 0.15 * u
    d = segment_distance(
        aa[:, pair[:, 0]], bb[:, pair[:, 0]], aa[:, pair[:, 1]], bb[:, pair[:, 1]]
    )
    depth = radii[pair[:, 0]] + radii[pair[:, 1]] - d
    # Loss repels positive CORE overlap. Existing >1mm screen threshold is retained
    # separately; neither radius nor epsilon is changed to make a result pass.
    loss = (torch.relu(depth) / 0.001).square().mean(-1)
    return {
        "loss": loss,
        "depth_m": depth,
        "screen_positive": depth > 0.001,
        "distance_m": d,
    }
