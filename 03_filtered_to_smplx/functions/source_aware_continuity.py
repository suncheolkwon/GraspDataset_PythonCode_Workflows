"""F03A source-aware edge residuals; all targets keep their original timestamps.

This is an optimization objective, not an anatomical feasibility certificate.
Each sample uses its own proper geometric palm frame. Relative TIP displacement
is not wrist-relative world displacement. Sources are detached, never smoothed.
"""

import math
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from functional_terms import _rotation_angle_rad


def palm_local(chains, wrist, palm):
    return torch.einsum("nji,ndlj->ndli", palm, chains - wrist[:, None, None])


def source_aware_edges(
    model_chains,
    model_wrist,
    model_palm,
    source_chains,
    source_wrist,
    source_palm,
    dt,
    edge_valid,
    policy,
):
    """Loss over explicit valid consecutive edges, including fixed/free edges.

    TIP and unit-link INCREMENT vectors are ordered, not merely speed magnitudes.
    A true rapid source motion reproduced exactly receives zero added loss.
    max over features and max over edges prevents isolated errors being averaged
    away. No temporal edge may cross invalid rows, segments or missing indices.
    """
    if len(model_chains) < 2:
        return model_chains.sum() * 0
    assert dt.shape == edge_valid.shape == (len(model_chains) - 1,)
    assert edge_valid.dtype == torch.bool
    assert bool(torch.isfinite(dt).all()) and bool((dt > 0).all())
    if not bool(edge_valid.any()):
        return model_chains.sum() * 0
    ml = palm_local(model_chains, model_wrist, model_palm)
    sl = palm_local(source_chains.detach(), source_wrist.detach(), source_palm.detach())
    mi = ml[1:, :, -1] - ml[:-1, :, -1]
    si = sl[1:, :, -1] - sl[:-1, :, -1]
    tip_allowance = dt[:, None] * policy["tip_increment_residual_allowance_m_per_s"]
    tip = (
        (torch.linalg.vector_norm(mi - si, dim=-1) / tip_allowance - 1)
        .clamp_min(0)
        .square()
    )
    md = ml[:, :, 1:] - ml[:, :, :-1]
    sd = sl[:, :, 1:] - sl[:, :, :-1]
    md = md / torch.linalg.vector_norm(md, dim=-1, keepdim=True).clamp_min(1e-12)
    sd = sd / torch.linalg.vector_norm(sd, dim=-1, keepdim=True).clamp_min(1e-12)
    link_allowance = 2 * torch.sin(
        dt
        * math.radians(policy["local_link_increment_residual_allowance_deg_per_s"])
        / 2
    )
    link = (
        (
            torch.linalg.vector_norm((md[1:] - md[:-1]) - (sd[1:] - sd[:-1]), dim=-1)
            / link_allowance[:, None, None]
            - 1
        )
        .clamp_min(0)
        .square()
    )
    mr = model_palm[:-1].transpose(-1, -2) @ model_palm[1:]
    sr = source_palm[:-1].detach().transpose(-1, -2) @ source_palm[1:].detach()
    palm_allowance = dt * math.radians(
        policy["palm_rotation_increment_residual_allowance_deg_per_s"]
    )
    palm = (_rotation_angle_rad(mr, sr) / palm_allowance - 1).clamp_min(0).square()
    edge = torch.cat([tip, link.flatten(1), palm[:, None]], dim=1).amax(1)[edge_valid]
    return edge.mean() + edge.max()


def continuation_initial_guess(
    reference_suffix, previous_output, previous_reference, elapsed_s, horizon_s
):
    """Transport preceding fitted SO(3) residual into F02A initial guess only.

    R_init(t)=Exp(w(t)*Log(R_previous_output*R_previous_F02A.T))*R_F02A(t).
    w decreases to zero over the declared horizon. The source and saved output
    are never interpolated/filtered. Dimensions [time,joint,rotvec3].
    """
    a = np.asarray(reference_suffix)
    if len(a) == 0:
        return a.copy()
    d = (
        Rotation.from_rotvec(np.asarray(previous_output).reshape(-1, 3))
        * Rotation.from_rotvec(np.asarray(previous_reference).reshape(-1, 3)).inv()
    )
    delta = d.as_rotvec()
    weight = np.maximum(0.0, 1.0 - np.asarray(elapsed_s, dtype=float) / horizon_s)
    corr = Rotation.from_rotvec((weight[:, None, None] * delta[None]).reshape(-1, 3))
    seed = Rotation.from_rotvec(a.reshape(-1, 3))
    result = (corr * seed).as_rotvec().reshape(a.shape).astype(a.dtype)
    # Beyond the finite initial transport horizon preserve exact F02A values.
    result[weight == 0] = a[weight == 0]
    return result
