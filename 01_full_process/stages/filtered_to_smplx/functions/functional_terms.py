"""Differentiable, source-timed hand relation bands for the bounded pilot.

All coordinates are metres; palm matrices have proper local-to-world columns.
Digit order is Thumb, Index, Middle, Ring, Pinky. Chain nodes are the existing
three named joints followed by the existing fixed TIP proxy. This module does
not infer contact, anatomical axes, native-rotation limits, or surface clearance.
Do not treat a low loss as a nonpenetration or anatomical-feasibility certificate.

Invalid rows remain NaN in public losses/metrics and carry explicit masks. Safe
finite fillers are used only internally before arithmetic so excluded rows cannot
poison backward; masked_mean_loss never counts them as valid zero-loss samples.
"""

from dataclasses import dataclass, asdict
import math
import torch

DIGITS = ("Thumb", "Index", "Middle", "Ring", "Pinky")


@dataclass(frozen=True)
class BandPolicy:
    absolute_m: float = 0.001
    relative: float = 0.10
    palm_deg: float = 5.0
    distal_deg: float = 10.0
    frame_atol: float = 1e-5
    minimum_direction_length_m: float = 1e-12

    def __post_init__(self):
        for name in (
            "absolute_m",
            "palm_deg",
            "distal_deg",
            "frame_atol",
            "minimum_direction_length_m",
        ):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(name + " must be finite and positive")
        if not math.isfinite(self.relative) or self.relative < 0:
            raise ValueError("relative must be finite and nonnegative")


DEFAULT_POLICY = BandPolicy()


def _require_finite(value, name):
    if not isinstance(value, torch.Tensor) or not value.is_floating_point():
        raise TypeError(name + " must be a floating Torch tensor")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(name + ": nonfinite inputs require explicit row masking")


def _distance_band(model_distance_m, source_distance_m, policy):
    tau = torch.clamp(policy.relative * source_distance_m, min=policy.absolute_m)
    lo = torch.clamp(source_distance_m - tau, min=0)
    hi = source_distance_m + tau
    outside = torch.clamp(
        torch.maximum(lo - model_distance_m, model_distance_m - hi), min=0
    )
    return dict(
        tolerance_m=tau,
        lower_m=lo,
        upper_m=hi,
        outside_m=outside,
        loss=(outside / tau).square(),
    )


def distance_band_terms(model_distance_m, source_distance_m, policy=DEFAULT_POLICY):
    """Numeric-reference-equivalent distance band, with no exact-equality pull."""
    for value, name in (
        (model_distance_m, "model_distance_m"),
        (source_distance_m, "source_distance_m"),
    ):
        _require_finite(value, name)
        if bool((value < 0).any()):
            raise ValueError(name + " cannot be negative")
    return _distance_band(model_distance_m, source_distance_m, policy)


def ordered_pair_vector_terms(model_vector_m, source_vector_m, policy=DEFAULT_POLICY):
    """Both ordered vectors must use corresponding fixed palm bases.

    The vectors themselves are never normalized, including at zero separation.
    """
    for value, name in (
        (model_vector_m, "model_vector_m"),
        (source_vector_m, "source_vector_m"),
    ):
        _require_finite(value, name)
        if value.shape[-1:] != (3,):
            raise ValueError(name + " must have last dimension 3")
    tau = torch.clamp(
        policy.relative * torch.linalg.vector_norm(source_vector_m, dim=-1),
        min=policy.absolute_m,
    )
    error = torch.linalg.vector_norm(model_vector_m - source_vector_m, dim=-1)
    outside = torch.clamp(error - tau, min=0)
    return dict(
        error_m=error, tolerance_m=tau, outside_m=outside, loss=(outside / tau).square()
    )


def _proper_frames(r, atol):
    eye = torch.eye(3, dtype=r.dtype, device=r.device)
    return ((r.transpose(-1, -2) @ r - eye).abs().amax(dim=(-1, -2)) <= atol) & (
        (torch.linalg.det(r) - 1).abs() <= atol
    )


def palm_local_pair(
    point_a_world_m, point_b_world_m, palm_R_world, policy=DEFAULT_POLICY
):
    for value, name in (
        (point_a_world_m, "point_a_world_m"),
        (point_b_world_m, "point_b_world_m"),
        (palm_R_world, "palm_R_world"),
    ):
        _require_finite(value, name)
    if point_a_world_m.shape[-1:] != (3,) or point_b_world_m.shape[-1:] != (3,):
        raise ValueError("points must have last dimension 3")
    if palm_R_world.shape[-2:] != (3, 3):
        raise ValueError("palm frame must have final dimensions (3,3)")
    if not bool(_proper_frames(palm_R_world, policy.frame_atol).all()):
        raise ValueError("palm frame must be proper orthonormal: no silent reflection")
    return torch.einsum(
        "...ji,...j->...i", palm_R_world, point_b_world_m - point_a_world_m
    )


def _rotation_angle_rad(model_r, source_r):
    r = source_r.transpose(-1, -2) @ model_r
    skew = torch.stack(
        (
            r[..., 2, 1] - r[..., 1, 2],
            r[..., 0, 2] - r[..., 2, 0],
            r[..., 1, 0] - r[..., 0, 1],
        ),
        dim=-1,
    )
    sine = 0.5 * torch.linalg.vector_norm(skew, dim=-1)
    cosine = 0.5 * (r.diagonal(dim1=-2, dim2=-1).sum(-1) - 1)
    return torch.atan2(sine, cosine)


def _direction_angle_rad(model_direction, source_direction):
    # Directions are normalized only after their positive lengths are checked.
    m = model_direction / torch.linalg.vector_norm(
        model_direction, dim=-1, keepdim=True
    )
    s = source_direction / torch.linalg.vector_norm(
        source_direction, dim=-1, keepdim=True
    )
    return torch.atan2(
        torch.linalg.vector_norm(torch.linalg.cross(m, s, dim=-1), dim=-1),
        (m * s).sum(-1),
    )


def _angle_band(angle_rad, tolerance_deg):
    tol = math.radians(tolerance_deg)
    excess = torch.clamp(angle_rad - tol, min=0)
    return (
        angle_rad * (180 / math.pi),
        excess * (180 / math.pi),
        (excess / tol).square(),
    )


def _masked(value, mask):
    while mask.ndim < value.ndim:
        mask = mask.unsqueeze(-1)
    return torch.where(mask, value, torch.full_like(value, float("nan")))


def functional_terms(
    model_chains_m,
    model_wrist_m,
    model_palm_R_world,
    source_chains_m,
    source_wrist_m,
    source_palm_R_world,
    valid_mask,
    pairs=((0, 1),),
    distal_digits=(0, 1),
    distal_valid_mask=None,
    policy=DEFAULT_POLICY,
    palm_relative_digits=(),
):
    """Return row losses and diagnostics without reducing over source time.

    Shapes: chains [B,5,4,3], wrist [B,3], palm [B,3,3], valid_mask [B].
    pairs specifies fixed ordered TIP pairs. distal_digits selects named distal
    chain directions (last joint -> TIP), compared in the declared palm frame.
    Optional distal_valid_mask [B,D] declares actual source support; unsupported
    directions remain NaN and are excluded with reported counts. Every requested
    supported direction must be nondegenerate for the row to be usable.
    palm_relative_digits optionally appends geometric palm center -> selected TIP
    features after the pair features. Center = mean(Index, Middle, Ring node 0);
    it is a named landmark-space proxy, not a skin patch or measured contact.

    loss = mean(pair_distance_loss) + mean(pair_vector_loss) + palm_loss +
           mean(available distal_direction_loss). No scientific weight tuning is
    hidden here; the solver may weight the returned separate row terms explicitly.
    Source targets are detached and always follow each original source row.
    """
    if (
        not isinstance(model_chains_m, torch.Tensor)
        or not model_chains_m.is_floating_point()
    ):
        raise TypeError("model_chains_m must be a floating Torch tensor")
    b = model_chains_m.shape[0]
    inputs = (
        (model_chains_m, (b, 5, 4, 3), "model_chains_m"),
        (model_wrist_m, (b, 3), "model_wrist_m"),
        (model_palm_R_world, (b, 3, 3), "model_palm_R_world"),
        (source_chains_m, (b, 5, 4, 3), "source_chains_m"),
        (source_wrist_m, (b, 3), "source_wrist_m"),
        (source_palm_R_world, (b, 3, 3), "source_palm_R_world"),
    )
    for value, shape, name in inputs:
        if not isinstance(value, torch.Tensor) or value.shape != shape:
            raise ValueError(name + " has incorrect shape; expected " + str(shape))
        if value.dtype != model_chains_m.dtype or value.device != model_chains_m.device:
            raise ValueError(name + " must use the model chains dtype and device")
    if (
        not isinstance(valid_mask, torch.Tensor)
        or valid_mask.shape != (b,)
        or valid_mask.dtype != torch.bool
    ):
        raise ValueError("valid_mask must be an explicit Boolean Torch tensor [B]")
    if valid_mask.device != model_chains_m.device:
        raise ValueError("valid_mask must use the model device")
    pairs = tuple(tuple(p) for p in pairs)
    distal_digits = tuple(distal_digits)
    palm_relative_digits = tuple(palm_relative_digits)
    if (not pairs and not palm_relative_digits) or any(
        len(p) != 2
        or any(type(i) is not int or i not in range(5) for i in p)
        or p[0] == p[1]
        for p in pairs
    ):
        raise ValueError(
            "pairs must contain fixed ordered, distinct digit indices 0..4"
        )
    if len(set(pairs)) != len(pairs):
        raise ValueError("duplicate ordered pairs would silently duplicate weights")
    if any(type(i) is not int or i not in range(5) for i in distal_digits) or len(
        set(distal_digits)
    ) != len(distal_digits):
        raise ValueError("distal_digits must be unique digit indices 0..4")
    if any(
        type(i) is not int or i not in range(5) for i in palm_relative_digits
    ) or len(set(palm_relative_digits)) != len(palm_relative_digits):
        raise ValueError("palm_relative_digits must be unique digit indices 0..4")
    if distal_valid_mask is None:
        distal_valid_mask = torch.ones(
            (b, len(distal_digits)), dtype=torch.bool, device=valid_mask.device
        )
    if (
        not isinstance(distal_valid_mask, torch.Tensor)
        or distal_valid_mask.shape != (b, len(distal_digits))
        or distal_valid_mask.dtype != torch.bool
        or distal_valid_mask.device != valid_mask.device
    ):
        raise ValueError(
            "distal_valid_mask must be Boolean [B,len(distal_digits)] on the model device"
        )
    source_chains_m, source_wrist_m, source_palm_R_world = (
        source_chains_m.detach(),
        source_wrist_m.detach(),
        source_palm_R_world.detach(),
    )
    finite = torch.ones_like(valid_mask)
    for value, _, _ in inputs:
        finite = finite & torch.isfinite(value).flatten(1).all(1)
    initially_valid = valid_mask & finite
    eye = torch.eye(3, dtype=model_chains_m.dtype, device=model_chains_m.device).expand(
        b, 3, 3
    )
    # Entire excluded rows are replaced before any norm, determinant or product.
    mc = torch.where(
        initially_valid[:, None, None, None],
        model_chains_m,
        torch.zeros_like(model_chains_m),
    )
    sc = torch.where(
        initially_valid[:, None, None, None],
        source_chains_m,
        torch.zeros_like(source_chains_m),
    )
    mr = torch.where(initially_valid[:, None, None], model_palm_R_world, eye)
    sr = torch.where(initially_valid[:, None, None], source_palm_R_world, eye)
    frame_valid = _proper_frames(mr, policy.frame_atol) & _proper_frames(
        sr, policy.frame_atol
    )
    frame_valid = frame_valid & finite
    mr = torch.where(frame_valid[:, None, None], mr, eye)
    sr = torch.where(frame_valid[:, None, None], sr, eye)
    md = mc[:, distal_digits, -1] - mc[:, distal_digits, -2]
    sd = sc[:, distal_digits, -1] - sc[:, distal_digits, -2]
    direction_defined = (
        torch.linalg.vector_norm(md, dim=-1) > policy.minimum_direction_length_m
    ) & (torch.linalg.vector_norm(sd, dim=-1) > policy.minimum_direction_length_m)
    valid = (
        initially_valid & frame_valid & (direction_defined | ~distal_valid_mask).all(-1)
    )
    a = [p[0] for p in pairs]
    z = [p[1] for p in pairs]
    mv = torch.einsum("bji,bpj->bpi", mr, mc[:, z, -1] - mc[:, a, -1])
    sv = torch.einsum("bji,bpj->bpi", sr, sc[:, z, -1] - sc[:, a, -1])
    feature_names = tuple(DIGITS[a] + "_TIP->" + DIGITS[z] + "_TIP" for a, z in pairs)
    if palm_relative_digits:
        mcenter = mc[:, 1:4, 0].mean(1)
        scenter = sc[:, 1:4, 0].mean(1)
        mpv = torch.einsum(
            "bji,bpj->bpi", mr, mc[:, palm_relative_digits, -1] - mcenter[:, None]
        )
        spv = torch.einsum(
            "bji,bpj->bpi", sr, sc[:, palm_relative_digits, -1] - scenter[:, None]
        )
        mv = torch.cat((mv, mpv), dim=1)
        sv = torch.cat((sv, spv), dim=1)
        feature_names += tuple(
            "geometric_palm_center->" + DIGITS[d] + "_TIP" for d in palm_relative_digits
        )
    m_dist = torch.linalg.vector_norm(mv, dim=-1)
    s_dist = torch.linalg.vector_norm(sv, dim=-1)
    db = _distance_band(m_dist, s_dist, policy)
    vector_error = torch.linalg.vector_norm(mv - sv, dim=-1)
    vector_outside = torch.clamp(vector_error - db["tolerance_m"], min=0)
    vector_loss = (vector_outside / db["tolerance_m"]).square()
    palm_deg, palm_outside, palm_loss = _angle_band(
        _rotation_angle_rad(mr, sr), policy.palm_deg
    )
    safe_direction = torch.zeros_like(md)
    safe_direction[..., 0] = 1
    direction_active = direction_defined & distal_valid_mask & valid[:, None]
    md = torch.where(direction_active[..., None], md, safe_direction)
    sd = torch.where(direction_active[..., None], sd, safe_direction)
    md = torch.einsum("bji,bdj->bdi", mr, md)
    sd = torch.einsum("bji,bdj->bdi", sr, sd)
    distal_deg, distal_outside, distal_losses = _angle_band(
        _direction_angle_rad(md, sd), policy.distal_deg
    )
    distal_count = direction_active.sum(-1)
    distal_row = torch.where(
        direction_active, distal_losses, torch.zeros_like(distal_losses)
    ).sum(-1) / distal_count.clamp_min(1)
    distance_row = db["loss"].mean(-1)
    vector_row = vector_loss.mean(-1)
    total = distance_row + vector_row + palm_loss + distal_row
    row_values = dict(
        loss=total,
        distance_loss=distance_row,
        vector_loss=vector_row,
        palm_loss=palm_loss,
        distal_loss=distal_row,
        palm_error_deg=palm_deg,
        palm_outside_deg=palm_outside,
        source_distance_m=s_dist,
        model_distance_m=m_dist,
        source_pair_vector_m=sv,
        model_pair_vector_m=mv,
        distance_lower_m=db["lower_m"],
        distance_upper_m=db["upper_m"],
        distance_tolerance_m=db["tolerance_m"],
        distance_outside_m=db["outside_m"],
        pair_distance_loss=db["loss"],
        pair_vector_loss=vector_loss,
        vector_error_m=vector_error,
        vector_outside_m=vector_outside,
    )
    result = {name: _masked(value, valid) for name, value in row_values.items()}
    result.update(
        distal_error_deg=_masked(distal_deg, direction_active),
        distal_outside_deg=_masked(distal_outside, direction_active),
        distal_direction_loss=_masked(distal_losses, direction_active),
        valid_mask=valid,
        requested_valid_mask=valid_mask.clone(),
        finite_mask=finite,
        proper_frame_mask=frame_valid,
        distal_requested_mask=distal_valid_mask.clone(),
        distal_defined_mask=direction_defined & initially_valid[:, None],
        distal_valid_mask=direction_active,
        distal_count=distal_count,
        all_selected_features_valid_mask=valid & distal_valid_mask.all(-1),
        pairs=pairs,
        distal_digits=distal_digits,
        digit_names=DIGITS,
        palm_relative_digits=palm_relative_digits,
        feature_names=feature_names,
        policy=asdict(policy),
    )
    return result


def masked_mean_loss(result, key="loss"):
    """Mean over explicitly valid rows; an empty valid set is an error."""
    valid = result["valid_mask"]
    if not bool(valid.any()):
        raise ValueError(
            "NO_VALID_FUNCTIONAL_ROWS: excluded samples are not zero-loss observations"
        )
    return result[key][valid].mean()
