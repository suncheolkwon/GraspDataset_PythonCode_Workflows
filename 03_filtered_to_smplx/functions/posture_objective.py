"""C01 supported geometric posture targets; no model load or anatomical inference.

Source/model coordinates share their unchanged world axes and metres. Palm
columns are geometric frame-to-world axes. A forearm expressed as P.T @ unit(v)
constrains a single long-axis relationship, not calibrated anatomical twist.
All target masks retain the original input validity/coverage denominator.
"""

import numpy as np
import torch
import reset_evaluation as e
from metrics_v2 import components, frame_condition
import torch_geometry as g

BODY_POSE_INDICES = {"L": (15, 17, 19), "R": (16, 18, 20)}
MODEL_ARM_JOINTS = {"L": (16, 18, 20), "R": (17, 19, 21)}
SOURCE_ARM_POINTS = {"L": (6, 7, 8), "R": (10, 11, 12)}
PALM_CONDITION_MIN = 0.1
FINGER_RELATIVE_FRACTION = 0.15
BOOL_KEYS = {"palm_mask", "angle_mask", "link_mask", "arm_mask", "palm_forearm_mask"}


def source_targets(wrist, chains, body, side):
    if side not in BODY_POSE_INDICES:
        raise ValueError("Unknown measured side")
    palm = e.palm_frame(wrist, chains)
    angle, conditioning = components(chains, palm, side)
    palm_mask = np.isfinite(palm).all((1, 2)) & (
        frame_condition(wrist, chains) >= PALM_CONDITION_MIN
    )
    directions = e.unit(np.diff(chains, axis=2))
    local_links = np.einsum("tji,tdkj->tdki", palm, directions)
    # A stable-looking local angle is not usable when its parent palm chart is
    # unstable. This masks target components, never removes source/output rows.
    angle_mask = np.isfinite(angle) & palm_mask[:, None, None]
    link_mask = np.isfinite(local_links).all(-1) & palm_mask[:, None, None]
    arm = e.unit(np.diff(np.asarray(body)[:, SOURCE_ARM_POINTS[side]], axis=1))
    arm_mask = np.isfinite(arm).all(-1)
    palm_forearm = np.einsum("tji,tj->ti", palm, arm[:, 1])
    pf_mask = palm_mask & arm_mask[:, 1] & np.isfinite(palm_forearm).all(-1)
    return dict(
        palm=palm,
        angles_rad=np.deg2rad(angle),
        local_links=local_links,
        arm_axes=arm,
        palm_forearm=palm_forearm,
        palm_mask=palm_mask,
        angle_mask=angle_mask,
        link_mask=link_mask,
        arm_mask=arm_mask,
        palm_forearm_mask=pf_mask,
        angle_condition=conditioning,
    )


def target_tensors(targets, start=0, end=None, *, device="cpu", dtype=torch.float64):
    return {
        key: torch.as_tensor(
            value[start:end]
            if key in BOOL_KEYS
            else np.nan_to_num(value[start:end], nan=0.0, posinf=0.0, neginf=0.0),
            device=device,
            dtype=torch.bool if key in BOOL_KEYS else dtype,
        )
        for key, value in targets.items()
    }


def arm_axes(joints, side):
    points = joints[:, MODEL_ARM_JOINTS[side]]
    return g.unit(points[:, 1:] - points[:, :-1])


def wrapped_difference(model, source):
    difference = model - source
    return torch.atan2(torch.sin(difference), torch.cos(difference))


def relative_bound_excess(model, source, fraction=FINGER_RELATIVE_FRACTION):
    # No absolute tolerance floor and no division by a small source angle.
    return torch.relu(wrapped_difference(model, source).abs() - fraction * source.abs())


def _mean(values, mask, zero):
    return values[mask].mean() if bool(mask.any()) else zero


def losses(chains, palm, arms, side, target):
    """Active losses independent of model internals, testable on exact geometry."""
    angles, quality = g.angles(chains, palm, side)
    directions = g.unit(chains[:, :, 1:] - chains[:, :, :-1])
    local = torch.einsum("tji,tdkj->tdki", palm, directions)
    zero = chains.sum() * 0 + arms.sum() * 0
    am, lm, pm = target["angle_mask"], target["link_mask"], target["palm_mask"]
    palm_loss = _mean((palm - target["palm"]).square().sum((1, 2)) / 2, pm, zero)
    link_loss = _mean((local - target["local_links"]).square().sum(-1), lm, zero)
    angle_loss = _mean(2 - 2 * torch.cos(angles - target["angles_rad"]), am, zero)
    chart_loss = _mean(
        (torch.relu(PALM_CONDITION_MIN - quality) / PALM_CONDITION_MIN).square(),
        am,
        zero,
    )
    arm_loss = _mean(
        (arms - target["arm_axes"]).square().sum(-1), target["arm_mask"], zero
    )
    relative_forearm = torch.einsum("tji,tj->ti", palm, arms[:, 1])
    palm_forearm_loss = _mean(
        (relative_forearm - target["palm_forearm"]).square().sum(-1),
        target["palm_forearm_mask"],
        zero,
    )
    # Worst defined component per finger: another easy component cannot dilute
    # a bad component. Equal known finger-frame observations then contribute.
    excess = relative_bound_excess(angles, target["angles_rad"])
    per_finger = torch.where(am, excess.square(), torch.zeros_like(excess)).amax(-1)
    bounds_loss = _mean(per_finger, am.any(-1), zero)
    return dict(
        palm=palm_loss,
        link_directions=link_loss,
        defined_angles=angle_loss,
        chart_condition=chart_loss,
        arm_axes=arm_loss,
        palm_forearm=palm_forearm_loss,
        relative_angle_bounds=bounds_loss,
    )
