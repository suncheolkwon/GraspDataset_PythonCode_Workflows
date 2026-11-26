"""Global rigid alignment followed by body-only fitting in overlapping native windows."""

import numpy as np
import torch
from body_lbfgs import LBFGS
from runtime import config, PARAMS
from reset_rotation import aa2rot_torch


def empty_parameters(n, device):
    shapes = {
        "body_pose": (21, 3),
        "lhand_pose": (15, 3),
        "rhand_pose": (15, 3),
        "betas": (10,),
        "expression": (10,),
    }
    return {
        k: torch.zeros((n, *shapes.get(k, (3,))), device=device, dtype=torch.float32)
        for k in PARAMS
    }


class JointMapper(torch.nn.Module):
    def __init__(self, indices):
        super().__init__()
        self.register_buffer("indices", indices)

    def forward(self, joints, **kwargs):
        return torch.index_select(joints, 1, self.indices)


def fit_window(model, kp, device):
    n = len(kp)
    p = empty_parameters(n, device)
    # The student's canonical ordering puts the opposite and measured wrists in slots 11 and 35.
    body_slots = list(range(12)) + list(range(27, 36))
    for variables in [
        ("global_orient", "transl"),
        ("global_orient", "transl", "body_pose"),
    ]:
        params = [p[k].requires_grad_(True) for k in variables]
        opt = LBFGS(params, line_search_fn="strong_wolfe", max_iter=30)

        def closure():
            opt.zero_grad()
            axis_angle = torch.cat(
                [
                    p["global_orient"][:, None],
                    p["body_pose"],
                    p["leye_pose"][:, None],
                    p["reye_pose"][:, None],
                    p["jaw_pose"][:, None],
                    p["lhand_pose"],
                    p["rhand_pose"],
                ],
                dim=1,
            )
            rotations = aa2rot_torch(axis_angle.reshape(-1, 3)).reshape(n, 55, 3, 3)
            joints = model(
                betas=p["betas"],
                expression=p["expression"],
                global_orient=rotations[:, 0],
                body_pose=rotations[:, 1:22],
                leye_pose=rotations[:, 22],
                reye_pose=rotations[:, 23],
                jaw_pose=rotations[:, 24],
                left_hand_pose=rotations[:, 25:40],
                right_hand_pose=rotations[:, 40:55],
                transl=p["transl"],
            ).joints
            pred = joints[:, body_slots, :3]
            loss_kp = (
                torch.sum(
                    ((kp[:, body_slots, :3] - pred) * kp[:, body_slots, 3][..., None])
                    ** 2
                )
                / n
            )
            pose = p["body_pose"]
            loss_reg = torch.sum(pose**2) / n
            if n < 3:
                smooth_pose = pose.sum() * 0
                smooth_body = joints.sum() * 0
            else:
                target = pose.clone().detach()
                target[1:-1] = (target[1:-1] + target[:-2] + target[2:]) / 3
                smooth_pose = torch.sum((pose[1:-1] - target[1:-1]) ** 2) / (n - 2)
                smooth_joints = joints[:, body_slots]
                target = smooth_joints.clone().detach()
                target[1:-1] = (target[:-2] + target[2:]) / 2
                smooth_body = torch.sum((smooth_joints[1:-1] - target[1:-1]) ** 2) / (
                    n - 2
                )
            loss = sum(
                [loss_kp * 1.0, loss_reg * 0.01, smooth_pose * 0.1, smooth_body * 0.5]
            )
            loss.backward()
            return loss

        previous = None
        for _ in range(50):
            loss = opt.step(closure).item()
            if not np.isfinite(loss):
                raise ValueError("Nonfinite seed objective")
            if (
                previous is not None
                and (previous - loss) / max(abs(previous), abs(loss), 1) <= 1e-9
            ):
                break
            previous = loss
        for v in params:
            v.requires_grad_(False)
    return {k: v.detach().cpu().numpy() for k, v in p.items()}


def fit(model, source, segments, side, device):
    pos = source["canonical_positions_m"]
    confidence = np.isfinite(pos).all(axis=-1)
    kp = np.concatenate(
        [np.where(confidence[..., None], pos, 0), confidence[..., None]], axis=-1
    )
    mapping = torch.tensor(
        config("mapping.json")[side]["model"], device=device, dtype=torch.long
    )
    model.joint_mapper = JointMapper(mapping)
    out = {
        k: np.full((len(pos), *v.shape[1:]), np.nan, np.float32)
        for k, v in empty_parameters(1, device).items()
    }
    for lo, hi in segments:
        starts = list(range(lo, max(lo + 1, hi - 128 + 1), 112))
        if starts[-1] + 128 < hi:
            starts.append(max(lo, hi - 128))
        windows = [(i, min(hi, i + 128)) for i in sorted(set(starts))]
        centers = np.array([(a + b - 1) / 2 for a, b in windows])
        owner = np.argmin(abs(np.arange(lo, hi)[:, None] - centers[None]), axis=1)
        for wi, (a, b) in enumerate(windows):
            print(f"Body seed window {wi + 1}/{len(windows)}", flush=True)
            solved = fit_window(
                model, torch.tensor(kp[a:b], device=device, dtype=torch.float32), device
            )
            indices = lo + np.flatnonzero(owner == wi)
            for k in PARAMS:
                out[k][indices] = solved[k][indices - a]
    model.joint_mapper = None
    return out
