"""Engineering hand-morphology extension of SMPL-X, phase-I fixed native poses.

Original model buffers are never modified. Geometry, FK and all renderers must
consume the same adapter output. No learned basis or model weights are persisted.
"""

from __future__ import annotations
import json
import numpy as np
import torch
from smplx.lbs import (
    blend_shapes,
    vertices2joints,
    batch_rigid_transform,
    vertices2landmarks,
    find_dynamic_lmk_idx_and_bcoords,
)
from smplx.utils import SMPLXOutput

SCHEMA = "HAND_MORPHOLOGY_PROFILE_V1"
DIGITS = ("thumb", "index", "middle", "ring", "pinky")
JOINTS = {
    "L": {
        "thumb": (37, 38, 39),
        "index": (25, 26, 27),
        "middle": (28, 29, 30),
        "ring": (34, 35, 36),
        "pinky": (31, 32, 33),
    },
    "R": {
        "thumb": (52, 53, 54),
        "index": (40, 41, 42),
        "middle": (43, 44, 45),
        "ring": (49, 50, 51),
        "pinky": (46, 47, 48),
    },
}
WRIST = {"L": 20, "R": 21}
TIP_VERTICES = {
    "L": dict(zip(DIGITS, (5361, 4933, 5058, 5169, 5286))),
    "R": dict(zip(DIGITS, (8079, 7669, 7794, 7905, 8022))),
}
PARAMS = (
    "body_pose",
    "lhand_pose",
    "rhand_pose",
    "jaw_pose",
    "leye_pose",
    "reye_pose",
    "betas",
    "expression",
    "global_orient",
    "transl",
)


def identity_profile(profile_id="identity"):
    return {
        "schema": SCHEMA,
        "profile_id": profile_id,
        "extension": "hand-morphology-extended SMPL-X",
        "digit_length_scale": {d: 1.0 for d in DIGITS},
        "finger_width_scale": 1.0,
        "finger_depth_scale": 1.0,
        "palm_width_scale": 1.0,
        "palm_length_scale": 1.0,
        "units": "metres internally; dimensionless multipliers",
        "anatomical_calibration": "ENGINEERING_CANDIDATE_NOT_PERSON_SKIN_CALIBRATED",
    }


def validate_profile(profile):
    if (
        not isinstance(profile, dict)
        or profile.get("schema") != SCHEMA
        or not profile.get("profile_id")
    ):
        raise ValueError(
            "Custom morphology metadata/schema/profile_id required; native fallback prohibited"
        )
    out = json.loads(json.dumps(profile))
    lengths = out.get("digit_length_scale")
    if not isinstance(lengths, dict) or set(lengths) != set(DIGITS):
        raise ValueError("Exactly five named digit length scales required")
    for k, lo, hi in [
        ("finger_width_scale", 0.7, 1.1),
        ("finger_depth_scale", 0.7, 1.1),
        ("palm_width_scale", 0.9, 1.1),
        ("palm_length_scale", 0.9, 1.1),
    ]:
        v = out.get(k)
        assert isinstance(v, (int, float)) and np.isfinite(v) and lo <= v <= hi, (k, v)
    for d, v in lengths.items():
        assert np.isfinite(v) and 0.85 <= v <= 1.15, (d, v)
    return out


def _unit(v):
    n = np.linalg.norm(v)
    if not np.isfinite(n) or n < 1e-10:
        raise ValueError("Degenerate rest anatomical chart")
    return v / n


def _weight_gradients(v, faces, weights, needed):
    """Least-squares tangential differential of native per-material skin weights.
    Native mesh vertex weights are a piecewise-linear material field; one-ring
    area-normal projection supplies zero normal derivative extension. This is
    not an invented change of the original skinning weights used by LBS.
    """
    adj = [set() for _ in v]
    norm = np.zeros_like(v)
    tri = v[faces]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    for c in range(3):
        np.add.at(norm, faces[:, c], fn)
    for a, b, c in faces:
        adj[a].update((b, c))
        adj[b].update((a, c))
        adj[c].update((a, b))
    result = np.zeros((len(v), weights.shape[1], 3))
    for i in np.flatnonzero(needed):
        nb = np.asarray(sorted(adj[i]), dtype=int)
        n = _unit(norm[i])
        dx = v[nb] - v[i]
        dx = dx - np.outer(dx @ n, n)
        # Normal component is deterministically zero, not regularized into a tilt.
        result[i] = np.linalg.lstsq(dx, weights[nb] - weights[i], rcond=1e-9)[0].T
    return result


class MorphologyAdapter:
    def __init__(self, model, profile, side):
        self.model = model
        self.profile = validate_profile(profile)
        self.side = side.upper()
        if self.side not in JOINTS:
            raise ValueError("side must be L or R")
        if getattr(model, "joint_mapper", None) is not None:
            raise ValueError(
                "Native SMPL-X joint order required; external mapper not supported"
            )
        self.device = model.v_template.device
        self.dtype = model.v_template.dtype
        self.faces = np.asarray(model.faces, dtype=np.int64)
        p = self.profile
        self.is_identity = all(
            v == 1.0 for v in p["digit_length_scale"].values()
        ) and all(
            p[k] == 1.0
            for k in (
                "finger_width_scale",
                "finger_depth_scale",
                "palm_width_scale",
                "palm_length_scale",
            )
        )
        v = model.v_template.detach().cpu().numpy().astype(np.float64)
        j = (
            (model.J_regressor @ model.v_template)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )
        allw = model.lbs_weights.detach().cpu().numpy().astype(np.float64)
        wrist = j[WRIST[self.side]]
        jj = JOINTS[self.side]
        handids = np.asarray([i for d in DIGITS for i in jj[d]], int)
        w = allw[:, handids]
        # Proper palm chart: x toward index from little; y wrist to middle,
        # z=x cross y. Mirroring is geometric; no source/pose rotation is added.
        y = _unit(j[jj["middle"][0]] - wrist)
        x = j[jj["index"][0]] - j[jj["pinky"][0]]
        x = _unit(x - y * (x @ y))
        z = _unit(np.cross(x, y))
        P = np.column_stack((x, y, z))
        palm_len = float(np.linalg.norm(j[jj["middle"][0]] - wrist))
        transition = 0.20 * palm_len
        u = np.clip((v - wrist) @ y / transition, 0, 1)
        t = u * u * (3 - 2 * u)
        gt = (6 * u * (1 - u) / transition)[:, None] * y
        digit_support = w.sum(1) > 0
        palm_support = (
            (allw[:, WRIST[self.side]] + w.sum(1) > 0.5)
            & (((v - wrist) @ y) > 0)
            & (np.linalg.norm(v - wrist, axis=1) < 0.26)
        )
        support = digit_support | palm_support
        t[~support] = 0
        gt[~support] = 0
        # All vertices affected by changed finger FK remain inside support.
        assert np.all(support[w.sum(1) > 0])
        Ap = (
            np.eye(3)
            if p["palm_width_scale"] == 1.0 and p["palm_length_scale"] == 1.0
            else P @ np.diag([p["palm_width_scale"], p["palm_length_scale"], 1.0]) @ P.T
        )
        jnew = j.copy()
        mats = []
        shifts = []
        for d in DIGITS:
            ids = jj[d]
            base = j[ids[0]]
            bnew = wrist + Ap @ (base - wrist)
            length = p["digit_length_scale"][d]
            for ji in ids:
                jnew[ji] = (
                    j[ji]
                    if length == 1.0 and np.array_equal(Ap, np.eye(3))
                    else bnew + length * (j[ji] - base)
                )
            chain = np.vstack((j[list(ids)], v[TIP_VERTICES[self.side][d]]))
            for k, ji in enumerate(ids):
                longitudinal = _unit(chain[k + 1] - chain[k])
                depth = z - longitudinal * (z @ longitudinal)
                depth = _unit(depth)
                lateral = _unit(np.cross(longitudinal, depth))
                depth = np.cross(lateral, longitudinal)
                F = np.column_stack((lateral, longitudinal, depth))
                A = (
                    F
                    @ np.diag(
                        [p["finger_width_scale"], length, p["finger_depth_scale"]]
                    )
                    @ F.T
                )
                mats.append(A)
                shifts.append(jnew[ji] - j[ji])
        mats = np.asarray(mats)
        shifts = np.asarray(shifts)
        I = np.eye(3)
        # Affine chart displacement around each joint; native vertex skin weights
        # blend charts. Remaining weight belongs to the palm chart.
        palm_disp = (v - wrist) @ (Ap - I).T
        joint_disp = (
            np.einsum("vjk,jlk->vjl", v[:, None, :] - j[handids][None], mats - I)
            + shifts[None]
        )
        d = palm_disp + np.einsum("vj,vjk->vk", w, joint_disp - palm_disp[:, None, :])
        grad = (
            np.zeros((len(v), 15, 3))
            if self.is_identity
            else _weight_gradients(v, self.faces, w, support)
        )
        differential = (
            (Ap - I)[None]
            + np.einsum("vj,jkl->vkl", w, mats - Ap)
            + np.einsum("vjk,vjl->vkl", joint_disp - palm_disp[:, None, :], grad)
        )
        D = I[None] + t[:, None, None] * differential + np.einsum("vi,vj->vij", d, gt)
        vnew = v + t[:, None] * d
        vnew[~support] = v[~support]
        D[~support] = I
        if self.is_identity:
            vnew = v.copy()
            jnew = j.copy()
            D = np.broadcast_to(I, (len(v), 3, 3)).copy()
        self.rest_vertices = torch.as_tensor(vnew, device=self.device, dtype=self.dtype)
        self.rest_joints = torch.as_tensor(jnew, device=self.device, dtype=self.dtype)
        self.jacobian = torch.as_tensor(D, device=self.device, dtype=self.dtype)
        self.support_mask = support
        self.native_rest_joints = torch.as_tensor(
            j, device=self.device, dtype=self.dtype
        )

    def forward(self, params):
        missing = set(PARAMS) - set(params)
        if missing:
            raise ValueError("Missing saved native parameters: " + str(sorted(missing)))
        p = {
            k: torch.as_tensor(params[k], device=self.device, dtype=self.dtype)
            for k in PARAMS
        }
        aa = torch.cat(
            [
                p["global_orient"][:, None],
                p["body_pose"],
                p["jaw_pose"][:, None],
                p["leye_pose"][:, None],
                p["reye_pose"][:, None],
                p["lhand_pose"],
                p["rhand_pose"],
            ],
            1,
        )
        z = torch.zeros_like(aa[..., 0])
        a, b, c = aa.unbind(-1)
        skew = torch.stack((z, -c, b, c, z, -a, -b, a, z), -1).reshape(
            *aa.shape[:-1], 3, 3
        )
        R = torch.matrix_exp(skew)
        return self.forward_rotations(
            betas=p["betas"],
            expression=p["expression"],
            global_orient=R[:, 0],
            body_pose=R[:, 1:22],
            jaw_pose=R[:, 22],
            leye_pose=R[:, 23],
            reye_pose=R[:, 24],
            left_hand_pose=R[:, 25:40],
            right_hand_pose=R[:, 40:55],
            transl=p["transl"],
        )

    def forward_rotations(self, **kwargs):
        m = self.model
        device = self.device
        dtype = self.dtype
        B = max(
            [len(v) for v in kwargs.values() if isinstance(v, torch.Tensor) and v.ndim]
            + [1]
        )
        eye = torch.eye(3, device=device, dtype=dtype)
        rot = (
            lambda name, n: kwargs.get(name, eye.view(1, 1, 3, 3).expand(B, n, 3, 3))
            .reshape(-1, n, 3, 3)
            .expand(B, -1, -1, -1)
        )
        betas = kwargs.get(
            "betas", torch.zeros(B, m.num_betas, device=device, dtype=dtype)
        )
        if torch.any(betas != 0):
            raise ValueError(
                "This fixed-neutral morphology adapter requires betas=0; shape fitting is not authorized"
            )
        expression = kwargs.get(
            "expression",
            torch.zeros(B, m.num_expression_coeffs, device=device, dtype=dtype),
        )
        transl = kwargs.get("transl", torch.zeros(B, 3, device=device, dtype=dtype))
        full = torch.cat(
            [
                rot("global_orient", 1),
                rot("body_pose", 21),
                rot("jaw_pose", 1),
                rot("leye_pose", 1),
                rot("reye_pose", 1),
                rot("left_hand_pose", 15),
                rot("right_hand_pose", 15),
            ],
            1,
        )
        global_rot = [full[:, 0]]
        for ji in range(1, 55):
            global_rot.append(global_rot[int(m.parents[ji])] @ full[:, ji])
        global_rot = torch.stack(global_rot, 1)
        if self.is_identity:
            out = m(**kwargs)
            out.global_joint_rotations = global_rot
            return out
        # Native expression coefficients remain unchanged; hand-supported
        # displacement gets the same differential as native pose corrections.
        native_shaped = m.v_template + blend_shapes(
            torch.cat([betas, expression], -1),
            torch.cat([m.shapedirs, m.expr_dirs], -1),
        )
        native_J = vertices2joints(m.J_regressor, native_shaped)
        J = native_J + (self.rest_joints - self.native_rest_joints)[None]
        shape_delta = native_shaped - m.v_template
        shaped = self.rest_vertices[None] + torch.einsum(
            "vij,bvj->bvi", self.jacobian, shape_delta
        )
        pose_offsets = ((full[:, 1:] - eye).reshape(B, -1) @ m.posedirs).reshape(
            B, -1, 3
        )
        posed = shaped + torch.einsum("vij,bvj->bvi", self.jacobian, pose_offsets)
        joints, A = batch_rigid_transform(full, J, m.parents, dtype=dtype)
        T = (m.lbs_weights[None].expand(B, -1, -1) @ A.reshape(B, -1, 16)).reshape(
            B, -1, 4, 4
        )
        vh = torch.cat(
            [
                posed,
                torch.ones(B, len(self.rest_vertices), 1, device=device, dtype=dtype),
            ],
            -1,
        )
        vertices = (T @ vh[..., None])[:, :, :3, 0]
        lf = m.lmk_faces_idx[None].expand(B, -1).contiguous()
        lb = m.lmk_bary_coords[None].repeat(B, 1, 1)
        if m.use_face_contour:
            df, db = find_dynamic_lmk_idx_and_bcoords(
                vertices,
                full,
                m.dynamic_lmk_faces_idx,
                m.dynamic_lmk_bary_coords,
                m.neck_kin_chain,
                pose2rot=False,
            )
            lf = torch.cat([lf, df], 1)
            lb = torch.cat([lb, db], 1)
        landmarks = vertices2landmarks(vertices, m.faces_tensor, lf, lb)
        joints = torch.cat([m.vertex_joint_selector(vertices, joints), landmarks], 1)
        vertices = vertices + transl[:, None]
        joints = joints + transl[:, None]
        out = SMPLXOutput(
            vertices=vertices,
            joints=joints,
            betas=betas,
            expression=expression,
            global_orient=full[:, 0],
            body_pose=full[:, 1:22],
            left_hand_pose=full[:, 25:40],
            right_hand_pose=full[:, 40:55],
            jaw_pose=full[:, 22],
            transl=transl,
            full_pose=full if kwargs.get("return_full_pose") else None,
        )

        out.global_joint_rotations = global_rot
        return out
