"""Seated free-root H1 inverse kinematics in the measured world frame."""

import numpy as np
import mujoco
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from .model import MODEL, SIDES


def unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def basis(wrist, mcps):
    radial = unit(mcps[..., 0, :] - mcps[..., 3, :])
    distal = mcps.mean(axis=-2) - wrist
    distal = unit(distal - radial * np.sum(distal * radial, axis=-1, keepdims=True))
    return np.stack([radial, distal, np.cross(radial, distal)], axis=-1)


def body_frames(b):
    hips = (b["LThigh"] + b["RThigh"]) / 2
    shoulder = (b["LUArm"] + b["RUArm"]) / 2
    up = unit(shoulder - hips)
    lateral = b["LThigh"] - b["RThigh"]
    forward = unit(np.cross(lateral, up))
    lateral = unit(np.cross(up, forward))
    matrix = np.stack([forward, lateral, up], axis=-1)
    # The H1 torso has yaw only. The floating base therefore carries trunk lean.
    shoulder_left = unit(b["LUArm"] - b["RUArm"])
    yaw = np.arctan2(-np.sum(shoulder_left * forward, axis=1), np.sum(shoulder_left * lateral, axis=1))
    return matrix, hips, yaw


def angle(a, b):
    return np.rad2deg(np.arctan2(np.linalg.norm(np.cross(a, b), axis=-1), np.sum(a * b, axis=-1)))


def rotation_error(a, b):
    return np.rad2deg(Rotation.from_matrix(a @ np.swapaxes(b, -2, -1)).magnitude())


def summarize(x):
    return {
        k: float(f(x))
        for k, f in [
            ("mean", np.mean),
            ("median", np.median),
            ("p95", lambda v: np.percentile(v, 95)),
            ("max", np.max),
            ("min", np.min),
        ]
    }


class Kinematics:
    def __init__(self, active):
        self.active = active
        self.m = mujoco.MjModel.from_xml_path(str(MODEL))
        self.d = mujoco.MjData(self.m)
        self.q_neutral = self.m.qpos0.copy()
        self.q_neutral[:3] = 0
        self.q_neutral[3:7] = [1, 0, 0, 0]
        self.setq(self.q_neutral)
        self.local_palm = []
        for p in ["lh", "rh"]:
            mcp = np.array(
                [self.d.xanchor[self.m.joint(f"{p}_{s}J4").id] for s in ["FF", "MF", "RF", "LF"]]
            )
            palm = self.m.body(p + "_palm").id
            self.local_palm.append(self.d.xmat[palm].reshape(3, 3).T @ basis(self.d.xpos[palm], mcp))

    def setq(self, q):
        self.d.qpos[:] = q
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_comPos(self.m, self.d)

    def hinge(self, n):
        return self.m.joint(n).id

    def body(self, n):
        return self.m.body(n).id

    def point(self, id, offset=None):
        return (
            self.d.xpos[id]
            if offset is None
            else self.d.xpos[id] + self.d.xmat[id].reshape(3, 3) @ offset
        )

    def jac(self, id, cols, p=None):
        jp = np.empty((3, self.m.nv))
        jr = np.empty_like(jp)
        if p is None:
            mujoco.mj_jacBody(self.m, self.d, jp, jr, id)
        else:
            mujoco.mj_jac(self.m, self.d, jp, jr, p, id)
        return jp[:, cols], jr[:, cols]


class Limb:
    def __init__(self, kin, side, arm):
        self.kin = kin
        self.side = side
        self.arm = arm
        self.which = SIDES.index(side)
        self.branch = "negative"
        p = ("lh", "rh")[self.which]
        kin.setq(kin.q_neutral)
        if arm:
            names = [
                side + "_" + j
                for j in ["shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_yaw"]
            ] + [p + "_WRJ2", p + "_WRJ1"]
            ids = [side + "_shoulder_roll_link", side + "_elbow_link", p + "_palm"]
        else:
            names = [side + "_" + j for j in ["hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle"]]
            ids = [side + "_hip_pitch_link", side + "_knee_link", side + "_ankle_link"]
        self.ids = [kin.body(j) for j in ids]
        self.js = [kin.hinge(j) for j in names]
        self.qa = kin.m.jnt_qposadr[self.js]
        self.va = kin.m.jnt_dofadr[self.js]
        self.lo, self.hi = kin.m.jnt_range[self.js].T
        pos = kin.d.xpos[self.ids]
        self.l1, self.l2 = np.linalg.norm(np.diff(pos, axis=0), axis=1)
        self.localfoot = np.array([0.15, 0, -0.04])

    def geometry(self):
        k = self.kin
        a, b, c = k.d.xpos[self.ids]
        dir0 = unit(b - a)
        dir1 = unit(c - b)
        if self.arm:
            orient = k.d.xmat[self.ids[2]].reshape(3, 3) @ k.local_palm[self.which]
        else:
            orient = unit(k.d.xmat[self.ids[2]].reshape(3, 3) @ self.localfoot)
        return a, b, c, dir0, dir1, orient

    def objective(self, base, directions, palm_or_foot, prev):
        k = self.kin
        I = np.eye(len(self.js))
        xlast = None
        cached = None

        def evaluate(x):
            nonlocal xlast, cached
            if xlast is not None and np.array_equal(x, xlast):
                return cached
            q = base.copy()
            q[self.qa] = x
            k.setq(q)
            a, b, c, d0, d1, ori = self.geometry()
            ja, jb, jc = [k.jac(i, self.va) for i in self.ids]
            j0 = ((np.eye(3) - np.outer(d0, d0)) @ (jb[0] - ja[0])) / np.linalg.norm(b - a)
            j1 = ((np.eye(3) - np.outer(d1, d1)) @ (jc[0] - jb[0])) / np.linalg.norm(c - b)
            # Angular errors enter as chord distances between unit directions.
            # Active palm is primary; orients both radix and normal in source axes.
            wt0, wt1 = (1.5, 1.5) if self.arm else (2.5, 2.5)
            r = [wt0 * (d0 - directions[0]), wt1 * (d1 - directions[1])]
            j = [wt0 * j0, wt1 * j1]
            if self.arm:
                weights = (
                    np.array([2.8, 2.8, 5.0]) if self.side == k.active else np.array([1.0, 1.0, 1.3])
                )
                radial_j = np.stack([np.cross(jc[1].T, ori[:, col]).T for col in range(3)], axis=1)
                r.extend(
                    [
                        ((ori - palm_or_foot) * weights).ravel(),
                        2.5 * ((c - a) - (self.l1 * directions[0] + self.l2 * directions[1])),
                    ]
                )
                j.extend([(radial_j * weights[None, :, None]).reshape(9, -1), 2.5 * (jc[0] - ja[0])])
            else:
                r.append(1.3 * (ori - palm_or_foot))
                j.append(1.3 * np.cross(jc[1].T, ori).T)
            # Weak minimum-change prior to choose a continuous solution when underdetermined.
            if prev is not None:
                r.append(0.12 * (x - prev))
                j.append(0.12 * I)
            r.append(0.008 * x)
            j.append(0.008 * I)
            cached = np.concatenate(r), np.concatenate(j)
            xlast = x.copy()
            return cached

        return evaluate

    def solve(self, base, directions, orientation, prev):
        k = self.kin
        obj = self.objective(base, directions, orientation, prev)
        if prev is None:
            if self.arm:
                sgn = 1 if self.which == 0 else -1
                seeds = [
                    np.zeros(7),
                    [-1, 0.2 * sgn, 0, 0.5, 0.4, 0, 0],
                    [-0.6, 0.45 * sgn, -0.75 * sgn, 0.6, 0.1, 0, -0.1],
                    [-1, 0.1 * sgn, 1.6 * sgn, -0.4, 0.8, -0.15, 0.15],
                ]
                if self.side == k.active and self.branch == "negative":
                    # Full-motion screening finds a continuous shoulder-yaw solution through
                    # the grasp only on this periodic branch (all mechanical bounds retained).
                    seeds = np.array(
                        [
                            [-1.8, -0.62, -4.35, 1.1, 1.42, -0.50, 0],
                            [-1.2, -0.4, -4.0, 1.3, 1.35, -0.4, 0.12],
                        ]
                    )
                    if self.side == "left":
                        seeds *= np.array([1, -1, -1, 1, -1, -1, 1])
            else:
                seeds = [np.array([0, 0, -1.25, 1.35, -0.50]), np.zeros(5)]
        else:
            seeds = [prev]
        best = None
        for initial in seeds:
            x0 = np.clip(initial, self.lo + 1e-8, self.hi - 1e-8)
            sol = least_squares(
                lambda x: obj(x)[0],
                x0,
                jac=lambda x: obj(x)[1],
                bounds=(self.lo, self.hi),
                max_nfev=110 if prev is None else 32,
                ftol=1e-7,
                xtol=1e-7,
                gtol=1e-7,
            )
            if best is None or best.cost > sol.cost:
                best = sol
        base[self.qa] = best.x
        return best.x, int(best.status)


def screen_seed(limb, q, directions, palms):
    """Screen forward and reverse initializations on the measured motion."""
    ix = np.unique(np.r_[np.arange(0, len(q), 28), len(q) - 1])
    choices = []
    for branch in ("negative", "positive"):
        limb.branch = branch
        for reverse in (False, True):
            samples = q[ix].copy()
            errors = []
            prev = None
            for j in reversed(range(len(ix))) if reverse else range(len(ix)):
                i = ix[j]
                prev, _ = limb.solve(samples[j], directions[i], palms[i], prev)
                limb.kin.setq(samples[j])
                _, _, _, u, v, rot = limb.geometry()
                errors.append(
                    [
                        angle(rot[:, 2], palms[i, :, 2]),
                        rotation_error(rot, palms[i]),
                        angle(u, directions[i, 0]),
                        angle(v, directions[i, 1]),
                    ]
                )
            error = np.array(errors) @ np.array([1.0, 0.35, 0.13, 0.13])
            jumps = np.degrees(np.abs(np.diff(samples[:, limb.qa], axis=0)).max(axis=1))
            score = (
                error.mean()
                + 0.4 * np.percentile(error, 95)
                + 0.12 * error.max()
                + 0.2 * np.maximum(jumps - 50, 0).mean()
            )
            choices.append((score, samples[0, limb.qa].copy()))
    return min(choices, key=lambda x: x[0])[1]


def fit_body(body, wrist, chains, times, active, trial, number):
    k = Kinematics(active)
    torso, hip, yaw = body_frames(body)
    center = (k.d.xpos[k.body("left_hip_pitch_link")] + k.d.xpos[k.body("right_hip_pitch_link")]) / 2
    q = np.tile(k.q_neutral, (len(times), 1))
    q[:, :3] = hip - np.einsum("nij,j->ni", torso, center)
    q[:, 3:7] = Rotation.from_matrix(torso).as_quat()[:, [3, 0, 1, 2]]
    q[:, k.m.jnt_qposadr[k.hinge("torso")]] = yaw
    targets = {"source_time_s": times}
    for arm in [False, True]:
        for side in SIDES:
            limb = Limb(k, side, arm)
            s = side[0].upper()
            names = ("UArm", "FArm", "Hand") if arm else ("Thigh", "Shin", "Foot")
            a, b, c = [body[s + n] for n in names]
            dirs = np.stack((unit(b - a), unit(c - b)), axis=1)
            if arm and side == active:
                ori = basis(wrist, chains[:, 1:, 0])
            elif arm:
                ori = basis(
                    c, np.stack([body[f"{s}{d}1"] for d in ("Index", "Middle", "Ring", "Pinky")], axis=1)
                )
            else:
                ori = unit(body[s + "Toe"] - c)
            if arm:
                targets[side + "_arm_source_dirs"] = dirs
                targets[side + "_arm_source_palm_basis"] = ori
            prev = None
            if arm and side == active and trial != "Tripod_R_1":
                prev = screen_seed(limb, q, dirs, ori)
            for i, pose in enumerate(q):
                prev, _ = limb.solve(pose, dirs[i], ori[i], prev)
    # A single constant shift sets the lowest sole point to zero and preserves foot motion.
    soles = []
    for side in SIDES:
        bid = k.body(side + "_ankle_link")
        gid = int(np.flatnonzero((k.m.geom_bodyid == bid) & (k.m.geom_group == 3))[0])
        mid = k.m.geom_dataid[gid]
        vertices = k.m.mesh_vert[k.m.mesh_vertadr[mid] : k.m.mesh_vertadr[mid] + k.m.mesh_vertnum[mid]]
        for pose in q:
            k.setq(pose)
            soles.append(k.d.geom_xpos[gid, 2] + np.min(vertices @ k.d.geom_xmat[gid].reshape(3, 3)[2]))
    q[:, 2] -= np.min(soles)
    if trial != "Tripod_R_1":
        from .refine_body import refine

        q = refine(k, q, targets, number)
    return q
