"""Reduce crossings of the opposing arm and collisions between the hip shells."""

import numpy as np
import mujoco
from scipy.interpolate import RectBivariateSpline
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from .model import MODEL, SIDES


class PassiveArm:
    def __init__(self, active):
        passive = "left" if active == "right" else "right"
        prefix = "lh_" if passive == "left" else "rh_"
        self.passive = passive
        self.sign = 1 if passive == "left" else -1
        self.m = m = mujoco.MjModel.from_xml_path(str(MODEL))
        self.d = mujoco.MjData(m)
        hinges = [
            passive + "_shoulder_pitch",
            passive + "_shoulder_roll",
            passive + "_shoulder_yaw",
            passive + "_elbow",
            passive + "_wrist_yaw",
            prefix + "WRJ2",
            prefix + "WRJ1",
        ]
        ids = [m.joint(n).id for n in hinges]
        self.addr = m.jnt_qposadr[ids]
        self.dof = m.jnt_dofadr[ids]
        self.limits = m.jnt_range[ids]
        self.palm = m.body(prefix + "palm").id
        self.elbow = m.body(passive + "_elbow_link").id
        self.torso = m.body("torso_link").id
        self.jacp = np.zeros((3, m.nv))
        self.jacr = np.zeros((3, m.nv))
        names = [m.body(int(b)).name for b in m.geom_bodyid]
        self.left = np.array(
            [n.startswith(("lh_", "left_elbow", "left_hand", "left_wrist")) for n in names]
        )
        self.right = np.array(
            [n.startswith(("rh_", "right_elbow", "right_hand", "right_wrist")) for n in names]
        )

    def pose(self, q):
        self.d.qpos[:] = q
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_comPos(self.m, self.d)

    def cross(self):
        mujoco.mj_collision(self.m, self.d)
        return max(
            [0.0]
            + [
                -1000 * c.dist
                for c in self.d.contact
                if (self.left[c.geom1] and self.right[c.geom2])
                or (self.right[c.geom1] and self.left[c.geom2])
            ]
        )

    def jac(self, b):
        mujoco.mj_jacBody(self.m, self.d, self.jacp, self.jacr, b)
        return self.jacp[:, self.dof], self.jacr[:, self.dof]

    def move(self, qref, local_delta):
        self.pose(qref)
        d = self.d
        goal = d.xpos[self.palm].copy() + d.xmat[self.torso].reshape(3, 3) @ local_delta
        egoal = d.xpos[self.elbow].copy()
        ori = d.xmat[self.palm].reshape(3, 3).copy()
        q = qref.copy()
        qr = qref[self.addr]
        eye = np.eye(len(qr))
        for _ in range(9):
            rot = Rotation.from_matrix(ori @ d.xmat[self.palm].reshape(3, 3).T).as_rotvec()
            Jp, Jr = self.jac(self.palm)
            Je, _ = self.jac(self.elbow)
            r = np.r_[
                goal - d.xpos[self.palm],
                0.12 * rot,
                0.11 * (egoal - d.xpos[self.elbow]),
                0.025 * (qr - q[self.addr]),
            ]
            J = np.vstack([Jp, 0.12 * Jr, 0.11 * Je, 0.025 * eye])
            delta = np.linalg.solve(J.T @ J + 0.00005 * eye, J.T @ r)
            new = np.clip(
                q[self.addr] + np.clip(delta, -0.14, 0.14), self.limits[:, 0], self.limits[:, 1]
            )
            if np.max(np.abs(new - q[self.addr])) < 3e-6:
                break
            q[self.addr] = new
            self.pose(q)
        return (
            q,
            float(
                np.linalg.norm(d.xpos[self.palm] - goal + d.xmat[self.torso].reshape(3, 3) @ local_delta)
                * 1000
            ),
            float(np.rad2deg(Rotation.from_matrix(ori @ d.xmat[self.palm].reshape(3, 3).T).magnitude())),
        )


def blend(depth):
    # Smooth 250-ms shoulders on both sides of every >1-mm overlap. Merge overlapping ramps.
    ramp = 50
    mask = np.zeros(len(depth))
    for i in np.flatnonzero(depth > 1):
        lo = max(i - ramp, 0)
        hi = min(i + ramp + 1, len(depth))
        x = np.arange(lo, hi)
        mask[lo:hi] = np.maximum(mask[lo:hi], 0.5 + 0.5 * np.cos(np.pi * (x - i) / ramp))
    return mask


def avoid_crossings(q, rig):
    pre = []
    for v in q:
        rig.pose(v)
        pre.append(rig.cross())
    pre = np.asarray(pre)
    weight = blend(pre)

    def score(ids, offset):
        depths = []
        move = []
        angles = []
        qx = []
        for ix in ids:
            if weight[ix] > 0:
                cand, mm, ang = rig.move(q[ix], offset * weight[ix])
                dep = rig.cross()
            else:
                cand, mm, ang, dep = q[ix], 0.0, 0.0, pre[ix]
            qx.append(cand[rig.addr])
            depths.append(dep)
            move.append(mm)
            angles.append(ang)
        return list(map(np.asarray, [depths, move, angles, qx]))

    if pre.max() <= 2:
        return q
    sample = np.union1d(np.where(weight > 0)[0][::6], np.argmax(pre))
    # Anatomical +Y is the robot's left. Also test slightly forward, down or up if the hands approach along another axis.
    choices = []
    best = None
    for mm in [12, 22, 35, 50, 65, 85]:
        for direction in [
            [0, 1, 0],
            [0.35, 1, 0],
            [0, 1, 0.65],
            [0, 1, -0.65],
            [-0.5, 1, 0],
            [0, 0, -1],
            [0, 0, 1],
            [-1, 0, 0],
            [1, 0, 0],
            [-1, 0.2, -1],
            [1, 0.2, 1],
            [1, 0.2, -1],
            [-1, 0.2, 1],
            [0, 0.25, -1],
            [0, 0.25, 1],
        ]:
            off = np.array(direction) * [1, rig.sign, 1] / np.linalg.norm(direction) * mm * 0.001
            dep, movement, tilt, _ = score(sample, off)
            objective = max(0, dep.max() - 1) ** 2 * 5 + np.mean(movement) / 22 + np.mean(tilt) / 5
            choices.append((objective, off, float(dep.max())))
        best = min(choices, key=lambda c: c[0])
        if best[2] <= 1.5:
            break
    dep, mm, tilt, pos = score(range(len(q)), best[1])
    step = np.max(np.abs(np.diff(pos, axis=0))) * 180 / np.pi
    oldstep = np.max(np.abs(np.diff(q[:, rig.addr], axis=0))) * 180 / np.pi
    take = dep.max() < pre.max() - 1 and step < max(10, oldstep + 3) and np.max(mm) < 100
    if take:
        q[:, rig.addr] = pos
    return q


def clear_shoulder(q, rig):
    m, d = rig.m, rig.d
    names = [m.body(int(x)).name for x in m.geom_bodyid]

    def depth():
        mujoco.mj_collision(m, d)
        return max(
            [0.0]
            + [
                float(-c.dist * 1000)
                for c in d.contact
                if {names[c.geom1], names[c.geom2]}
                == {"torso_link", rig.passive + "_shoulder_roll_link"}
            ]
        )

    pre = []
    for v in q:
        rig.pose(v)
        pre.append(depth())
    if max(pre) <= 2:
        return q
    weight = blend(np.array(pre))
    ids = rig.addr

    def sweep(offset):
        ds = []
        for v, w in zip(q, weight):
            p = v.copy()
            p[ids[1]] = np.clip(p[ids[1]] + w * offset, *rig.limits[1])
            rig.pose(p)
            ds.append(depth())
        return max(ds)

    offset = next(
        (o * rig.sign for o in [0.1, 0.15, 0.18, 0.20, 0.24, 0.28] if sweep(o * rig.sign) < 1), None
    )
    if offset is None:
        return q
    result = []
    rot = []
    dist = []
    depths = []
    eb = []
    for i, (v, w) in enumerate(zip(q, weight)):
        if w == 0:
            result.append(v[ids])
            rot.append(0)
            dist.append(0)
            eb.append(0)
            depths.append(pre[i])
            continue
        rig.pose(v)
        P0 = d.xpos[rig.palm].copy()
        R0 = d.xmat[rig.palm].reshape(3, 3).copy()
        E0 = d.xpos[rig.elbow].copy()
        cand = v.copy()
        cand[ids[1]] = np.clip(cand[ids[1]] + offset * w, *rig.limits[1])
        movable = ids[2:]
        lo, hi = rig.limits[2:].T

        def res(x):
            cand[movable] = x
            rig.pose(cand)
            rv = Rotation.from_matrix(R0 @ d.xmat[rig.palm].reshape(3, 3).T).as_rotvec()
            return np.r_[
                d.xpos[rig.palm] - P0, 0.1 * rv, 0.3 * (d.xpos[rig.elbow] - E0), 0.04 * (x - v[movable])
            ]

        start = v[movable] if not result else np.asarray(result[-1])[2:]
        opt = least_squares(
            res, np.clip(start, lo + 1e-10, hi - 1e-10), bounds=(lo, hi), max_nfev=25, gtol=1e-8
        )
        r = res(opt.x)
        result.append(cand[ids].copy())
        dist.append(float(np.linalg.norm(r[:3]) * 1000))
        rot.append(float(np.linalg.norm(r[3:6]) / 0.1 * 180 / np.pi))
        eb.append(float(np.linalg.norm(r[6:9]) / 0.3 * 1000))
        depths.append(depth())
    if max(depths) < 1 and np.max(np.abs(np.diff(result, axis=0))) < np.radians(15):
        q[:, ids] = result
    return q


class Rig:
    def __init__(self):
        self.m = m = mujoco.MjModel.from_xml_path(str(MODEL))
        self.d = mujoco.MjData(m)
        self.joints = [
            m.joint(f"{s}_{key}").id
            for s in SIDES
            for key in ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle")
        ]
        self.qa = m.jnt_qposadr[self.joints]
        self.va = m.jnt_dofadr[self.joints]
        self.lo, self.hi = m.jnt_range[self.joints].T
        self.hip_geom_ids = [
            int(
                np.flatnonzero((m.geom_bodyid == m.body(s + "_hip_yaw_link").id) & (m.geom_group == 3))[
                    0
                ]
            )
            for s in SIDES
        ]
        self.refs = np.array(
            [[m.body(f"{s}_{key}_link").id for key in ("hip_pitch", "knee", "ankle")] for s in SIDES]
        )
        self.jp = np.empty((3, m.nv))
        self.jr = self.jp.copy()
        self.feet = []
        for s in SIDES:
            fid = int(
                np.flatnonzero((m.geom_bodyid == m.body(s + "_ankle_link").id) & (m.geom_group == 3))[0]
            )
            mid = m.geom_dataid[fid]
            assert mid >= 0
            self.feet.append(
                (
                    fid,
                    m.mesh_vert[m.mesh_vertadr[mid] : m.mesh_vertadr[mid] + m.mesh_vertnum[mid]].astype(
                        float
                    ),
                )
            )

    def prepare_surface(self):
        angles = np.linspace(-0.43, 0.43, 129)
        distances = np.zeros((len(angles), len(angles)))
        for i, angle in enumerate(angles):
            for j, other in enumerate(angles):
                self.d.qpos[self.qa[[0, 5]]] = [angle, other]
                mujoco.mj_kinematics(self.m, self.d)
                mujoco.mj_comPos(self.m, self.d)
                distances[i, j] = self.dist()
        self.surf = RectBivariateSpline(angles, angles, distances)

    def set(self, q):
        self.d.qpos[:] = q
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_comPos(self.m, self.d)

    def dist(self):
        return float(mujoco.mj_geomDistance(self.m, self.d, *self.hip_geom_ids, 0.12, None))

    def foot(self):
        out = []
        for gid, verts in self.feet:
            mat = self.d.geom_xmat[gid].reshape(3, 3)
            local = verts[np.argmin(verts @ mat[2])]
            out.append(self.d.geom_xpos[gid] + mat @ local)
        return np.array(out)

    def jac(self, which, pt=None):
        if pt is None:
            mujoco.mj_jacBody(self.m, self.d, self.jp, self.jr, int(which))
        else:
            mujoco.mj_jac(self.m, self.d, self.jp, self.jr, pt, int(which))
        return self.jp[:, self.va], self.jr[:, self.va]

    def geom(self):
        F = self.d.xmat[self.refs[:, 2]].reshape(2, 3, 3)
        return self.d.xpos[self.refs].copy(), F, self.foot()

    def solve(self, q, prev):
        self.set(q)
        pose, axes, feet = self.geom()
        x0 = q[self.qa]
        orig = self.dist()
        if orig >= 0.0005:
            return np.zeros(10)

        def joint_penalty(gap):
            # Fully engage before the first hull intersection, with submillimetre margin.
            return 32 * min(0, gap - 0.0005)

        last_x = None
        last = None

        def evaluate(x):
            nonlocal last_x, last
            if last_x is not None and np.array_equal(x, last_x):
                return last
            w = q.copy()
            w[self.qa] = x
            self.set(w)
            P, F, S = self.geom()
            residual = []
            J = []
            for k in range(2):
                hip, knee, ankle = self.refs[k]
                # Measured kinematic baseline, weights favour ankle position and seated knee alignment.
                for j, body, weight in [(0, hip, 0.3), (1, knee, 1.5), (2, ankle, 2.8)]:
                    jac, rot = self.jac(body)
                    residual.append(weight * (P[k, j] - pose[k, j]))
                    J.append(weight * jac)
                for axis, weight in [(0, 0.1), (2, 0.11)]:  # preserve sole roll/pitch and foot heading
                    v = F[k, :, axis]
                    residual.append(weight * (v - axes[k, :, axis]))
                    J.append(weight * np.cross(rot.T, v).T)
                jac, _ = self.jac(ankle, S[k])
                floor = S[k, 2]
                residual.append(np.array([4 * (floor - feet[k, 2]), 12 * min(0, floor + 0.0001)]))
                J.append(np.stack([4 * jac[2], 12 * jac[2] if floor < -0.0001 else np.zeros(10)]))
            gap = self.surf.ev(x[0], x[5]).item()
            row = np.zeros(10)
            if gap < 0.0005:
                row[0] = 32 * self.surf.ev(x[0], x[5], dx=1).item()
                row[5] = 32 * self.surf.ev(x[0], x[5], dy=1).item()
            residual.append(np.array([joint_penalty(gap)]))
            J.append(row[None, :])
            residual.extend([0.018 * (x - x0), 0.035 * (x - x0 - prev)])
            J.extend([0.018 * np.eye(10), 0.035 * np.eye(10)])
            last = (np.concatenate(residual), np.concatenate(J))
            last_x = x.copy()
            return last

        seed = np.clip(x0 + prev, self.lo + 1e-10, self.hi - 1e-10)
        sol = least_squares(
            lambda x: evaluate(x)[0],
            seed,
            jac=lambda x: evaluate(x)[1],
            bounds=(self.lo, self.hi),
            ftol=1e-7,
            xtol=2e-7,
            gtol=5e-7,
            max_nfev=36,
        )
        return sol.x - x0


def hip_clearance(q):
    rig = Rig()
    depth = []
    for pose in q:
        rig.set(pose)
        depth.append(rig.dist())
    if min(depth) >= -0.002:
        return q
    rig.prepare_surface()
    candidate = q.copy()
    prev = np.zeros(10)
    knee_shift = 0
    ankle_shift = 0
    flexion_change = 0
    floor = np.inf
    remaining = 0

    def flex(points):
        a, b = np.diff(points, axis=1).transpose(1, 0, 2)
        return np.degrees(np.arctan2(np.linalg.norm(np.cross(a, b), axis=1), np.sum(a * b, axis=1)))

    for i, pose in enumerate(candidate):
        rig.set(pose)
        base = rig.geom()[0]
        prev = rig.solve(pose, prev)
        pose[rig.qa] += prev
        rig.set(pose)
        new, axes, soles = rig.geom()
        knee_shift = max(knee_shift, np.linalg.norm(new[:, 1] - base[:, 1], axis=1).max())
        ankle_shift = max(ankle_shift, np.linalg.norm(new[:, 2] - base[:, 2], axis=1).max())
        flexion_change = max(flexion_change, np.max(np.abs(flex(new) - flex(base))))
        floor = min(floor, soles[:, 2].min())
        remaining = max(remaining, -rig.dist())
    if (
        remaining <= 0.002
        and knee_shift < 0.075
        and ankle_shift < 0.040
        and flexion_change < 5
        and floor >= -0.002
    ):
        return candidate
    return q


def correct(q, active):
    rig = PassiveArm(active)
    q = clear_shoulder(avoid_crossings(q, rig), rig)
    return hip_clearance(q)
