"""Native Shadow finger IK with unscaled ten-pair distance targets."""

from itertools import combinations
import mujoco
import numpy as np
from scipy.optimize import least_squares
from .model import MODEL, DIGITS

PAIRS = np.array(list(combinations(range(5), 2)))
A, B = PAIRS.T


def normalized(a):
    return a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-10)


def basis(wrist, bases):
    x = normalized(bases[..., 0, :] - bases[..., -1, :])
    y = bases.mean(axis=-2) - wrist
    y = normalized(y - np.sum(x * y, axis=-1, keepdims=True) * x)
    return np.stack([x, y, np.cross(x, y)], axis=-1)


class Hand:
    def __init__(self, side, bone_weight=0.27):
        prefix = "lh_" if side == "left" else "rh_"
        self.bone_weight = bone_weight
        self.m = m = mujoco.MjModel.from_xml_path(str(MODEL))
        self.d = d = mujoco.MjData(m)
        d.qpos[7:] = 0
        d.qpos[:7] = [0, 0, 0, 1, 0, 0, 0]
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        self.jids = np.array(
            [
                i
                for i in range(m.njnt)
                if m.joint(i).name.startswith(prefix) and "WRJ" not in m.joint(i).name
            ]
        )
        self.qa = m.jnt_qposadr[self.jids].copy()
        self.va = m.jnt_dofadr[self.jids].copy()
        self.names = [m.joint(i).name for i in self.jids]
        independent = [
            n for n in self.names if not (n[3:5] in ["FF", "MF", "RF", "LF"] and n.endswith("J1"))
        ]
        self.g = np.zeros((len(self.jids), len(independent)))
        self.lo = np.full(len(independent), -np.inf)
        self.hi = -self.lo
        for i, (n, j) in enumerate(zip(self.names, self.jids)):
            command = n if n in independent else n[:-1] + "2"
            k = independent.index(command)
            self.g[i, k] = 1
            lo, hi = m.jnt_range[j]
            if n != command:
                act = m.actuator(prefix + "A_" + n[3:-1] + "0")
                lo = max(lo, act.ctrlrange[0] / 2)
                hi = min(hi, act.ctrlrange[1] / 2)
            self.lo[k] = max(self.lo[k], lo)
            self.hi[k] = min(self.hi[k], hi)
        self.group_names = independent
        bodies = []
        offsets = []
        for digit, code in DIGITS.items():
            chain = (
                ["thbase", "thmiddle", "thdistal", "thdistal"]
                if digit == "Thumb"
                else [code + "proximal", code + "middle", code + "distal", code + "distal"]
            )
            for k, name in enumerate(chain):
                bodies.append(m.body(prefix + name).id)
                length = 0.034996109 if digit == "Thumb" else 0.032016930
                offsets.append([0.0, 0.0, length if k == 3 else 0.0])
        self.bid = np.asarray(bodies)
        self.off = np.asarray(offsets)
        self.jp = np.empty((3, m.nv))
        self.jr = self.jp.copy()
        self.wrist = d.xpos[m.body(prefix + "palm").id].copy()
        neutral = self.fk_positions()
        self.axes = basis(self.wrist, neutral.reshape(5, 4, 3)[1:, 0])
        self.neutral = ((neutral - self.wrist) @ self.axes * 1000).reshape(5, 4, 3)
        self.length = np.linalg.norm(np.diff(self.neutral, axis=1), axis=-1)
        self.cats = []
        self.links = []
        for gi in range(m.ngeom):
            b = m.body(int(m.geom_bodyid[gi])).name
            self.links.append(b)
            cat = next(
                (i for i, ab in enumerate(DIGITS.values()) if b.startswith(prefix + ab)),
                5 if b == prefix + "palm" else -1,
            )
            self.cats.append(cat)
        self.collpairs = list(combinations(range(6), 2))
        self.ref = np.zeros(self.g.shape[1])
        assert m.neq == 0 and self.g.shape == (22, 18)

    def fk_positions(self):
        return self.d.xpos[self.bid] + np.einsum(
            "nij,nj->ni", self.d.xmat[self.bid].reshape(-1, 3, 3), self.off
        )

    def jac(self, pos, bid):
        mujoco.mj_jac(self.m, self.d, self.jp, self.jr, pos, bid)
        return self.jp[:, self.va] @ self.g

    def contact_rows(self):
        best = {}
        for c in self.d.contact[: self.d.ncon]:
            a, b = self.cats[c.geom1], self.cats[c.geom2]
            if a < 0 or b < 0 or a == b or c.dist >= 0:
                continue
            pair = tuple(sorted([a, b]))
            if pair not in best or c.dist < best[pair].dist:
                best[pair] = c
        return best

    def eval(self, q):
        d, m = self.d, self.m
        d.qpos[self.qa] = self.g @ q
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        p = self.fk_positions()
        j = np.array([self.jac(pp, int(b)) for pp, b in zip(p, self.bid)])
        j = (np.einsum("ij,njk->nik", self.axes.T, j) * 1000).reshape(5, 4, 3, -1)
        p = ((p - self.wrist) @ self.axes * 1000).reshape(5, 4, 3)
        mujoco.mj_collision(m, d)
        contacts = self.contact_rows()
        # One row per inter-digit or finger/palm category, with fixed row count.
        # Allow 0.2 mm contact slop; analytic Jacobian is the normal relative velocity.
        gaps = np.zeros(len(self.collpairs))
        grad = np.zeros((len(gaps), len(q)))
        for i, pair in enumerate(self.collpairs):
            c = contacts.get(pair)
            if c is None:
                continue
            gaps[i] = min(0, c.dist * 1000 + 0.2)
            if gaps[i] >= 0:
                continue
            ja = self.jac(c.pos, int(m.geom_bodyid[c.geom1]))
            jb = self.jac(c.pos, int(m.geom_bodyid[c.geom2]))
            grad[i] = 1000 * c.frame[:3] @ (jb - ja)
        return p, j, gaps, grad

    def solve(self, target, start, prior, first=False):
        pair_goal = target[A, -1] - target[B, -1]
        gap_goal = np.linalg.norm(pair_goal, axis=1)
        direction_goal = normalized(np.diff(target, axis=1)) * self.length[:, :, None]
        local_goal = self.neutral[:, 0] + np.sum(direction_goal, axis=1)
        last = {}

        def evaluation(q):
            if "q" in last and np.array_equal(q, last["q"]):
                return last["v"]
            p, j, depth, contact_jac = self.eval(q)
            tip = p[:, -1]
            jt = j[:, -1]
            sep = tip[A] - tip[B]
            dj = jt[A] - jt[B]
            size = np.linalg.norm(sep, axis=1)
            dist_j = np.einsum("ai,aik->ak", normalized(sep), dj)
            # Absolute in millimetres for pair distances; bone-direction target
            # is rescaled only on each individual robot bone, for posture.
            vals = [
                size - gap_goal,
                0.30 * (sep - pair_goal),
                0.14 * (tip - local_goal),
                self.bone_weight * (np.diff(p, axis=1) - direction_goal),
                3.5 * depth,
                0.30 * (q - self.ref),
                (0.15 if first else 1.0) * (q - prior),
            ]
            jac = [
                dist_j,
                0.30 * dj,
                0.14 * jt,
                self.bone_weight * np.diff(j, axis=1),
                3.5 * contact_jac,
                0.30 * np.eye(len(q)),
                (0.15 if first else 1.0) * np.eye(len(q)),
            ]
            last["v"] = (
                np.concatenate([v.ravel() for v in vals]),
                np.concatenate([a.reshape(-1, len(q)) for a in jac]),
            )
            last["q"] = q.copy()
            return last["v"]

        opt = least_squares(
            lambda q: evaluation(q)[0],
            start,
            jac=lambda q: evaluation(q)[1],
            bounds=(self.lo, self.hi),
            ftol=3e-6,
            xtol=1e-6,
            gtol=1e-5,
            max_nfev=120 if first else 30,
        )
        return opt.x, opt.cost, opt.nfev

    def measures(self, q):
        p, _, depth, _ = self.eval(q)
        c = self.contact_rows()
        deepest = min([0.0] + [f.dist * 1000 for f in c.values()])
        if c:
            worst = min(c.values(), key=lambda q: q.dist)
            pair = (self.links[worst.geom1], self.links[worst.geom2])
        else:
            pair = ("", "")
        return p, deepest, pair


def fit_hand(wrist, chains, active, trial):
    axes = basis(wrist, chains[:, 1:, 0])
    target = np.einsum("nfpi,nij->nfpj", chains - wrist[:, None, None, :], axes) * 1000
    model = Hand(active)
    xs = []
    positions = []
    prev = np.zeros(len(model.lo))
    for i, goal in enumerate(target):
        if i == 0:
            choices = [
                model.solve(goal, np.clip(s * (model.hi - model.lo), model.lo, model.hi), prev, True)
                for s in [0.0, 0.3, 0.65]
            ]
            prev = min(choices, key=lambda c: c[1])[0]
        else:
            prev = model.solve(goal, prev, prev)[0]
        xs.append(prev)
        positions.append(model.measures(prev)[0])
    xs = np.array(xs)
    positions = np.array(positions)
    if trial != "Tripod_R_1":
        from .refine_hand import refine

        xs, positions = refine(model, xs, positions, target)
    errors = np.abs(
        np.linalg.norm(positions[:, A, -1] - positions[:, B, -1], axis=-1)
        - np.linalg.norm(target[:, A, -1] - target[:, B, -1], axis=-1)
    )
    return model.qa, xs @ model.g.T, errors
