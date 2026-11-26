"""Refit temporal branch changes relative to the motion of the measured finger."""

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.sparse import block_diag, diags
from scipy.sparse.linalg import spsolve
from . import hand as hand_ik

INDS = ["TH", "FF", "MF", "RF", "LF"]


def angles(a, b):
    a = hand_ik.normalized(a)
    b = hand_ik.normalized(b)
    return np.degrees(np.arctan2(np.linalg.norm(np.cross(a, b), axis=-1), np.sum(a * b, axis=-1)))


def check_motion(z):
    q = z["finger_qpos"]
    names = z["joint_names"]
    src = np.diff(z["source_chains_local_m"], axis=2)
    srcang = angles(src[:-1], src[1:]).max(axis=2)
    srcstep = np.diff(z["source_tips_local_m"], axis=0) * 1000
    botstep = np.diff(z["robot_tips_local_m"], axis=0) * 1000
    qstep = np.abs(np.rad2deg(np.diff(q, axis=0)))
    groups = ["TH", "FF", "MF", "RF", "LF"]
    qstepd = np.stack([qstep[:, [n[3:5] == g for n in names]].max(axis=1) for g in groups], axis=1)
    # Flag fast joint switches unsupported by source bone-direction changes, or TIP jumps in a quiet digit.
    mismatch = qstepd - 2.5 * srcang
    errstep = np.linalg.norm(botstep - srcstep, axis=-1)
    tipmis = errstep - 0.75 * np.linalg.norm(srcstep, axis=-1)
    mask = ((mismatch > 10) & (qstepd > 12)) | ((tipmis > 5) & (errstep > 6))
    return mask, dict(
        max_excess_joint_step_deg=round(float(mismatch.max()), 3),
        max_excess_tip_step_mm=round(float(tipmis.max()), 3),
        flagged_transitions=int(np.any(mask, axis=1).sum()),
        worst_frames=np.flatnonzero(np.any(mask, axis=1)).tolist(),
    )


def frame_system(hand, x, t):
    p, j, d, dc = hand.eval(x)
    sep = p[hand_ik.A, -1] - p[hand_ik.B, -1]
    s = t[hand_ik.A, -1] - t[hand_ik.B, -1]
    dr = j[hand_ik.A, -1] - j[hand_ik.B, -1]
    l = hand.length[:, :, None]
    srcbones = np.diff(t, axis=1)
    goal = hand_ik.normalized(srcbones) * l
    r = [
        np.linalg.norm(sep, axis=1) - np.linalg.norm(s, axis=1),
        0.30 * (sep - s),
        0.14 * (p[:, -1] - (hand.neutral[:, 0] + goal.sum(axis=1))),
        hand.bone_weight * (np.diff(p, axis=1) - goal),
        3.5 * d,
        0.3 * x,
    ]
    J = [
        np.einsum("ai,aik->ak", hand_ik.normalized(sep), dr),
        0.3 * dr,
        0.14 * j[:, -1],
        hand.bone_weight * np.diff(j, axis=1),
        3.5 * dc,
        0.3 * np.eye(len(x)),
    ]
    return (
        np.concatenate([v.ravel() for v in r]),
        np.concatenate([v.reshape(-1, len(x)) for v in J]),
        p[:, -1].ravel(),
        j[:, -1].reshape(15, -1),
    )


def candidate(hand, init, goal, alpha, its=45, velocity_weight=0.0):
    n, d = init.shape
    bones = np.diff(goal, axis=2)
    srcang = angles(bones[:-1], bones[1:]).max(axis=2)
    srcstep = np.linalg.norm(np.diff(goal[:, :, -1], axis=0), axis=-1)
    digit = [INDS.index(n[3:5]) for n in hand.group_names]
    w = alpha / (1 + srcang[:, digit] / 3.0 + srcstep[:, digit] / 6)
    w = np.maximum(w, 4.5)
    # Use the two unchanged end samples as boundary conditions.
    x = np.clip(gaussian_filter1d(init, 2.2, axis=0, mode="nearest"), hand.lo, hand.hi)
    x[[0, -1]] = init[[0, -1]]

    def evaluate(x):
        H = []
        g = []
        loss = 0.0
        tips = []
        jac = []
        for prev, target in zip(x, goal):
            r, j, p, jt = frame_system(hand, prev, target)
            tips.append(p)
            jac.append(jt)
            loss += r @ r
            H.append(j.T @ j)
            g.append(j.T @ r)
        H, g = np.asarray(H), np.asarray(g)
        cross = None
        if velocity_weight:
            # Signed TIP motions stay at their native timestamp; no velocity lag target.
            jac = np.asarray(jac)
            dr = np.diff(tips, axis=0) - np.diff(goal[:, :, -1].reshape(n, 15), axis=0)
            v2 = velocity_weight**2
            mass = jac.transpose(0, 2, 1) @ jac
            H[:-1] += v2 * mass[:-1]
            H[1:] += v2 * mass[1:]
            g[:-1] -= v2 * np.einsum("nij,ni->nj", jac[:-1], dr)
            g[1:] += v2 * np.einsum("nij,ni->nj", jac[1:], dr)
            cross = -v2 * (jac[:-1].transpose(0, 2, 1) @ jac[1:])
            loss += v2 * np.sum(dr**2)
        return H, g, loss + np.sum((w * np.diff(x, axis=0)) ** 2), cross

    H, g, loss, cross = evaluate(x)
    damping = 1.0
    reductions = 0
    for it in range(its):
        prior = np.diff(x, axis=0) * (w * w)
        grad = g.copy()
        grad[:-1] -= prior
        grad[1:] += prior
        blocked = ((x <= hand.lo + 1e-7) & (grad > 0)) | ((x >= hand.hi - 1e-7) & (grad < 0))
        blocked[[0, -1], :] = True
        free = np.flatnonzero(~blocked.ravel())
        pdiag = np.zeros_like(x)
        pdiag[:-1] += w * w
        pdiag[1:] += w * w
        sysmat = block_diag(list(H), format="csr") + diags(
            [(pdiag + damping).ravel(), -(w * w).ravel(), -(w * w).ravel()], [0, -d, d], format="csr"
        )
        if cross is not None:
            from scipy.sparse import coo_matrix

            bsize = x.shape[1]
            frame, xi, yi = np.indices(cross.shape)
            mat = coo_matrix(
                (cross.ravel(), ((frame * bsize + xi).ravel(), ((frame + 1) * bsize + yi).ravel())),
                shape=sysmat.shape,
                dtype=float,
            ).tocsr()
            sysmat = sysmat + mat + mat.T
        step = np.zeros(x.size)
        step[free] = spsolve(sysmat[free][:, free], -grad.ravel()[free])
        step = step.reshape(x.shape)
        step *= min(1.0, 0.45 / max(0.001, np.abs(step).max()))
        nx = np.clip(x + step, hand.lo, hand.hi)
        nH, ng, nloss, ncross = evaluate(nx)
        if nloss < loss:
            improvement = loss - nloss
            x, H, g, loss, cross = nx, nH, ng, nloss, ncross
            damping = max(0.04, damping * 0.55)
            reductions = reductions + 1 if improvement < max(0.02, 4e-6 * loss) else 0
            if reductions >= 3:
                break
        else:
            damping = max(1, damping * 4)
            if damping > 2e6:
                break
    return x


def runs(f, length, margin=20):
    out = []
    for idx in f:
        a, b = max(0, int(idx) - margin), min(length - 1, int(idx) + margin + 1)
        if not out or a > out[-1][1]:
            out.append([a, b])
        else:
            out[-1][1] = max(b, out[-1][1])
    return out


def refine(hand, xs, positions, target):
    def screen(x, p, t):
        return check_motion(
            dict(
                finger_qpos=x @ hand.g.T,
                joint_names=hand.names,
                source_chains_local_m=t / 1000,
                robot_chains_local_m=p / 1000,
                source_tips_local_m=t[:, :, -1] / 1000,
                robot_tips_local_m=p[:, :, -1] / 1000,
            )
        )

    flags, _ = screen(xs, positions, target)
    for lo, hi in runs(np.flatnonzero(flags.any(axis=1)), len(xs)):
        idx = slice(lo, hi + 1)
        source = target[idx]
        initial = xs[idx].copy()
        desired = np.linalg.norm(source[:, hand_ik.A, -1] - source[:, hand_ik.B, -1], axis=-1)
        best = None
        for velocity in [False, True]:
            for alpha, weight in zip([40.0, 75.0, 125.0], [2.0, 3.5, 6.0] if velocity else [0.0] * 3):
                x = candidate(hand, initial, source, alpha, velocity_weight=weight)
                measured = [hand.measures(row) for row in x]
                points = np.asarray([m[0] for m in measured])
                depth = max(-m[1] for m in measured)
                _, stats = screen(x, points, source)
                pairs = np.linalg.norm(points[:, hand_ik.A, -1] - points[:, hand_ik.B, -1], axis=-1)
                score = (
                    stats["flagged_transitions"],
                    np.percentile(np.abs(pairs - desired), 95) + 3 * max(0, depth - 2),
                )
                if best is None or score < best[0]:
                    best = score, x, points
                if best[0][0] == 0:
                    break
            if best[0][0] == 0:
                break
        _, xs[idx], positions[idx] = best
    return xs, positions
