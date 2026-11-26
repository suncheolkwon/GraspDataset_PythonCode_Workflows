"""Temporal and torso refinements of bounded arm inverse kinematics."""

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares
from scipy.sparse import block_diag, diags, eye, kron, vstack
from . import body as body_ik


def torso_fit(kin, arm, q, z):
    arm.qa = np.r_[kin.m.jnt_qposadr[kin.hinge("torso")], arm.qa]
    arm.va = np.r_[kin.m.jnt_dofadr[kin.hinge("torso")], arm.va]
    arm.js = np.r_[kin.hinge("torso"), arm.js]
    lower, upper = kin.m.jnt_range[arm.js].T
    src = z[kin.active + "_arm_source_dirs"]
    basis = z[kin.active + "_arm_source_palm_basis"]
    trajectory = q.copy()
    previous = None
    for i, single in enumerate(q):
        yaw = single[arm.qa[0]]
        lo = lower.copy()
        hi = upper.copy()
        lo[0] = max(lo[0], yaw - np.deg2rad(25))
        hi[0] = min(hi[0], yaw + np.deg2rad(25))

        def build(initial):
            obj = arm.objective(single, src[i], basis[i], previous)

            def cost(x):
                val, jac = obj(x)
                return np.r_[val, 2.5 * (x[0] - yaw)], np.vstack([jac, [2.5] + [0] * 7])

            sol = least_squares(
                lambda x: cost(x)[0],
                np.clip(initial, lo + 1e-10, hi - 1e-10),
                jac=lambda x: cost(x)[1],
                bounds=(lo, hi),
                max_nfev=40,
                ftol=2e-7,
                xtol=2e-7,
                gtol=2e-7,
            )
            return sol

        best = build(single[arm.qa] if previous is None else previous)
        if (
            previous is not None
            and np.linalg.norm(best.fun) > 1.1
            and np.max(np.abs(single[arm.qa] - best.x)) > 0.14
        ):
            alt = build(single[arm.qa])
            if alt.cost < best.cost:
                best = alt
        trajectory[i, arm.qa] = best.x
        previous = best.x
    # Left arm is refitted to its own measured global targets after the torso moves.
    passive = "left" if kin.active == "right" else "right"
    left = body_ik.Limb(kin, passive, True)
    prev = None
    for i, single in enumerate(trajectory):
        prev, _ = left.solve(
            single,
            z[passive + "_arm_source_dirs"][i],
            z[passive + "_arm_source_palm_basis"][i],
            q[i, left.qa] if i == 0 else prev,
        )
    return trajectory


def continuity(kin, arm, q, z):
    qa = arm.qa
    step = np.max(abs(np.diff(np.degrees(q[:, qa]), axis=0)), axis=1)
    jumps = np.flatnonzero(step > 11) + 1
    # Both endpoints remain fixed. World orientation/limb targets are evaluated
    # on every original row and a symmetric path derivative discourages spin.
    windows = []
    for i in jumps:
        width = max(22, min(85, int(step[i - 1] * 0.75)))
        lo = max(0, i - width)
        hi = min(len(q) - 1, i + width)
        if windows and lo <= windows[-1][1] + 3:
            windows[-1][1] = max(windows[-1][1], hi)
        else:
            windows.append([lo, hi])
    report = []
    for lo, hi in windows:
        if hi - lo < 2:
            continue
        base = q[lo : hi + 1].copy()
        old = base[:, qa].copy()
        n = len(old)
        m = len(qa)
        funcs = [
            arm.objective(
                q[i],
                z[kin.active + "_arm_source_dirs"][i],
                z[kin.active + "_arm_source_palm_basis"][i],
                None,
            )
            for i in range(lo, hi + 1)
        ]
        D1 = diags([-np.ones(n - 1), np.ones(n - 1)], [0, 1], shape=(n - 1, n), format="csr")
        D2 = diags(
            [np.ones(n - 2), -2 * np.ones(n - 2), np.ones(n - 2)],
            [0, 1, 2],
            shape=(n - 2, n),
            format="csr",
        )
        # The derivative terms regularize redundant robot joint motion. Absolute
        # world palm and segment directions still track all native observations.
        regularize = kron(vstack([4.5 * D1, 10.0 * D2], format="csr"), eye(m), format="csr")
        free = np.arange(m, (n - 1) * m)
        RJ = regularize[:, free]
        initial = gaussian_filter1d(old, 4.0, axis=0, mode="nearest")

        def whole(x):
            return np.vstack([old[0], x.reshape(-1, m), old[-1]])

        def fun(x):
            path = whole(x)
            return np.concatenate(
                [*(f(a)[0] for f, a in zip(funcs[1:-1], path[1:-1])), regularize @ path.ravel()]
            )

        def jac(x):
            mats = [f(a)[1] for f, a in zip(funcs[1:-1], x.reshape(-1, m))]
            return vstack([block_diag(mats, format="csr"), RJ], format="csr")

        result = least_squares(
            fun,
            np.clip(initial[1:-1], arm.lo + 1e-9, arm.hi - 1e-9).ravel(),
            jac=jac,
            bounds=(np.tile(arm.lo, n - 2), np.tile(arm.hi, n - 2)),
            max_nfev=34,
            ftol=2e-5,
            xtol=2e-5,
            gtol=2e-5,
            tr_options={"maxiter": 160, "atol": 1e-6, "btol": 1e-6},
        )
        q[lo : hi + 1, qa] = whole(result.x)
        report.append(
            dict(
                rows=[int(lo), int(hi)],
                original_s=[float(z["source_time_s"][lo]), float(z["source_time_s"][hi])],
                old_max_hinge_step_deg=float(np.max(abs(np.diff(np.degrees(old), axis=0)))),
                new_max_hinge_step_deg=float(np.max(abs(np.diff(np.degrees(whole(result.x)), axis=0)))),
                termination=int(result.status),
                evaluations=int(result.nfev),
            )
        )
    return report


def arm_eval(kin, q, z, arm):
    prefix = arm.side + "_arm"
    src = z[prefix + "_source_dirs"]
    ori = z[prefix + "_source_palm_basis"]
    fields = {
        k: []
        for k in (
            "robot_dirs",
            "robot_end_m",
            "adapted_target_m",
            "robot_mid_m",
            "robot_start_m",
            "robot_palm_basis",
        )
    }
    for i, pose in enumerate(q):
        kin.setq(pose)
        a, b, c, u, v, R = arm.geometry()
        r = (np.stack([u, v]), c, a + arm.l1 * src[i, 0] + arm.l2 * src[i, 1], b, a, R)
        for key, val in zip(fields, r):
            fields[key].append(val.copy())
    fields = {k: np.asarray(v) for k, v in fields.items()}
    dirs = fields["robot_dirs"]
    R = fields["robot_palm_basis"]
    values = {
        "palm_normal_deg": body_ik.angle(ori[:, :, 2], R[:, :, 2]),
        "orientation_deg": body_ik.rotation_error(ori, R),
        "proximal_direction_deg": body_ik.angle(src[:, 0], dirs[:, 0]),
        "distal_direction_deg": body_ik.angle(src[:, 1], dirs[:, 1]),
        "adapted_endpoint_mm": np.linalg.norm(fields["robot_end_m"] - fields["adapted_target_m"], axis=1)
        * 1000,
        "adapted_intermediate_mm": np.linalg.norm(
            fields["robot_mid_m"] - fields["robot_start_m"] - arm.l1 * src[:, 0], axis=1
        )
        * 1000,
    }
    return (
        {prefix + "_" + k: v for k, v in fields.items()},
        {k: body_ik.summarize(v) for k, v in values.items()},
        values,
    )


def centered(kin, baseline, z, yaw):
    q = baseline.copy()
    arm = body_ik.Limb(kin, kin.active, True)
    torso_id = kin.hinge("torso")
    arm.qa = np.r_[kin.m.jnt_qposadr[torso_id], arm.qa]
    arm.va = np.r_[kin.m.jnt_dofadr[torso_id], arm.va]
    arm.js = np.r_[torso_id, arm.js]
    low, high = kin.m.jnt_range[arm.js].T
    q[:, arm.qa[0]] = yaw
    mid = len(q) // 2

    def cost_for(i, prev):
        lo = low.copy()
        hi = high.copy()
        lo[0] = max(lo[0], yaw[i] - np.deg2rad(25))
        hi[0] = min(hi[0], yaw[i] + np.deg2rad(25))
        native = arm.objective(
            q[i],
            z[kin.active + "_arm_source_dirs"][i],
            z[kin.active + "_arm_source_palm_basis"][i],
            prev,
        )

        def fun(x):
            return np.r_[native(x)[0], 2.5 * (x[0] - yaw[i])]

        def jac(x):
            return np.vstack([native(x)[1], [2.5] + [0] * 7])

        return fun, jac, lo, hi

    fun, jac, lo, hi = cost_for(mid, None)
    seeds = np.random.default_rng(301).uniform(lo, hi, (55, 8))
    seeds[0] = np.r_[yaw[mid], [-1.1, -0.3, 0.5, 1.1, 0.5, -0.3, 0.0]]
    solutions = [
        least_squares(fun, x, jac=jac, bounds=(lo, hi), max_nfev=90, ftol=3e-7, xtol=3e-7, gtol=3e-7)
        for x in seeds
    ]
    q[mid, arm.qa] = min(solutions, key=lambda s: s.cost).x
    for sequence in (range(mid + 1, len(q)), range(mid - 1, -1, -1)):
        prev = q[mid, arm.qa]
        for i in sequence:
            fun, jac, lo, hi = cost_for(i, prev)
            solution = least_squares(
                fun,
                np.clip(prev, lo + 1e-9, hi - 1e-9),
                jac=jac,
                bounds=(lo, hi),
                max_nfev=80,
                ftol=2e-7,
                xtol=2e-7,
                gtol=2e-7,
            )
            q[i, arm.qa] = solution.x
            prev = solution.x
    continuity(kin, body_ik.Limb(kin, kin.active, True), q, z)
    passive = "left" if kin.active == "right" else "right"
    arm = body_ik.Limb(kin, passive, True)
    prev = baseline[0, arm.qa]
    for i, pose in enumerate(q):
        prev, _ = arm.solve(
            pose, z[passive + "_arm_source_dirs"][i], z[passive + "_arm_source_palm_basis"][i], prev
        )
    return q


def refine(kin, q, z, number):
    active = body_ik.Limb(kin, kin.active, True)
    torso = kin.m.jnt_qposadr[kin.hinge("torso")]
    yaw = q[:, torso].copy()

    def metrics(q):
        return arm_eval(kin, q, z, active)[1]

    def score(m):
        return (
            m["palm_normal_deg"]["mean"]
            + 0.3 * m["palm_normal_deg"]["p95"]
            + 0.02 * m["adapted_endpoint_mm"]["mean"]
            + 0.01 * m["adapted_intermediate_mm"]["mean"]
        )

    if number in (18, 19, 30, 32):
        trial = torso_fit(kin, body_ik.Limb(kin, kin.active, True), q, z)
        yaw_change = np.mean(np.abs(np.degrees(trial[:, torso] - yaw)))
        if score(metrics(trial)) + 0.04 * yaw_change < score(metrics(q)) - 0.1:
            q = trial
    continuity(kin, active, q, z)
    if number == 30:
        candidate = centered(kin, q, z, yaw)
        old, new = metrics(q), metrics(candidate)
        if all(new[k]["mean"] < old[k]["mean"] for k in ("palm_normal_deg", "adapted_endpoint_mm")):
            q = candidate
    return q
