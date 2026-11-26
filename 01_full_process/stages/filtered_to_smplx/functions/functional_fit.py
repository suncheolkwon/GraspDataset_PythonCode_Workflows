"""F02A and F03A measured-wrist/finger refinement, with frozen body and root."""

import time
import numpy as np
import torch
import reset_evaluation as ev
import torch_geometry as g
import posture_objective as po
import proxy_torch
from runtime import forward, PARAMS
from reset_rotation import aa2rot_torch
from window_connection import windows_for_segment
from source_aware_continuity import continuation_initial_guess, source_aware_edges
from functional_terms import functional_terms


def fit(stage, m, source, a, seed, segments, contract, side, device, proxycfg, fkwargs):
    hand = "lhand_pose" if side == "L" else "rhand_pose"
    bi = [po.BODY_POSE_INDICES[side][-1]]
    valid = source["valid"]
    targets = po.source_targets(
        source["wrist_m"], source["chains_m"], source["body_positions_m"], side
    )
    functional_valid = (
        valid & targets["palm_mask"] & np.isfinite(source["chains_m"]).all((1, 2, 3))
    )
    assert np.array_equal(functional_valid, valid), (
        "Required functional chart missing: retain mask; do not invent target"
    )

    times = source["source_time_s"]
    N = len(times)
    windows = []
    for si, (lo, hi) in enumerate(segments):
        windows.extend(
            (si, l, h, c)
            for l, h, c in windows_for_segment(
                lo, hi, contract["window_size"], contract["overlap"]
            )
        )
    output = {k: np.full_like(v, np.nan) for k, v in a.items()}
    T = lambda x: torch.as_tensor(x, device=device, dtype=torch.float32)
    for wi, (si, l, h, c) in enumerate(windows):
        print(f"{stage} window {wi + 1}/{len(windows)}", flush=True)
        ref = {k: T(v[l:h]) for k, v in a.items()}
        prefix = c - l
        fixedarm = (
            T(output["body_pose"][l:c][:, bi]) if prefix else ref["body_pose"][:0, bi]
        )
        fixedhand = T(output[hand][l:c]) if prefix else ref[hand][:0]
        if stage == "F03A":
            initial_arm = seed["body_pose"][c:h][:, bi].copy()
            initial_hand = seed[hand][c:h].copy()
            if prefix:
                elapsed = times[c:h] - times[c - 1]
                horizon = contract["continuity"]["initial_continuation_horizon_s"]
                initial_arm = continuation_initial_guess(
                    initial_arm,
                    output["body_pose"][c - 1, bi],
                    seed["body_pose"][c - 1, bi],
                    elapsed,
                    horizon,
                )
                initial_hand = continuation_initial_guess(
                    initial_hand,
                    output[hand][c - 1],
                    seed[hand][c - 1],
                    elapsed,
                    horizon,
                )
        else:
            initial_arm = a["body_pose"][c:h][:, bi].copy()
            initial_hand = a[hand][c:h].copy()
        arm = T(initial_arm).clone().requires_grad_()
        hp = T(initial_hand).clone().requires_grad_()
        params = [arm, hp]
        target = po.target_tensors(targets, l, h, device=device, dtype=torch.float32)
        sc = T(source["chains_m"][l:h])
        sw = T(source["wrist_m"][l:h])
        sp = T(targets["palm"][l:h])
        fv = torch.as_tensor(functional_valid[l:h], device=device)
        calls = 0
        last = time.monotonic()
        dt = T(np.diff(times[l:h]))
        edge_valid = torch.arange(h - l - 1, device=device) >= max(prefix - 1, 0)

        def parameters():
            ar = torch.cat([fixedarm, arm])
            hh = torch.cat([fixedhand, hp])
            p = dict(ref)
            p["body_pose"] = torch.stack(
                [
                    ar[:, bi.index(j)] if j in bi else ref["body_pose"][:, j]
                    for j in range(21)
                ],
                1,
            )
            p[hand] = hh
            return p, ar, hh

        def objective():
            p, ar, hh = parameters()
            z = forward(m, p)
            ch, pm = g.chain(z, side)
            terms = po.losses(ch, pm, po.arm_axes(z.joints, side), side, target)
            ff = functional_terms(
                ch, z.joints[:, ev.WRIST[side]], pm, sc, sw, sp, fv, **fkwargs
            )
            for name in ["distance", "vector", "palm", "distal"]:
                terms["functional_" + name] = ff[name + "_loss"][
                    ff["valid_mask"]
                ].mean()
            terms["collision_proxy"] = proxy_torch.evaluate(ch, proxycfg)["loss"].mean()
            if stage == "F03A":
                terms["source_aware_edges"] = source_aware_edges(
                    ch,
                    z.joints[:, ev.WRIST[side]],
                    pm,
                    sc,
                    sw,
                    sp,
                    dt,
                    edge_valid,
                    contract["continuity"],
                )
            rot = aa2rot_torch(torch.cat([ar, hh], 1))
            terms["temporal_rotation"] = (
                ((rot[1:] - rot[:-1]).square().sum((-1, -2)).sum(-1) / 16).mean()
                / (2 * 0.05**2)
                if len(ar) > 1
                else arm.sum() * 0
            )
            terms["baseline_anchor"] = (
                ar - ref["body_pose"][:, bi]
            ).square().mean() + (hh - ref[hand]).square().mean()
            return sum(terms[k] * contract["weights"][k] for k in terms), terms

        opt = torch.optim.LBFGS(
            params,
            lr=1.0,
            max_iter=contract["max_iter"],
            max_eval=contract["max_eval"],
            tolerance_grad=1e-5,
            tolerance_change=1e-9,
            history_size=50,
            line_search_fn="strong_wolfe",
        )

        def closure():
            nonlocal calls, last
            opt.zero_grad()
            loss, terms = objective()
            assert torch.isfinite(loss)
            loss.backward()
            assert all(torch.isfinite(p.grad).all() for p in params)
            calls += 1
            return loss

        ts = time.monotonic()
        initial, _ = objective()
        iv = float(initial.detach())
        del initial
        opt.step(closure)
        with torch.no_grad():
            final, terms = objective()
            p, _, _ = parameters()
            out = {k: v.detach().cpu().numpy() for k, v in p.items()}
        assert all(np.isfinite(v).all() for v in out.values())
        if prefix:
            assert all(np.array_equal(out[k][:prefix], output[k][l:c]) for k in PARAMS)
        for k in PARAMS:
            output[k][c:h] = out[k][prefix:]

    return output
