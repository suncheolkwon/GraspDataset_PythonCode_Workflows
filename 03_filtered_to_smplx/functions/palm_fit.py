"""Windowed native palm and finger fit, including the robust root offset."""

import time
import numpy as np
import torch
import reset_evaluation as e
import torch_geometry as g
from metrics_v2 import components, frame_condition
from reset_rotation import aa2rot_torch
from window_connection import windows_for_segment
from runtime import forward, PARAMS


def fit(m, s, baseline, segments, cfg, side, device):
    N = len(s["time_s"])
    valid = s["valid"]
    wrist_index = 19 if side == "L" else 20
    hand = "lhand_pose" if side == "L" else "rhand_pose"
    sp = e.palm_frame(s["wrist_m"], s["chains_m"])
    sa, sq = components(s["chains_m"], sp, side)
    am = np.isfinite(sa)
    pm = np.isfinite(sp).all((1, 2)) & (
        frame_condition(s["wrist_m"], s["chains_m"]) >= 0.1
    )
    src_d = e.unit(np.diff(s["chains_m"], axis=2))
    sl = np.einsum("tji,tdkj->tdki", sp, src_d)
    lm = np.isfinite(sl).all(-1)
    tensor = lambda x: torch.as_tensor(x, device=device, dtype=torch.float32)
    output = {k: np.full_like(v, np.nan) for k, v in baseline.items()}
    owner = np.full(N, -1)
    done_mask = np.zeros(N, bool)
    windows = []
    for seg, (lo, hi) in enumerate(segments):
        windows.extend(
            (seg, a, b, c)
            for a, b, c in windows_for_segment(
                lo, hi, cfg["window_size"], cfg["overlap"]
            )
        )
    for wi, (seg, a, b, c) in enumerate(windows):
        print(f"P33 window {wi + 1}/{len(windows)}", flush=True)
        ref = {k: tensor(v[a:b]).clone() for k, v in baseline.items()}
        prefix = c - a
        if prefix:
            assert done_mask[a:c].all()
            fw = tensor(output["body_pose"][a:c, wrist_index])
            fh = tensor(output[hand][a:c])
            wr = tensor(
                np.repeat(output["body_pose"][c - 1 : c, wrist_index], b - c, axis=0)
            ).requires_grad_()
            hp = tensor(
                np.repeat(output[hand][c - 1 : c], b - c, axis=0)
            ).requires_grad_()
        else:
            fw = ref["body_pose"][:0, wrist_index]
            fh = ref[hand][:0]
            wr = ref["body_pose"][:, wrist_index].clone().requires_grad_()
            hp = ref[hand].clone().requires_grad_()
        params = [wr, hp]
        TP = tensor(np.nan_to_num(sp[a:b]))
        TA = tensor(np.nan_to_num(sa[a:b])) * np.pi / 180
        TL = tensor(np.nan_to_num(sl[a:b]))
        AM = torch.tensor(am[a:b], device=device)
        PM = torch.tensor(pm[a:b], device=device)
        LM = torch.tensor(lm[a:b], device=device)
        opt = torch.optim.LBFGS(
            params,
            lr=1.0,
            max_iter=cfg["max_iter"],
            max_eval=cfg["max_eval"],
            tolerance_grad=1e-5,
            tolerance_change=1e-9,
            history_size=50,
            line_search_fn="strong_wolfe",
        )
        calls = 0
        last = time.monotonic()

        def parameters():
            w = torch.cat([fw, wr], 0)
            h = torch.cat([fh, hp], 0)
            p = dict(ref)
            p["body_pose"] = torch.cat(
                [
                    ref["body_pose"][:, :wrist_index],
                    w[:, None],
                    ref["body_pose"][:, wrist_index + 1 :],
                ],
                1,
            )
            p[hand] = h
            return p, w, h

        def objective():
            p, w, h = parameters()
            z = forward(m, p)
            ch, P = g.chain(z, side)
            A, Q = g.angles(ch, P, side)
            d = g.unit(ch[:, :, 1:] - ch[:, :, :-1])
            local = torch.einsum("tji,tdkj->tdki", P, d)
            palm = (
                ((P - TP).square().sum((1, 2)) / 2)[PM].mean()
                if PM.any()
                else wr.sum() * 0
            )
            link = (
                (local - TL).square().sum(-1)[LM].mean() if LM.any() else hp.sum() * 0
            )
            angle = (2 - 2 * torch.cos(A - TA))[AM].mean() if AM.any() else hp.sum() * 0
            chart = (
                (torch.relu(0.1 - Q) / 0.1).square()[AM].mean()
                if AM.any()
                else hp.sum() * 0
            )
            rot = aa2rot_torch(torch.cat([w[:, None], h], 1))
            temporal = (
                (rot[1:] - rot[:-1]).square().sum((-1, -2)).mean() / (2 * 0.05**2)
                if len(w) > 1
                else wr.sum() * 0
            )
            anchor = (w - ref["body_pose"][:, wrist_index]).square().mean() + (
                h - ref[hand]
            ).square().mean()
            terms = {
                "palm": palm,
                "link_directions": link,
                "defined_angles": angle,
                "chart_condition": chart,
                "temporal_rotation": temporal,
                "baseline_anchor": anchor,
            }
            return sum(terms[k] * cfg["weights"][k] for k in terms), terms

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
        initial, terms = objective()
        iv = float(initial.detach())
        del initial
        opt.step(closure)
        with torch.no_grad():
            final, terms = objective()
            p, _, _ = parameters()
            done = {k: v.detach().cpu().numpy() for k, v in p.items()}
        assert all(np.isfinite(v).all() for v in done.values())
        dt = time.monotonic() - ts
        if prefix:
            for k in PARAMS:
                assert np.array_equal(done[k][:prefix], output[k][a:c]), k
        for k in PARAMS:
            output[k][c:b] = done[k][prefix:]
        done_mask[c:b] = True
        owner[c:b] = wi
    root_start = time.monotonic()
    model_ch = np.full_like(s["chains_m"], np.nan)
    model_w = np.full_like(s["wrist_m"], np.nan)
    ids = np.flatnonzero(valid)
    for off in range(0, len(ids), 32):
        ii = ids[off : off + 32]
        with torch.no_grad():
            z = forward(m, {k: tensor(v[ii]) for k, v in output.items()})
            ch, _ = g.chain(z, side)
        model_ch[ii] = ch.cpu().numpy()
        model_w[ii] = z.joints[:, e.WRIST[side]].cpu().numpy()
    shifts = []
    for seg, (a, b) in enumerate(segments):
        # Balanced blocks: wrist 1/4, MCP geometric palm centre 1/4, all chain landmarks 1/2.
        residual = np.concatenate(
            [
                (s["wrist_m"][a:b] - model_w[a:b])[:, None],
                (s["chains_m"][a:b, 1:4, 0].mean(1) - model_ch[a:b, 1:4, 0].mean(1))[
                    :, None
                ],
                (s["chains_m"][a:b] - model_ch[a:b]).reshape(b - a, 20, 3),
            ],
            axis=1,
        ).astype(float)
        weights = np.broadcast_to(
            np.r_[0.25, 0.25, np.full(20, 0.5 / 20)], residual.shape[:-1]
        ).reshape(-1)
        rr = residual.reshape(-1, 3)
        offset = np.median(rr, axis=0)
        for it in range(50):
            norm = np.linalg.norm(rr - offset, axis=1)
            w = weights * np.minimum(1, 0.020 / np.maximum(norm, 1e-12))
            new = np.sum(rr * w[:, None], axis=0) / w.sum()
            change = np.linalg.norm(new - offset)
            offset = new
            if change < 1e-9:
                break
        output["transl"][a:b] += offset.astype(np.float32)
        shifts.append(
            {
                "segment": seg,
                "rows_half_open": [a, b],
                "offset_world_m": offset.tolist(),
                "IRLS_iterations": it + 1,
                "converged": bool(change < 1e-9),
                "max_native_float32_offset_roundoff_m": float(
                    np.max(
                        abs(output["transl"][a:b] - baseline["transl"][a:b] - offset)
                    )
                ),
            }
        )
    return output
