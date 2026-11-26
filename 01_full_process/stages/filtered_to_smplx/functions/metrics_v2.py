import numpy as np
import reset_evaluation as old

PROJECTION_MIN = 0.1  # unit-vector projection >= 0.1 => local angle amplification <=10
LONG_AXES = np.array([6, 7, 10, 11, 13, 14, 17, 18], dtype=int)
CONNECTION_AXES = np.array([i for i in range(20) if i not in LONG_AXES], dtype=int)


def conditioning(c, palm, side):
    d = old.unit(np.diff(c, axis=-2))
    normal = palm[..., 2] * (1 if side == "L" else -1)
    local = np.einsum("tji,tdj->tdi", palm, d[:, :, 0])
    transverse = np.cross(d[:, :, 0], normal[:, None])
    tn = np.linalg.norm(transverse, axis=-1)
    axis = old.unit(transverse)
    projected = d - axis[:, :, None, :] * np.sum(
        d * axis[:, :, None, :], axis=-1, keepdims=True
    )
    pn = np.linalg.norm(projected, axis=-1)
    quality = np.stack(
        [
            np.linalg.norm(local[:, :, [1, 2]], axis=-1),
            np.linalg.norm(local[:, :, [0, 1]], axis=-1),
            np.minimum(np.minimum(pn[:, :, 0], pn[:, :, 1]), tn),
            np.minimum(np.minimum(pn[:, :, 1], pn[:, :, 2]), tn),
        ],
        axis=-1,
    )
    quality[:, 0] = np.nan
    return quality


def frame_condition(w, c):
    reach = c[:, 1:4, 0].mean(1) - w
    span = c[:, 4, 0] - c[:, 1, 0]
    u = old.unit(reach)
    v = old.unit(span)
    return np.linalg.norm(np.cross(u, v), axis=-1)


def components(c, palm, side):
    a = old.signed_components(c, palm, side)
    q = conditioning(c, palm, side)
    a[~np.isfinite(q) | (q < PROJECTION_MIN)] = np.nan
    return a, q
