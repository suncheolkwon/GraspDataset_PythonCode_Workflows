import numpy as np

DIGITS = ("Thumb", "Index", "Middle", "Ring", "Pinky")
BASE = {"L": (37, 25, 28, 34, 31), "R": (52, 40, 43, 49, 46)}
TIPS = {"L": (5361, 4933, 5058, 5169, 5286), "R": (8079, 7669, 7794, 7905, 8022)}
WRIST = {"L": 20, "R": 21}
BODY_IDS = np.array(
    [0, 3, 9, 12, 15, 13, 16, 18, 20, 14, 17, 19, 21, 1, 4, 7, 10, 2, 5, 8, 11]
)
BODY_NAMES = (
    "Motive_Skeleton001",
    "Motive_Ab",
    "Motive_Chest",
    "Motive_Neck",
    "Motive_Head",
    "Motive_LShoulder",
    "Motive_LUArm",
    "Motive_LFArm",
    "Motive_LHand",
    "Motive_RShoulder",
    "Motive_RUArm",
    "Motive_RFArm",
    "Motive_RHand",
    "Motive_LThigh",
    "Motive_LShin",
    "Motive_LFoot",
    "Motive_LToe",
    "Motive_RThigh",
    "Motive_RShin",
    "Motive_RFoot",
    "Motive_RToe",
)


def unit(a, eps=1e-10):
    a = np.asarray(a, np.float64)
    n = np.linalg.norm(a, axis=-1, keepdims=True)
    return np.divide(
        a, n, out=np.full_like(a, np.nan), where=(n > eps) & np.isfinite(n)
    )


def palm_frame(wrist, chains):
    """Frame-to-world columns x=index→pinky, y=wrist→MCP-centroid, z=x×y.

    The identical construction is fixed once for both source and neutral model.
    It is a geometric chart, not optical quaternion/anatomical calibration.
    """
    wrist = np.asarray(wrist, np.float64)
    c = np.asarray(chains, np.float64)
    y = unit(c[..., 1:4, 0, :].mean(axis=-2) - wrist)
    span = c[..., 4, 0, :] - c[..., 1, 0, :]
    x = unit(span - y * np.sum(span * y, axis=-1, keepdims=True))
    z = unit(np.cross(x, y))
    return np.stack((x, y, z), axis=-1)


def signed_components(chains, palm, side):
    """Named common geometric components; no rotvec-axis subtraction.

    Thumb components remain UNKNOWN: a clinical CMC correspondence is absent.
    Proximal direction is resolved in its own same-rule palm chart. PIP/DIP
    projected turns use the proximal-direction × palm-normal transverse axis.
    Projection degeneracies remain NaN, never filled as angle zero.
    """
    c = np.asarray(chains, np.float64)
    d = unit(np.diff(c, axis=-2))
    R = np.asarray(palm, np.float64)
    normal = R[..., 2] * (1 if side == "L" else -1)
    local = np.einsum("tji,tdj->tdi", R, d[:, :, 0])
    flex = np.degrees(
        np.arctan2(local[:, :, 2] * (1 if side == "L" else -1), local[:, :, 1])
    )
    abd = np.degrees(np.arctan2(local[:, :, 0], local[:, :, 1]))
    flex[np.linalg.norm(local[:, :, [1, 2]], axis=-1) < 1e-8] = np.nan
    abd[np.linalg.norm(local[:, :, [0, 1]], axis=-1) < 1e-8] = np.nan
    axis = unit(np.cross(d[:, :, 0], normal[:, None, :]))
    values = [flex, abd]
    for k in range(2):
        u = d[:, :, k]
        v = d[:, :, k + 1]
        u = unit(u - axis * np.sum(u * axis, axis=-1, keepdims=True))
        v = unit(v - axis * np.sum(v * axis, axis=-1, keepdims=True))
        values.append(
            np.degrees(
                np.arctan2(
                    np.sum(axis * np.cross(u, v), axis=-1), np.sum(u * v, axis=-1)
                )
            )
        )
    out = np.stack(values, axis=-1)
    out[:, 0, :] = np.nan
    return out


def angular_error(a, b):
    return np.abs((np.asarray(a) - np.asarray(b) + 180) % 360 - 180)
