"""Fixed-link reconstruction and XYZW quaternion operations for the GRASP chains."""
import numpy as np

EPS = 1e-10

def qnorm(q, context="quaternion"):
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    if not np.isfinite(q).all() or np.any(n <= EPS):
        bad = np.argwhere((~np.isfinite(q).all(-1)) | (n[..., 0] <= EPS))
        raise ValueError(f"INVALID_{context}: indices={bad[:20].tolist()}, count={len(bad)}")
    return q / n


def qconj(q):
    out = q.copy()
    out[..., :3] *= -1
    return out


def qmul(a, b):
    av, bv = a[..., :3], b[..., :3]
    aw, bw = a[..., 3:4], b[..., 3:4]
    return np.concatenate((aw * bv + bw * av + np.cross(av, bv),
                           aw * bw - np.sum(av * bv, axis=-1, keepdims=True)), axis=-1)


def qapply(q, v):
    return v + 2 * np.cross(q[..., :3], np.cross(q[..., :3], v) + q[..., 3:4] * v)


def topological_order(parents):
    parents = np.asarray(parents, dtype=int)
    pending = set(range(len(parents)))
    order = []
    while pending:
        ready = sorted(i for i in pending if parents[i] < 0 or parents[i] in order)
        if not ready:
            raise ValueError("CYCLIC_OR_INVALID_PARENT_TOPOLOGY")
        order.extend(ready)
        pending.difference_update(ready)
    return order


def classify_link(parent, child):
    if child.startswith("Manus_") and child != "Manus_Hand":
        return "MANUS_HAND", True, "MANUS digit chain"
    for side in "LR":
        if (parent, child) == (f"Motive_{side}UArm", f"Motive_{side}FArm"):
            return "UPPER_ARM", True, "Named humerus segment with parent identity resolved from optical headers"
        if (parent, child) == (f"Motive_{side}FArm", f"Motive_{side}Hand"):
            return "FOREARM", True, "Named forearm segment with parent identity resolved from optical headers"
        if (parent, child) in [(f"Motive_{side}Thigh", f"Motive_{side}Shin"),
                              (f"Motive_{side}Shin", f"Motive_{side}Foot"),
                              (f"Motive_{side}Foot", f"Motive_{side}Toe")]:
            return "OTHER_FIXED_BODY", True, "Named thigh, shank, or foot-to-toe segment; optical solver geometry, not independent anatomy measurement"
    if child == "Manus_Hand":
        return "WRIST_ATTACHMENT", False, "Observed MotiveHand to ManusHand registration offset, not an anatomical bone"
    if any(child.startswith(f"Motive_{side}{digit}") for side in "LR"
           for digit in ["Thumb", "Index", "Middle", "Ring", "Pinky"]):
        return "OPTICAL_AUXILIARY", False, "Optical solver finger auxiliary; not independently measured MANUS digit"
    return "UNCONFIRMED_EXPORT", False, "Export-origin or girdle/spine connection without confirmed fixed anatomical segment meaning"


def get_link_policy(data):
    names, parents = data["names"], np.asarray(data["parents"])
    result = []
    for child, parent in enumerate(parents):
        if parent < 0:
            continue
        group, fixed, reason = classify_link(names[parent], names[child])
        result.append(dict(child_index=int(child), parent_index=int(parent),
                           child=names[child], parent=names[parent],
                           link=names[parent] + "->" + names[child],
                           group=group, fixed=fixed, policy=reason))
    if sum(x["group"] == "MANUS_HAND" for x in result) != 24:
        raise ValueError("EXPECTED_EXACTLY_24_MANUS_LINKS")
    if sum(x["fixed"] for x in result) != 34:
        raise ValueError("EXPECTED_34_FIXED_LINKS_24_HAND_10_BODY")
    return result


def continuous_quaternions(values, segments):
    """Normalize and choose quaternion signs continuously inside each retained segment."""
    out = np.array(values, dtype=float, copy=True)
    for first, stop in segments:
        x = qnorm(out[first:stop], "SOURCE_RELATIVE_QUATERNION")
        if len(x) > 1:
            flip = np.where(np.sum(x[1:] * x[:-1], axis=-1) < 0, -1.0, 1.0)
            signs = np.concatenate((np.ones_like(flip[:1]), np.cumprod(flip, axis=0)), axis=0)
            x *= signs[..., None]
        out[first:stop] = qnorm(x, "FILTERED_QUATERNION")
    return out


def length_baseline(data):
    """Project fixed links onto their trial-mean lengths in measured directions."""
    src = data["source"]
    pc, qc = data["pc"], data["qc"]
    parents = data["parents"]
    xyz, rawq = src[:, pc], src[:, qc]
    q = qnorm(rawq, "SOURCE_QUATERNION")
    order = topological_order(parents)
    links = get_link_policy(data)
    ci = np.array([r["child_index"] for r in links], dtype=int)
    pi = parents[ci]
    fixed = np.array([r["fixed"] for r in links], dtype=bool)
    vectors = xyz[:, ci] - xyz[:, pi]
    lengths = np.linalg.norm(vectors, axis=-1)
    local = qapply(qconj(q[:, pi]), vectors)
    reference = np.mean(lengths, axis=0)
    if np.any(lengths[:, fixed] <= EPS):
        raise ValueError("Degenerate fixed source length")
    localq = q.copy()
    localq[:, ci] = qmul(qconj(q[:, pi]), q[:, ci])
    newlocalq = continuous_quaternions(localq, data["segments"])
    directions = local[:, fixed] / lengths[:, fixed, None]
    norms = np.linalg.norm(directions, axis=-1, keepdims=True)
    if not np.isfinite(norms).all() or np.any(norms <= EPS):
        raise ValueError("Degenerate measured bone direction")
    directions /= norms
    newlocal = local.copy()
    newlocal[:, fixed] = directions * reference[None, fixed, None]
    newxyz = np.empty_like(xyz)
    newq = np.empty_like(q)
    edge_index = {int(c): j for j, c in enumerate(ci)}
    for child in order:
        parent = parents[child]
        if parent < 0:
            newxyz[:, child] = xyz[:, child]
            newq[:, child] = newlocalq[:, child]
        else:
            newxyz[:, child] = newxyz[:, parent] + qapply(newq[:, parent], newlocal[:, edge_index[child]])
            newq[:, child] = qnorm(qmul(newq[:, parent], newlocalq[:, child]), "FK_QUATERNION")
    output = src.copy()
    output[:, 0] = np.arange(len(src))
    output[:, 1] = np.arange(len(src)) / 200.0
    output[:, pc] = newxyz
    output[:, qc] = newq
    for j, link in enumerate(links):
        link["reference_length_mm"] = float(reference[j])
    return output, {"links": links}
