"""Rebuild GRASP registered kinematics from the released Motive and MANUS CSVs.

The local-to-world composition follows the student implementation. The
distributed registered tier uses side-specific mounting rotations about Z.
"""
import csv
from pathlib import Path

import numpy as np
import pandas as pd


DIGITS = ("Thumb", "Index", "Middle", "Ring", "Pinky")
MOTIVE_BONES = (
    ["Skeleton001", "Ab", "Chest", "Neck", "Head"]
    + [side + name for side in "LR" for name in
       ["Shoulder", "UArm", "FArm", "Hand"] + [d + n for d in DIGITS for n in "123"]]
    + [side + name for side in "LR" for name in ("Thigh", "Shin", "Foot", "Toe")]
)
MANUS_NODES = sorted(["Hand"] + [
    f"{digit}_{joint}" for digit in DIGITS for joint in ("CMC", "MCP", "PIP", "DIP", "TIP")
    if (digit, joint) != ("Thumb", "PIP")
])
OUT_FIELDS = ("PosX", "PosY", "PosZ", "RotX", "RotY", "RotZ", "RotW")


def _read_motive(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        headers = [next(reader, []) for _ in range(8)]
    meta = dict(zip(headers[0][::2], headers[0][1::2]))
    try:
        rate = float(meta["Export Frame Rate"])
        exported = int(meta["Total Exported Frames"])
    except (KeyError, ValueError) as exc:
        raise ValueError(f"Missing or invalid Motive export rate/frame count: {path}") from exc
    if (meta.get("Length Units"), meta.get("Coordinate Space"), meta.get("Rotation Type")) != (
        "Millimeters", "Global", "Quaternion"
    ) or rate != 200:
        raise ValueError(f"Expected the release's global 200 Hz Motive export in mm: {path}")
    kinds, names, fields, axes = headers[2], headers[3], headers[6], headers[7]
    if axes[:2] != ["Frame", "Time (Seconds)"]:
        raise ValueError(f"Invalid Motive frame/time header: {path}")
    selected = [i for i, kind in enumerate(kinds) if kind == "Bone"]
    if len(selected) != 7 * len(MOTIVE_BONES):
        raise ValueError(f"Expected 51 Motive bones with seven channels each: {path}")
    if any(len(header) <= selected[-1] for header in (names, fields, axes)):
        raise ValueError(f"Incomplete Motive name, field or axis header: {path}")
    for k, bone in enumerate(MOTIVE_BONES):
        group = selected[7*k:7*k+7]
        labels = [names[i].rsplit(":", 1)[-1].replace(" ", "") for i in group]
        if (labels != [bone]*7 or [fields[i] for i in group] != ["Rotation"]*4 + ["Position"]*3
                or [axes[i] for i in group] != list("XYZWXYZ")):
            raise ValueError(f"Invalid Motive channels for {bone}: {path}")
    # Marker columns can be empty in the acquisition. Registration uses Bone channels only.
    data = pd.read_csv(path, skiprows=8, header=None, usecols=[0, 1, *selected],
                       encoding="utf-8-sig", dtype=np.float64).to_numpy()
    if (not len(data) or len(data) != exported or
            not np.array_equal(data[:, 0], np.arange(len(data))) or
            not np.allclose(data[:, 1], np.arange(len(data)) / 200., atol=1e-10, rtol=0) or
            not np.isfinite(data).all()):
        raise ValueError(f"Incomplete Motive frame sequence or missing skeleton measurement: {path}")
    values = data[:, 2:].reshape(len(data), len(MOTIVE_BONES), 7)
    return data[:, 1], values[:, :, 4:], values[:, :, :4]


def _read_manus(path):
    fields = ["Position_" + a for a in "XYZ"] + ["Rotation_" + a for a in "XYZW"]
    channels = [f"{node}_{field}" for node in MANUS_NODES for field in fields]
    selected = ["Frame", "Elapsed_Time_In_Milliseconds", *channels]
    table = pd.read_csv(path, usecols=selected, encoding="utf-8-sig", dtype=np.float64)
    index = np.arange(len(table))
    if (not len(table) or not np.array_equal(table["Frame"], index) or
            not np.array_equal(table["Elapsed_Time_In_Milliseconds"], index * 10)):
        raise ValueError(f"Invalid 100 Hz MANUS frame sequence: {path}")
    data = table[channels].to_numpy().reshape(len(table), len(MANUS_NODES), 7)
    if not np.isfinite(data).all():
        raise ValueError(f"Non-finite MANUS position or orientation: {path}")
    return data[:, :, :3], data[:, :, 3:]


def _unit(q):
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    if np.any(norm == 0):
        raise ValueError("Undefined orientation (zero quaternion)")
    return q / norm


def _multiply(parent, local):
    x1, y1, z1, w1 = np.moveaxis(_unit(parent), -1, 0)
    x2, y2, z2, w2 = np.moveaxis(_unit(local), -1, 0)
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    return np.stack([x, y, z, w], axis=-1)


def _rotate(q, vectors):
    x, y, z, w = np.moveaxis(_unit(q), -1, 0)
    vx, vy, vz = np.moveaxis(vectors, -1, 0)
    rx = (1-2*(y*y+z*z))*vx + 2*(x*y-w*z)*vy + 2*(x*z+w*y)*vz
    ry = 2*(x*y+w*z)*vx + (1-2*(x*x+z*z))*vy + 2*(y*z-w*x)*vz
    rz = 2*(x*z-w*y)*vx + 2*(y*z+w*x)*vy + (1-2*(x*x+y*y))*vz
    return np.stack([rx, ry, rz], axis=-1)


def build_table(motive_path: Path, manus_path: Path, hand: str) -> pd.DataFrame:
    """Register one released raw pair. Positions are mm, time seconds, rotations XYZW."""
    if hand not in ("L", "R"):
        raise ValueError("hand must be L or R")
    times, bone_pos, bone_rot = _read_motive(motive_path)
    local_pos, local_rot = _read_manus(manus_path)
    frames = min(len(times), 2 * len(local_pos))
    bone_pos, bone_rot = bone_pos[:frames], bone_rot[:frames]
    wrist = MOTIVE_BONES.index(hand + "Hand")
    wrist_pos = bone_pos[:, wrist:wrist+1]
    wrist_rot = bone_rot[:, wrist:wrist+1]
    theta = np.pi/2 if hand == "R" else -np.pi/2
    mounting = np.array([0.0, 0.0, np.sin(theta / 2), np.cos(theta / 2)])
    glove_index = np.arange(frames) // 2
    hand_pos = wrist_pos + _rotate(wrist_rot, _rotate(mounting, local_pos[glove_index]))
    hand_rot = _multiply(wrist_rot, _multiply(mounting, local_rot[glove_index]))

    values = np.concatenate([np.concatenate([bone_pos, bone_rot], axis=2).reshape(frames, -1),
                             np.concatenate([hand_pos, hand_rot], axis=2).reshape(frames, -1)], axis=1)
    columns = [f"{source}_{name}_{field}" for source, names in
               (("Motive", MOTIVE_BONES), ("Manus", MANUS_NODES)) for name in names for field in OUT_FIELDS]
    result = pd.DataFrame(values, columns=columns)
    result.insert(0, "Time", times[:frames])
    result.insert(0, "Frame", np.arange(frames))
    return result


def compare_table(table: pd.DataFrame, reference_path: Path) -> dict:
    """Compare all cells, preserving quaternion signs and the Motive time axis."""
    reference = pd.read_csv(reference_path, float_precision="round_trip", encoding="utf-8-sig")
    if not table.columns.equals(reference.columns) or table.shape != reference.shape:
        raise ValueError(f"Different registered schema or row count: {reference_path}")
    actual = table.to_numpy(dtype=np.float64)
    expected = reference.to_numpy(dtype=np.float64)
    if not np.isfinite(expected).all() or not np.isfinite(actual).all():
        raise ValueError(f"Non-finite registered measurement: {reference_path}")
    delta = np.abs(actual - expected)
    columns = table.columns
    pos = columns.str.endswith(("PosX", "PosY", "PosZ"))
    rot = columns.str.endswith(("RotX", "RotY", "RotZ", "RotW"))
    tol = np.zeros(len(columns))
    tol[pos & columns.str.startswith("Manus_")] = 5e-10
    tol[rot & columns.str.startswith("Manus_")] = 5e-13
    mismatches = delta > tol
    if mismatches.any():
        row, col = np.unravel_index(mismatches.argmax(), mismatches.shape)
        raise ValueError(f"Mismatch at frame {row}, {table['Time'].iat[row]} s, {columns[col]}: "
                         f"generated {actual[row,col]}, reference {expected[row,col]} ({reference_path})")
    return {"position_max_mm": float(delta[:, pos].max()), "quaternion_max_abs": float(delta[:, rot].max())}
