"""Read measured XYZ directly from the public filtered CSV and restore its original clock."""

import csv
import re
import numpy as np
from .model import DIGITS


def locate(root, trial):
    match = re.fullmatch(r"([A-Za-z0-9_]+)_([LR])_([1-9]|1[0-9]|20)", trial)
    if not match:
        raise ValueError("Expected a trial such as Tripod_R_1.")
    grasp, hand, repetition = match.groups()
    if (root / "Registration_data_GRASP_filtered").is_dir():
        root = root / "Registration_data_GRASP_filtered"
    paths = list(root.glob(f"*.{grasp}/{grasp}_{hand}/{trial}_registered.csv"))
    if len(paths) != 1:
        raise ValueError(f"Expected one filtered CSV for {trial} in {root}, found {len(paths)}")
    path = paths[0]
    return (
        path,
        int(path.parents[1].name.split(".", 1)[0]),
        grasp,
        ("right" if hand == "R" else "left"),
        int(repetition),
    )


def load(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        r = csv.reader(f)
        header = next(r)
        names = [c[:-5] for c in header if c.endswith("_PosX")]
        index = [header.index(n + "_Pos" + ax) for n in names for ax in "XYZ"]
        time = header.index("Time")
        frame = header.index("Frame")
        if len(set(header)) != len(header):
            raise ValueError("Duplicate CSV header")
        data = np.array([[float(row[i]) for i in [frame, time, *index]] for row in r])
    if len(data) < 2 or not np.isfinite(data).all():
        raise ValueError("Missing or non-finite motion samples")
    output_rows = np.arange(len(data))
    if not np.array_equal(data[:, 0], output_rows) or not np.allclose(
        data[:, 1], output_rows / 200.0, rtol=0, atol=1e-9
    ):
        raise ValueError("Filtered Frame and Time must start at zero and advance at 200 Hz")
    xyz = dict(zip(names, (data[:, 2:].reshape(len(data), len(names), 3) * 0.001).transpose(1, 0, 2)))
    body = {n.removeprefix("Motive_"): p for n, p in xyz.items() if n.startswith("Motive_")}
    wrist = xyz["Manus_Hand"]
    chains = np.stack(
        [
            np.stack(
                [
                    xyz[f"Manus_{digit}_{joint}"]
                    for joint in (
                        ("CMC", "MCP", "DIP", "TIP")
                        if digit == "Thumb"
                        else ("MCP", "PIP", "DIP", "TIP")
                    )
                ],
                axis=1,
            )
            for digit in DIGITS
        ],
        axis=1,
    )
    sidecar = path.with_suffix(".rowmap.csv")
    with sidecar.open(encoding="utf-8-sig", newline="") as f:
        records = list(csv.DictReader(f))
    indices = np.array([int(row["source_row0"]) for row in records])
    times = np.array([float(row["source_time_s"]) for row in records])
    mapped_time = np.array([float(row["Time"]) for row in records])
    if (
        len(records) != len(data)
        or any(
            int(r["output_row0"]) != i
            or int(r["Frame"]) != data[i, 0]
            for i, r in enumerate(records)
        )
        or not np.allclose(mapped_time, data[:, 1], rtol=0, atol=1e-9)
        or indices[0] < 0
        or np.any(np.diff(indices) != 1)
        or not np.allclose(times, indices / 200.0, rtol=0, atol=1e-9)
    ):
        raise ValueError("Expected aligned, contiguous 200 Hz filtered CSV and row map")
    return body, wrist, chains, times, indices
