"""Read released filtered trajectories, topology and original acquisition times."""

import csv
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DIGITS = ("Thumb", "Index", "Middle", "Ring", "Pinky")
DIGIT_COLORS = ("#aa562a", "#187c82", "#435fab", "#8c538d", "#64803b")
PATTERN = re.compile(r"(.+)_([LR])_([1-9]|1[0-9]|20)_registered\.csv")


@dataclass(frozen=True)
class Trial:
    number: int
    grasp: str
    side: str
    take: int
    path: Path

    @property
    def name(self):
        return f"{self.grasp}_{self.side}_{self.take}"


def discover(root):
    root = Path(root).expanduser()
    if (root / "Registration_data_GRASP_filtered").is_dir():
        root = root / "Registration_data_GRASP_filtered"
    if root.name in {
        "Registration_data_GRASP",
        "Manus_data_GRASP",
        "Motive_data_GRASP",
    }:
        raise ValueError(
            "Select the filtered tier containing the motion CSVs and their .rowmap.csv files."
        )
    trials = []
    for p in root.glob("*/*/*_registered.csv"):
        m = PATTERN.fullmatch(p.name)
        if not m:
            continue
        if not p.with_suffix(".rowmap.csv").is_file():
            raise ValueError(
                f"{p.name} has no companion .rowmap.csv. Select a complete filtered tier."
            )
        grasp, side, take = m.groups()
        number, sep, suffix = p.parent.parent.name.partition(".")
        if (
            number.isdigit()
            and sep
            and suffix == grasp
            and p.parent.name == f"{grasp}_{side}"
            and 1 <= int(number) <= 33
        ):
            trials.append(Trial(int(number), grasp, side, int(take), p))
    if not trials:
        raise ValueError(
            "No filtered trials found. Select the extracted tier or its parent folder."
        )
    trials.sort(key=lambda t: (t.number, t.side, t.take))
    identifiers = [t.name for t in trials]
    if len(set(identifiers)) != len(trials):
        raise ValueError("Duplicate trial identifiers in the selected folder.")
    return trials


def topology(side):
    """Return source column names and connected line strips for both views."""
    names = []
    lines = []

    def add(points, color, hand=False):
        ids = []
        for p in points:
            if p not in names:
                names.append(p)
            ids.append(names.index(p))
        lines.append((ids, color, hand))

    def bones(chain, color):
        add(["Motive_" + joint for joint in chain], color)

    neutral = "#536975"
    bones(["Skeleton001", "Ab", "Chest", "Neck", "Head"], neutral)
    for s in ("L", "R"):
        bones(["Chest", s + "Shoulder", s + "UArm", s + "FArm", s + "Hand"], neutral)
        bones(["Skeleton001", s + "Thigh", s + "Shin", s + "Foot", s + "Toe"], neutral)
        if s != side:
            for digit in DIGITS:
                bones([s + "Hand", *[s + digit + str(i) for i in (1, 2, 3)]], "#83939b")
    for digit, color in zip(DIGITS, DIGIT_COLORS):
        segments = ["CMC", "MCP", "PIP", "DIP", "TIP"]
        if digit == "Thumb":
            segments.remove("PIP")
        add(
            ["Manus_Hand", *["Manus_" + digit + "_" + s for s in segments]], color, True
        )
    return names, lines


@dataclass
class Recording:
    trial: Trial
    points: np.ndarray
    times: np.ndarray
    source_rows: np.ndarray
    lines: list
    hand_ids: np.ndarray
    wrist: int
    body_center: np.ndarray
    body_radius: float
    body_basis: np.ndarray
    hand_center: np.ndarray
    hand_radius: float
    hand_basis: np.ndarray


def unit(v):
    norm = np.linalg.norm(v)
    if norm < 1e-8:
        raise ValueError("Degenerate direction in the initial skeleton.")
    return v / norm


def load(trial):
    names, lines = topology(trial.side)
    channels = [name + "_Pos" + axis for name in names for axis in "XYZ"]
    with trial.path.open(encoding="utf-8-sig", newline="") as stream:
        header = next(csv.reader(stream))
    indices = [header.index(name) for name in ["Frame", "Time", *channels]]
    data = np.loadtxt(trial.path, delimiter=",", skiprows=1, usecols=indices, ndmin=2)
    if not len(data) or not np.isfinite(data).all():
        raise ValueError(f"Empty or non-finite filtered coordinates in {trial.name}.")
    rows = np.arange(len(data))
    times = data[:, 1]
    if not np.array_equal(data[:, 0], rows) or np.any(np.diff(times) <= 0):
        raise ValueError("Invalid filtered frame sequence.")
    sidecar = trial.path.with_suffix(".rowmap.csv")
    with sidecar.open(encoding="utf-8-sig", newline="") as stream:
        header = next(csv.reader(stream))
    cols = [
        header.index(k)
        for k in ("output_row0", "Frame", "Time", "source_row0", "source_time_s")
    ]
    info = np.loadtxt(sidecar, delimiter=",", skiprows=1, usecols=cols, ndmin=2)
    if (
        len(info) != len(data)
        or not np.isfinite(info).all()
        or not np.array_equal(info[:, 0], rows)
        or not np.array_equal(info[:, 1], rows)
        or not np.allclose(info[:, 2], times, atol=1e-10, rtol=0)
        or not np.array_equal(info[:, 3], info[:, 3].astype(np.int64))
        or np.any(info[:, 3] < 0)
        or np.any(np.diff(info[:, 3]) <= 0)
        or not np.allclose(info[:, 4], info[:, 3] / 200, atol=1e-10, rtol=0)
    ):
        raise ValueError(f"Filtered CSV and row map disagree in {trial.name}.")
    rows, times = info[:, 3].astype(np.int64), info[:, 4]
    xyz = data[:, 2:].reshape(len(data), len(names), 3)
    hand_ids = np.array(
        [i for i, name in enumerate(names) if name.startswith("Manus_")]
    )
    wrist = names.index("Manus_Hand")
    up = np.array([0.0, 0.0, 1.0])
    right = (
        xyz[0, names.index("Motive_RShoulder")]
        - xyz[0, names.index("Motive_LShoulder")]
    )
    right = unit(right - up * np.dot(right, up))
    body_basis = np.stack([right, np.cross(up, right), up])
    direction = unit(xyz[0, names.index("Manus_Middle_MCP")] - xyz[0, wrist])
    across = (
        xyz[0, names.index("Manus_Index_MCP")] - xyz[0, names.index("Manus_Pinky_MCP")]
    )
    across = unit(across - direction * np.dot(across, direction))
    # Viewed from the palm; keep the basis fixed for the whole recording.
    normal = np.cross(direction, across)
    if trial.side == "L":
        across, normal = -across, -normal
    hand_basis = np.stack([across, normal, direction])
    body_lo, body_hi = xyz.min(axis=(0, 1)), xyz.max(axis=(0, 1))
    body_center = (body_lo + body_hi) / 2
    relative = xyz[:, hand_ids] - xyz[:, wrist : wrist + 1]
    hand_center = (relative.min(axis=(0, 1)) + relative.max(axis=(0, 1))) / 2
    return Recording(
        trial,
        xyz,
        times,
        rows,
        lines,
        hand_ids,
        wrist,
        body_center,
        max(1.0, np.linalg.norm(xyz - body_center, axis=-1).max()),
        body_basis,
        hand_center,
        max(1.0, np.linalg.norm(relative - hand_center, axis=-1).max()),
        hand_basis,
    )
