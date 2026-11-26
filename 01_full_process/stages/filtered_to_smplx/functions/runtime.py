"""Neutral SMPL-X runtime and canonical position targets."""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import smplx
import torch
from reset_rotation import aa2rot_torch
from reset_evaluation import BODY_NAMES, DIGITS

HERE = Path(__file__).resolve().parent
PARAMS = (
    "body_pose",
    "lhand_pose",
    "rhand_pose",
    "jaw_pose",
    "leye_pose",
    "reye_pose",
    "betas",
    "expression",
    "global_orient",
    "transl",
)


def config(name):
    return json.loads((HERE / name).read_text(encoding="utf-8"))


def load_model(model_path, device):
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = (
        smplx.SMPLXLayer(
            str(model_path),
            gender="neutral",
            ext="npz",
            num_betas=10,
            num_expression_coeffs=10,
            use_pca=False,
            flat_hand_mean=True,
            dtype=torch.float32,
            batch_size=1,
        )
        .to(device)
        .eval()
    )
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def forward(model, params):
    rot = aa2rot_torch(
        torch.cat(
            [
                params["global_orient"][:, None],
                params["body_pose"],
                params["jaw_pose"][:, None],
                params["leye_pose"][:, None],
                params["reye_pose"][:, None],
                params["lhand_pose"],
                params["rhand_pose"],
            ],
            dim=1,
        )
    )
    return model(
        betas=params["betas"],
        expression=params["expression"],
        global_orient=rot[:, 0],
        body_pose=rot[:, 1:22],
        jaw_pose=rot[:, 22],
        leye_pose=rot[:, 23],
        reye_pose=rot[:, 24],
        left_hand_pose=rot[:, 25:40],
        right_hand_pose=rot[:, 40:55],
        transl=params["transl"],
    )


def read_trial(path, side):
    df = pd.read_csv(path, float_precision="round_trip")
    rm = pd.read_csv(path.with_suffix(".rowmap.csv"), float_precision="round_trip")
    time = rm["source_time_s"].to_numpy()
    source_rows = rm["source_row0"].to_numpy()
    rows = np.arange(len(df))
    if not (
        len(rows)
        and df.shape[1] == 534
        and rm["output_row0"].dtype.kind in "iu"
        and source_rows.dtype.kind in "iu"
        and np.array_equal(rows, rm["output_row0"])
        and np.array_equal(rows, df["Frame"])
        and np.allclose(df["Time"], rows / 200.0, rtol=0, atol=1e-10)
        and np.array_equal(df["Time"], rm["Time"])
        and np.array_equal(df["Frame"], rm["Frame"])
        and np.isfinite(time).all()
        and np.all(source_rows >= 0)
        and np.all(np.diff(source_rows) > 0)
        and np.allclose(time, source_rows * 0.005, rtol=0, atol=1e-10)
    ):
        raise ValueError("Filtered CSV and original-time row map do not agree")

    def pos(name):
        # Match the native row-major layout, including the GPU reduction order.
        values = df[[name + "_Pos" + a for a in "XYZ"]].to_numpy(dtype=np.float64)
        return np.ascontiguousarray(values * 0.001)

    canonical = np.stack(
        [pos(n) for n in config("mapping.json")[side]["source"]], axis=1
    )
    body = np.stack([pos(n) for n in BODY_NAMES], axis=1)
    chains = np.stack(
        [
            np.stack(
                [
                    pos(f"Manus_{digit}_{k}")
                    for k in (
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
    valid = np.isfinite(df.to_numpy(dtype=float)).all(axis=1)
    for col in df.columns:
        if col.endswith("_RotX"):
            axes = [col[:-1] + c for c in "XYZW"]
            valid &= np.linalg.norm(df[axes].to_numpy(), axis=1) > 1e-8
    ids = np.flatnonzero(valid)
    if not len(ids):
        raise ValueError("No usable filtered rows")
    breaks = (
        np.flatnonzero(
            (np.diff(ids) != 1)
            | (np.diff(source_rows[ids]) != 1)
            | (np.diff(time[ids]) > 0.0075)
        )
        + 1
    )
    segments = [(int(group[0]), int(group[-1]) + 1) for group in np.split(ids, breaks)]
    return dict(
        canonical_positions_m=canonical,
        chains_m=chains,
        body_positions_m=body,
        wrist_m=pos("Manus_Hand"),
        time_s=df["Time"].to_numpy(),
        source_time_s=time,
        source_rows=source_rows,
        valid=valid,
    ), segments
