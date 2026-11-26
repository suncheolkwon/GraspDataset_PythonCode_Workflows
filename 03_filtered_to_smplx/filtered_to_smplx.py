"""Fit the release's native SMPL-X motion and make a full-body/hand MP4."""

import argparse
import hashlib
from contextlib import ExitStack
import os
import re
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "functions"))


def run(args):
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[key] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import torch
    import numpy as np
    from runtime import read_trial, load_model, config, PARAMS
    import body_seed, palm_fit, functional_fit
    from video import export_video, profile_for_trial

    match = re.fullmatch(r"([A-Za-z0-9_]+)_([LR])_(\d+)", args.trial)
    if not match or not 1 <= int(match[3]) <= 20:
        raise ValueError("Expected a trial such as Tripod_R_1")
    grasp, side, take = match.groups()
    catalog = config("catalog.json")
    if grasp not in catalog:
        raise ValueError(f"Unknown grasp: {grasp}")
    idx = catalog[grasp]["index"]
    relative = (
        Path(f"{idx}.{grasp}") / f"{grasp}_{side}" / (args.trial + "_registered.csv")
    )
    csv = args.data_root / relative
    if not csv.is_file():
        csv = args.data_root / "Registration_data_GRASP_filtered" / relative
    if not csv.is_file():
        raise FileNotFoundError(csv)
    if not args.model.is_file():
        raise FileNotFoundError(args.model)
    if args.output.suffix.lower() != ".mp4":
        raise ValueError("--output must name an MP4")
    if args.output.exists() or (
        args.save_parameters is not None and args.save_parameters.exists()
    ):
        raise ValueError("Output already exists. Select a new output filename.")
    if (
        args.save_parameters is not None
        and args.save_parameters.suffix.lower() != ".npz"
    ):
        raise ValueError("--save-parameters must name an NPZ")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError(
            "CUDA unavailable. Install a compatible PyTorch/CUDA environment, or select --device cpu."
        )
    device = torch.device(args.device)
    provenance = {}
    if args.save_parameters:
        for label, file in [
            ("model", args.model),
            ("filtered_csv", csv),
            ("source_rowmap", csv.with_suffix(".rowmap.csv")),
        ]:
            with file.open("rb") as stream:
                provenance[label + "_sha256"] = hashlib.file_digest(
                    stream, "sha256"
                ).hexdigest()
    s, segments = read_trial(csv, side)
    model = load_model(args.model, device)
    opts = config("optimizer.json")
    baseline = body_seed.fit(model, s, segments, side, device)
    baseline = palm_fit.fit(model, s, baseline, segments, opts["P33"], side, device)
    fit = baseline
    for stage in ("F02A", "F03A"):
        fit = functional_fit.fit(
            stage,
            model,
            s,
            baseline,
            fit,
            segments,
            opts[stage],
            side,
            device,
            config("capsules.json")[side],
            catalog[grasp]["policy"],
        )
    if not all(np.isfinite(fit[k][s["valid"]]).all() for k in PARAMS):
        raise ValueError("Nonfinite fitted parameters")
    with ExitStack() as stack:
        video_dir = stack.enter_context(
            tempfile.TemporaryDirectory(prefix=".smplx-", dir=args.output.parent)
        )
        video = Path(video_dir) / "sample.mp4"
        export_video(model, fit, s, side, idx, grasp, int(take), video)
        staged = []
        if args.save_parameters:
            args.save_parameters.parent.mkdir(parents=True, exist_ok=True)
            pose_dir = stack.enter_context(
                tempfile.TemporaryDirectory(
                    prefix=".smplx-", dir=args.save_parameters.parent
                )
            )
            pose = Path(pose_dir) / "parameters.npz"
            np.savez_compressed(
                pose,
                **fit,
                **provenance,
                trial_id=args.trial,
                finger_width_scale=0.7,
                finger_depth_scale=0.7,
                palm_width_scale=profile_for_trial(int(take))["palm_width_scale"],
                source_time_s=s["source_time_s"],
                source_row0=s["source_rows"],
                valid=s["valid"],
                hand=np.array(side),
            )
            staged.append((pose, args.save_parameters))
        staged.append((video, args.output))
        installed = []
        try:
            for temporary, destination in staged:
                os.link(temporary, destination)
                installed.append((temporary, destination))
        except BaseException:
            for temporary, destination in installed:
                if destination.exists() and temporary.samefile(destination):
                    destination.unlink()
            raise
    print(args.output.resolve(), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input-dir",
        "--data-root",
        dest="data_root",
        type=Path,
        required=True,
        help="Directory containing the extracted filtered tier, or the tier itself.",
    )
    p.add_argument(
        "--trial", required=True, help="Exact recording identifier, e.g. Tripod_R_1"
    )
    p.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Your licensed neutral SMPL-X NPZ file",
    )
    output = p.add_mutually_exclusive_group(required=True)
    output.add_argument(
        "--output-dir", type=Path, help="Write <trial>.mp4 to this directory."
    )
    output.add_argument(
        "--output", type=Path, help="New output MP4 path, with a custom filename."
    )
    p.add_argument(
        "--save-parameters",
        type=Path,
        help="Optional native 200 Hz SMPL-X parameters and source times (.npz)",
    )
    p.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default="cuda",
        help="Fitting device (reference: cuda)",
    )
    args = p.parse_args()
    if args.output_dir is not None:
        args.output = args.output_dir / (args.trial + ".mp4")
    try:
        run(args)
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        p.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
