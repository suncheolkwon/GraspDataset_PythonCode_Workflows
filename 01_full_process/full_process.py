"""Build GRASP motion data or model videos directly from a pair of raw folders."""

import argparse
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

from stages.paths import (
    check_destination, products, publish, raw_directories, raw_path, select_trials,
)


STAGES = Path(__file__).resolve().parent / "stages"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-dir", type=Path,
                   help="Parent containing the extracted Manus_data_GRASP and Motive_data_GRASP.")
    p.add_argument("--manus-dir", type=Path, help="Direct MANUS folder when using separate input paths.")
    p.add_argument("--motive-dir", type=Path, help="Direct Motive folder when using separate input paths.")
    p.add_argument("--output-dir", type=Path, required=True, help="Where to create the selected products.")
    p.add_argument("--target", choices=("all", "registered", "filtered", "smplx", "mujoco"),
                   default="all", help="Default: all four products. A single target only saves that product.")
    p.add_argument("--trial", action="append", metavar="TRIAL",
                   help="Exact trial ID (repeatable). By default process every available raw pair.")
    p.add_argument("--model", type=Path, help="Neutral SMPL-X NPZ, required for smplx or all.")
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda", help="SMPL-X fitting device.")
    p.add_argument("--save-motion", action="store_true", help="Also save the native model motion beside each MP4.")
    return p.parse_args()


def render(model, stage, trial, args):
    script = STAGES / f"filtered_to_{model}" / f"filtered_to_{model}.py"
    command = [sys.executable, "-B", str(script), "--input-dir", str(stage / "filtered"),
               "--trial", trial.name, "--output-dir", str(stage / model)]
    if model == "smplx":
        command.extend(("--model", str(args.model), "--device", args.device))
    if args.save_motion:
        option = "--save-parameters" if model == "smplx" else "--trajectory"
        command.extend((option, str(stage / model / (trial.name + ".npz"))))
    subprocess.run(command, check=True)


def process(trial, manus, motive, stage, args):
    from registration import build_table
    from grasp_filter.trials import convert

    source = raw_path(trial)
    registered = stage / "registered" / trial.relative
    registered.parent.mkdir(parents=True, exist_ok=True)
    table = build_table(motive / source, manus / source, trial.side)
    if len(table) != trial.rows:
        raise ValueError(f"{trial.name}: expected {trial.rows} registered frames, obtained {len(table)}.")
    table.to_csv(registered, index=False)
    if args.target == "registered":
        return
    # Read serialized registration with the same float parser as the standalone filter.
    frame, rowmap = convert(registered, trial)
    filtered = stage / "filtered" / trial.relative
    filtered.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(filtered, index=False, float_format="%.17g")
    rowmap.to_csv(filtered.with_suffix(".rowmap.csv"), index=False, float_format="%.17g")
    for model in ("smplx", "mujoco"):
        if args.target in (model, "all"):
            render(model, stage, trial, args)


def main():
    args = parse_args()
    manus, motive = raw_directories(args)
    output = args.output_dir.resolve()
    protected = {manus, motive}
    for parent in {manus.parent, motive.parent}:
        protected.update(path.resolve() for path in parent.glob("Registration_data_GRASP*") if path.is_dir())
    if any(output.is_relative_to(tier) or tier.is_relative_to(output) for tier in protected):
        raise ValueError("Choose an output folder separate from the input and distributed data tiers.")
    if args.target in ("all", "smplx") and (not args.model or not args.model.is_file()):
        raise ValueError("For this target, supply --model with the path to your SMPLX_NEUTRAL.npz.")
    if args.save_motion and args.target in ("registered", "filtered"):
        raise ValueError("--save-motion applies to model videos.")
    sys.path.insert(0, str(STAGES / "raw_to_registered"))
    sys.path.insert(0, str(STAGES / "registered_to_filtered"))
    from grasp_filter.trials import catalog

    trials = select_trials(catalog(), args.trial, manus, motive)
    for trial in trials:
        for _, relative in products(trial, args.target, args.save_motion):
            check_destination(output, relative)
    output.mkdir(parents=True, exist_ok=True)
    for index, trial in enumerate(trials, 1):
        print(f"[{index}/{len(trials)}] {trial.name}", flush=True)
        files = products(trial, args.target, args.save_motion)
        with TemporaryDirectory(dir=output, prefix=".grasp-") as temporary:
            stage = Path(temporary)
            process(trial, manus, motive, stage, args)
            publish(stage, output, files)
        for _, file in files:
            print(output / file, flush=True)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        sys.exit(exc.returncode)
    except (ImportError, OSError, ValueError) as exc:
        sys.exit(str(exc))
