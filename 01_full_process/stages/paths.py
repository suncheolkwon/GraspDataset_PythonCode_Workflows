"""Match raw recordings and publish the products of one complete trial."""

import os
from pathlib import Path


def raw_directories(args):
    if args.raw_dir is not None:
        if args.manus_dir is not None or args.motive_dir is not None:
            raise ValueError("Use either --raw-dir or the pair --manus-dir and --motive-dir.")
        manus = args.raw_dir / "Manus_data_GRASP"
        motive = args.raw_dir / "Motive_data_GRASP"
    elif args.manus_dir is not None and args.motive_dir is not None:
        manus, motive = args.manus_dir, args.motive_dir
    else:
        raise ValueError("Supply --raw-dir or both --manus-dir and --motive-dir.")
    manus, motive = manus.resolve(), motive.resolve()
    for tier in (manus, motive):
        if not tier.is_dir():
            raise ValueError(f"Missing raw folder: {tier}")
    if manus.is_relative_to(motive) or motive.is_relative_to(manus):
        raise ValueError("The MANUS and Motive inputs must be separate folders.")
    return manus, motive


def raw_path(trial):
    return trial.relative.with_name(trial.name + ".csv")


def select_trials(records, requested, manus, motive):
    if requested:
        if len(requested) != len(set(requested)):
            raise ValueError("Each --trial must be unique.")
        for name in requested:
            if name not in records:
                raise ValueError(f"Unknown trial: {name}")
            for tier in (manus, motive):
                file = tier / raw_path(records[name])
                if not file.is_file():
                    raise ValueError(f"Missing raw file: {file}")
        return [records[name] for name in requested]

    known = {raw_path(trial): trial for trial in records.values()}
    inventories = []
    for tier in (manus, motive):
        paths = {p.relative_to(tier) for p in tier.rglob("*.csv")
                 if p.is_file() and not p.name.startswith(".")}
        unexpected = paths - known.keys()
        if unexpected:
            raise ValueError(f"Unexpected CSV in {tier}: {min(unexpected)}")
        inventories.append(paths)
    left, right = inventories
    for tier, missing in ((manus, right - left), (motive, left - right)):
        if missing:
            raise ValueError(f"Missing {len(missing)} raw file(s) in {tier}, including {min(missing)}")
    if not left:
        raise ValueError("No matching raw CSVs found. Keep the numbered grasp folders from the release.")
    return [trial for path, trial in known.items() if path in left]


def products(trial, target, save_motion):
    """Return (relative staging path, relative output path) in publication order."""
    paths = []
    if target in ("registered", "all"):
        paths.append(Path("registered") / trial.relative)
    if target in ("filtered", "all"):
        motion = Path("filtered") / trial.relative
        paths.extend((motion.with_suffix(".rowmap.csv"), motion))
    for model in ("smplx", "mujoco"):
        if target in (model, "all"):
            if save_motion:
                paths.append(Path(model) / (trial.name + ".npz"))
            paths.append(Path(model) / (trial.name + ".mp4"))
    return [(path, path if target == "all" else path.relative_to(path.parts[0]))
            for path in paths]


def check_destination(output, relative):
    file = output / relative
    if file != file.resolve():
        raise ValueError(f"Output has a symlink in its path: {file}")
    if os.path.lexists(file):
        raise ValueError(f"Output already exists: {file}. Select a new output folder or different trials.")


def publish(stage, output, files):
    """All stages succeeded. Link each new file without overwriting existing work."""
    created = []
    try:
        for origin, relative in files:
            check_destination(output, relative)
            source, destination = stage / origin, output / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.link(source, destination)
            created.append((source, destination))
    except BaseException:
        for source, destination in reversed(created):
            if destination.exists() and source.samefile(destination):
                destination.unlink()
        raise
