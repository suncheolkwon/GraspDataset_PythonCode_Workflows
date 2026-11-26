"""Run H1 + Shadow Hand registration, inverse kinematics and video rendering for one trial."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile


def publish_pair(video, output, coordinates, trajectory):
    """Publish a complete render, rolling back this run's trajectory if the video link fails."""
    os.link(coordinates, trajectory)
    try:
        os.link(video, output)
    except BaseException:
        try:
            if trajectory.samefile(coordinates):
                trajectory.unlink()
        except FileNotFoundError:
            pass
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        "--data-root",
        dest="input_dir",
        metavar="FILTERED",
        type=Path,
        required=True,
        help="Extracted filtered dataset or its parent directory",
    )
    parser.add_argument("--trial", required=True, help="For example Tripod_R_1")
    outputs = parser.add_mutually_exclusive_group(required=True)
    outputs.add_argument("--output-dir", type=Path, metavar="VIDEOS", help="Write <trial>.mp4 here")
    outputs.add_argument("--output", type=Path, metavar="FILE.mp4", help="Use this exact MP4 path")
    parser.add_argument(
        "--trajectory", type=Path, help="Optionally save native 200 Hz joint coordinates as NPZ"
    )
    args = parser.parse_args()
    from grasp_robot.source import locate, load

    try:
        path, number, grasp, active, repetition = locate(args.input_dir, args.trial)
    except ValueError as exc:
        parser.error(str(exc))
    if args.output_dir is not None:
        args.output = args.output_dir / f"{args.trial}.mp4"
    if args.output.suffix.lower() != ".mp4" or (
        args.trajectory and args.trajectory.suffix.lower() != ".npz"
    ):
        parser.error("Use .mp4 for the video and .npz for the optional trajectory.")
    destinations = [args.output] + ([args.trajectory] if args.trajectory else [])
    for dst in destinations:
        if dst.exists():
            parser.error(f"Output already exists: {dst}")
        dst.parent.mkdir(parents=True, exist_ok=True)
    if sys.platform.startswith("linux"):
        os.environ.setdefault("MUJOCO_GL", "egl")
        os.environ.setdefault("EGL_PLATFORM", "surfaceless")
    import numpy as np
    import mujoco
    from grasp_robot.model import MODEL
    from grasp_robot.body import fit_body
    from grasp_robot.hand import fit_hand
    from grasp_robot.render import write_video

    body, wrist, chains, time, indices = load(path)
    print(f"{args.trial}: fitting body on {len(time)} measured frames", flush=True)
    q = fit_body(body, wrist, chains, time, active, args.trial, number)
    print("Fitting the measured hand", flush=True)
    addr, fingers, err = fit_hand(wrist, chains, active, args.trial)
    q[:, addr] = fingers
    if args.trial != "Tripod_R_1":
        from grasp_robot.clearance import correct

        q = correct(q, active)
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    limited = np.flatnonzero(model.jnt_limited)
    joints = q[:, model.jnt_qposadr[limited]]
    if (
        not np.isfinite(q).all()
        or np.any(joints < model.jnt_range[limited, 0] - 1e-6)
        or np.any(joints > model.jnt_range[limited, 1] + 1e-6)
    ):
        raise ValueError("Non-finite pose or robot joint outside physical limits")
    from grasp_robot.verify import evaluate

    verified_error, palm_error, overlap = evaluate(model, q, active, wrist, chains)
    if not np.allclose(err, verified_error, atol=1e-8, rtol=0):
        raise ValueError("Merged trajectory and local finger solve disagree")
    frame_ids = np.arange(0, len(time), 4)
    title = f"{number:02d} {grasp.replace('_',' ')}  |  {active.title()} Hand  |  Trial {repetition:02}"
    print(f"Rendering {len(frame_ids)} video frames", flush=True)
    with tempfile.TemporaryDirectory(dir=args.output.parent, prefix=".mujoco-") as tmp:
        video = Path(tmp) / "video.mp4"
        write_video(video, q[frame_ids], title, active)
        if args.trajectory:
            with tempfile.TemporaryDirectory(dir=args.trajectory.parent, prefix=".joints-") as td:
                file = Path(td) / "joints.npz"
                np.savez_compressed(
                    file,
                    qpos=q,
                    source_time_s=time,
                    registered_rows=indices,
                    video_source_indices=frame_ids,
                    trial=args.trial,
                    source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    joint_names=[model.joint(i).name for i in range(model.njnt)],
                    joint_qpos_addresses=model.jnt_qposadr,
                    tip_distance_error_mm=verified_error,
                    palm_normal_error_deg=palm_error,
                    collision_depth_mm=overlap,
                )
                publish_pair(video, args.output, file, args.trajectory)
        else:
            os.link(video, args.output)
    print(
        json.dumps(
            {
                "trial": args.trial,
                "native_frames": len(q),
                "video_frames": len(frame_ids),
                "palm_normal_mean_deg": float(palm_error.mean()),
                "max_self_overlap_mm": float(overlap.max()),
                "tip_distance_mae_mm": float(err.mean()),
                "tip_distance_max_mm": float(err.max()),
                "video": str(args.output),
            }
        )
    )


if __name__ == "__main__":
    main()
