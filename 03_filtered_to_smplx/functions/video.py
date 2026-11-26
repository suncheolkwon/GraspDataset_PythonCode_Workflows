"""Reconstruct the measured-hand surface and render on the original source clock."""

from dataclasses import dataclass
import subprocess

import imageio_ffmpeg
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from raster import render
from runtime import PARAMS, config
from shape_adapter import JOINTS, WRIST, MorphologyAdapter


def frame_map(source):
    times = np.asarray(source["source_time_s"])
    grid = (
        times[0] + np.arange(int(np.floor((times[-1] - times[0]) * 50 + 1e-7)) + 1) / 50
    )
    hi = np.clip(np.searchsorted(times, grid), 0, len(times) - 1)
    lo = np.maximum(hi - 1, 0)
    ids = np.where(np.abs(times[lo] - grid) <= np.abs(times[hi] - grid), lo, hi)
    valid = (np.abs(times[ids] - grid) <= 0.00250001) & source["valid"][ids]
    if not valid.any():
        raise ValueError("No valid source samples lie on the 50 Hz display clock")
    return ids, valid


def profile_for_trial(take):
    # All 33 grasps use dataset trial suffixes 1–10 for S01 and 11–20 for S02.
    if not 1 <= take <= 20:
        raise ValueError("Expected trial number 1 through 20")
    return config("morphology.json")["S01" if take <= 10 else "S02"]


def geometry(model, fit, source, side, take):
    ids, valid = frame_map(source)
    profile = profile_for_trial(take)
    model = model.to("cpu")
    model.joint_mapper = None
    shaped = MorphologyAdapter(model, profile, side)
    faces = np.asarray(model.faces, dtype=np.int32)
    weights = model.lbs_weights.detach().numpy()
    wrist = WRIST[side]
    digits = [joint for chain in JOINTS[side].values() for joint in chain]
    hand = shaped.support_mask | np.isin(weights.argmax(axis=1), [wrist, *digits])
    forearm = hand | (weights[:, [wrist - 2, wrist, *digits]].sum(1) > 0.40)
    vertices = np.full((len(ids), len(model.v_template), 3), np.nan, dtype=np.float32)
    joints = None
    slots = np.flatnonzero(valid)
    with torch.inference_mode():
        for begin in range(0, len(slots), 16):
            where = slots[begin : begin + 16]
            mesh = shaped.forward({key: fit[key][ids[where]] for key in PARAMS})
            positions, skeleton = mesh.vertices.numpy(), mesh.joints.numpy()
            if not np.isfinite(positions).all() or not np.isfinite(skeleton).all():
                raise ValueError("Nonfinite geometry for a valid source row")
            if joints is None:
                joints = np.full((len(ids), skeleton.shape[1], 3), np.nan, np.float32)
            joints[where], vertices[where] = skeleton, positions
    return (
        vertices,
        joints,
        faces,
        np.flatnonzero(hand),
        faces[forearm[faces].any(1)],
        valid,
    )


def title_font(title):
    def load(size):
        for face, index in (("NotoSansCJK-Medium.ttc", 1), ("DejaVuSans.ttf", 0)):
            try:
                return ImageFont.truetype(face, size, index=index)
            except OSError:
                continue
        return ImageFont.load_default(size=size)

    font = load(34)
    return (
        load(int(34 * 885 / font.getlength(title)))
        if font.getlength(title) > 885
        else font
    )


def render_sequence(
    vertices, joints, faces, hand, arm_faces, valid_frames, title, side, path
):
    first_valid = int(np.flatnonzero(valid_frames)[0])
    frame_count = len(vertices)
    font = title_font(title)
    wrist_id = WRIST[side]
    other_mcps = [
        JOINTS[side][digit][0] for digit in ("index", "middle", "pinky", "ring")
    ]
    index_mcp, little_mcp = other_mcps[0], other_mcps[2]

    def unit(vector):
        return vector / np.maximum(
            np.linalg.norm(vector, axis=-1, keepdims=True), 1e-12
        )

    def basis(eye, preferred_up):
        eye = unit(np.asarray(eye, dtype=np.float32))
        right = unit(np.cross(preferred_up, eye))
        up = unit(np.cross(eye, right))
        return np.asarray([right, up, eye], dtype=np.float32).T.copy()

    def normals(v):
        a, b, c = (v[faces[:, k]] for k in range(3))
        contribution = np.cross(b - a, c - a)
        ns = np.empty((len(v), 3), np.float32)
        for d in range(3):
            ns[:, d] = sum(
                np.bincount(faces[:, k], weights=contribution[:, d], minlength=len(v))
                for k in range(3)
            )
        return unit(ns)

    key_direction = unit(np.array([-0.49, 0.63, 0.62], np.float32))
    fill_direction = unit(np.array([0.63, 0.26, 0.42], np.float32))

    def light_visibility(pos, indices, direction, bias):
        axes = basis(direction, [0, 1, 0])
        light = pos @ axes
        involved = np.unique(indices)
        xy = light[:, :2]
        low = xy[involved].min(0)
        high = xy[involved].max(0)
        size = 480
        scale = (size - 4) / max(high - low)
        uv = (xy - low) * scale + 2
        coords = np.clip(np.floor(uv).astype(np.int32), 0, size - 1)
        depth = light[:, 2]
        buff = np.zeros((size, size, 3), np.uint8)
        z = render(buff, uv, depth, np.zeros_like(pos), indices)
        gap = z[coords[:, 1], coords[:, 0]] - depth
        return 1 - np.clip((gap - bias) / (bias * 4), 0, 1)

    def lighting(surface_normals, pos, indices, hand_view):
        to_light = np.maximum(surface_normals @ key_direction, 0)
        bias = 0.0016 if hand_view else 0.005
        v1 = light_visibility(pos, indices, key_direction, bias)
        v2 = light_visibility(
            pos, indices, unit(np.array([-0.31, 0.64, 0.71], np.float32)), bias
        )
        key = to_light * (0.31 + 0.345 * (v1 + v2))
        fill = np.maximum(surface_normals @ fill_direction, 0)
        half_angle = unit(key_direction + np.array([0, 0, 1], np.float32))
        specular = np.maximum(surface_normals @ half_angle, 0) ** 27
        diffuse = 0.62 + 0.34 * key + 0.115 * fill
        color = np.array([205, 188, 171], np.float32)[None, :] * diffuse[:, None]
        color += (specular * 10 * v1)[:, None]
        return color

    @dataclass
    class Camera:
        axes: np.ndarray
        origins: np.ndarray
        target_xy: np.ndarray
        scale: float
        mid_pixel: tuple[int, int]

    def palm_rotations_to_first_frame():
        # MCP bases are attached to the palm. Their positions are independent of finger flexion.
        j = joints.astype(np.float64)
        across_palm = unit(j[:, index_mcp] - j[:, little_mcp])  # index and little MCP
        towards_fingers = j[:, other_mcps].mean(axis=1) - j[:, wrist_id]
        normal = unit(np.cross(across_palm, towards_fingers))
        palm = np.stack([across_palm, np.cross(normal, across_palm), normal], axis=-1)
        return palm @ palm[first_valid].T

    def cameras(hand_eye):
        body_axes = basis([0.82, 1.00, 0.17], [0, 0, 1])
        initial_hand_axes = basis(hand_eye, [0, 1, 0])
        hand_axes = (palm_rotations_to_first_frame() @ initial_hand_axes).astype(
            np.float32
        )
        wrist = joints[:, wrist_id]
        initial_hand = (
            (vertices[first_valid, hand] - wrist[first_valid]) @ initial_hand_axes
        )[:, :2]
        hand_center = (initial_hand.min(0) + initial_hand.max(0)) / 2
        hand_extent = np.zeros(2)
        body_min, body_max = np.full(2, np.inf), np.full(2, -np.inf)
        for i, v in enumerate(vertices):
            if not valid_frames[i]:
                continue
            body = (v @ body_axes)[:, :2]
            body_min = np.minimum(body_min, body.min(0))
            body_max = np.maximum(body_max, body.max(0))
            local_hand = ((v[hand] - wrist[i]) @ hand_axes[i])[:, :2]
            hand_extent = np.maximum(
                hand_extent, np.abs(local_hand - hand_center).max(0)
            )
        # Both panels have constant zoom. The wrist stays at the same pixel throughout the close-up.
        body_scale = np.min(np.array([800, 870]) / (body_max - body_min))
        hand_scale = np.min(np.array([810, 815]) / (2 * hand_extent))
        return (
            Camera(
                np.broadcast_to(body_axes, hand_axes.shape),
                np.zeros_like(wrist),
                (body_min + body_max) / 2,
                body_scale,
                (480, 573),
            ),
            Camera(hand_axes, wrist, hand_center, hand_scale, (480, 510)),
        )

    def paint(i, camera, indices, ss, norm, hand_view):
        axes = camera.axes[i]
        pos = (vertices[i] - camera.origins[i]) @ axes
        xy = (
            (pos[:, :2] - camera.target_xy) * np.array([camera.scale, -camera.scale])
            + camera.mid_pixel
        ) * ss
        col = lighting(norm @ axes, pos, indices, hand_view)
        canvas = np.empty((round(1080 * ss), round(960 * ss), 3), np.uint8)
        canvas[:] = [242, 243, 241]
        render(canvas, xy, pos[:, 2], col, indices)
        return Image.fromarray(canvas)

    def composite(i, cams, ss):
        img = Image.new("RGB", (1920, 1080), (242, 243, 241))
        if valid_frames[i]:
            norm = normals(vertices[i])
            for k, (cam, inds) in enumerate(zip(cams, [faces, arm_faces])):
                tile = paint(i, cam, inds, ss, norm, hand_view=(k == 1))
                if tile.size != (960, 1080):
                    tile = tile.resize((960, 1080), Image.Resampling.LANCZOS)
                img.paste(tile, (k * 960, 0))
        ctx = ImageDraw.Draw(img)
        ctx.text((48, 35), title, font=font, fill=(45, 52, 55), anchor="lt")
        return img

    cams = cameras([-0.91 if side == "R" else 0.91, -0.17, -0.38])
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg,
        "-nostdin",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pixel_format",
        "rgb24",
        "-video_size",
        "1920x1080",
        "-framerate",
        "50",
        "-i",
        "pipe:0",
        "-an",
        "-vf",
        "scale=out_color_matrix=bt709:out_range=tv",
        "-color_primaries",
        "bt709",
        "-color_trc",
        "bt709",
        "-colorspace",
        "bt709",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "17",
        "-pix_fmt",
        "yuv420p",
        "-threads",
        "5",
        "-movflags",
        "+faststart",
        "-metadata",
        "title=" + title,
        str(path),
    ]
    with subprocess.Popen(cmd, stdin=subprocess.PIPE) as proc:
        for i in range(frame_count):
            proc.stdin.write(composite(i, cams, 1.5).tobytes())
        proc.stdin.close()
        if proc.wait():
            raise RuntimeError("FFmpeg could not encode the video")
    subprocess.run(
        [
            ffmpeg,
            "-nostdin",
            "-v",
            "error",
            "-xerror",
            "-i",
            str(path),
            "-map",
            "0:v:0",
            "-f",
            "null",
            "-",
        ],
        check=True,
    )


def export_video(model, fit, source, side, index, grasp, take, path):
    prop = {1: " (110 mm)", 2: " (30 mm)"}.get(index, "")
    name = grasp.replace("_", " ")
    title = f"{index:02d} {name}{prop}  |  {'Right' if side == 'R' else 'Left'} Hand  |  Trial {take:02d}"
    data = geometry(model, fit, source, side, take)
    print(f"Rendering {len(data[-1])} frames", flush=True)
    render_sequence(*data, title, side, path)
