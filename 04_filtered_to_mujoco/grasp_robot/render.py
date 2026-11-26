"""Fixed-angle whole-body and moving-hand views, with an adjustable display chair."""

import subprocess
import imageio_ffmpeg
import numpy as np
import mujoco
from PIL import Image, ImageDraw, ImageFont
from .model import MODEL

W, H = 960, 1080
BG = (245, 246, 244)


def unit(v):
    return v / np.linalg.norm(v)


def basis(toward, screen_up):
    forward = -unit(toward)
    right = unit(np.cross(forward, screen_up))
    return np.array([right, unit(np.cross(right, forward)), forward])


class Display:
    def __init__(self, q, side):
        prefix = "lh_" if side == "left" else "rh_"
        self.model = m = mujoco.MjModel.from_xml_path(str(MODEL))
        self.data = mujoco.MjData(m)
        self.vis_ids = np.flatnonzero((m.geom_group == 2) & (m.geom_type == mujoco.mjtGeom.mjGEOM_MESH))
        self.body_names = [m.body(int(m.geom_bodyid[g])).name for g in self.vis_ids]
        self.hand = np.array([n.startswith(prefix) for n in self.body_names])
        self.hand_ids = set(self.vis_ids[self.hand])
        self.hand_body = m.body(prefix + "palm").id
        self.boxes = []
        self.verts = []
        for g in self.vis_ids:
            i = m.geom_dataid[g]
            verts = m.mesh_vert[m.mesh_vertadr[i] : m.mesh_vertadr[i] + m.mesh_vertnum[i]]
            lo = verts.min(0)
            hi = verts.max(0)
            self.boxes.append(np.stack(np.meshgrid(*zip(lo, hi)), axis=-1).reshape(-1, 3))
            self.verts.append(verts)
        self.boxes = np.array(self.boxes)
        self.seat_parts = np.array(
            [
                i
                for i, n in enumerate(self.body_names)
                if n == "pelvis"
                or any(s in n for s in ["hip_roll_link", "hip_pitch_link", "hip_yaw_link"])
            ]
        )
        self.feet_parts = np.array(
            [i for i, n in enumerate(self.body_names) if n.endswith("ankle_link")]
        )
        m.vis.global_.offwidth = W
        m.vis.global_.offheight = H
        m.vis.quality.offsamples = 2
        m.vis.headlight.ambient[:] = [0.62, 0.62, 0.62]
        m.vis.headlight.diffuse[:] = [1.00, 1.00, 1.00]
        m.vis.headlight.specular[:] = [0.23, 0.23, 0.23]
        self.opt = mujoco.MjvOption()
        self.opt.geomgroup[:] = 0
        self.opt.geomgroup[2] = 1
        self.opt.sitegroup[:] = 0
        for k in [
            "mjVIS_CONTACTPOINT",
            "mjVIS_CONTACTFORCE",
            "mjVIS_JOINT",
            "mjVIS_TENDON",
            "mjVIS_ACTUATOR",
        ]:
            self.opt.flags[getattr(mujoco.mjtVisFlag, k)] = False
        self.setq(q[0])
        self.palm0 = self.data.xmat[self.hand_body].reshape(3, 3).copy()
        self.prepare(q)
        self.renderer = mujoco.Renderer(m, height=H, width=W, max_geom=512)
        self.cam = mujoco.MjvCamera()

    def setq(self, q):
        self.data.qpos[:] = q
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)

    def vertices(self, i):
        g = self.vis_ids[i]
        return self.verts[i] @ self.data.geom_xmat[g].reshape(3, 3).T + self.data.geom_xpos[g]

    def prepare(self, q):
        hip = []
        front = []
        m = self.model
        for p in q:
            self.setq(p)
            hip.append(self.data.xpos[m.body("pelvis").id].copy())
            knees = np.mean(
                [self.data.xpos[m.body(prefix + "_knee_link").id] for prefix in ("left", "right")],
                axis=0,
            )
            front.append(knees - hip[-1])
        hip = np.array(hip)
        fx = np.mean(front, 0)
        fx[2] = 0
        fx = unit(fx)
        fy = np.cross([0.0, 0.0, 1.0], fx)
        fz = np.array([0.0, 0.0, 1.0])
        self.R = np.column_stack([fx, fy, fz])
        self.chair_center = np.median(hip, axis=0) - 0.025 * fx
        self.bodycam = basis(0.64 * fx - 1.0 * fy + 0.20 * fz, fz)
        mirror = -1 if self.model.body(self.hand_body).name.startswith("lh_") else 1
        self.handcam = basis(
            self.palm0 @ np.array([-0.83 * mirror, -0.54, 0.20]),
            self.palm0 @ np.array([0.15 * mirror, 0, 1.0]),
        )
        floor = []
        seat = []
        torso_back = []
        sample = np.arange(0, len(q), max(1, len(q) // 75))
        for idx in sample:
            self.setq(q[idx])
            floor.append(min(self.vertices(i)[:, 2].min() for i in self.feet_parts))
            pts = np.concatenate([self.vertices(i) for i in self.seat_parts])
            local = (pts - self.chair_center) @ self.R
            near = (np.abs(local[:, 0]) < 0.166) & (np.abs(local[:, 1]) < 0.228)
            seat.append(float(pts[near, 2].min()))
            torso_i = self.body_names.index("torso_link")
            torso_back.append(((self.vertices(torso_i) - self.chair_center) @ fx).min())
        self.floor = float(np.min(floor))
        self.seat = float(np.quantile(seat, 0.08) - 0.002)
        self.seat_bottom_offsets = np.asarray(seat) - self.seat
        self.back = min(-0.204, float(np.min(torso_back) - 0.032))
        self.chair_center[2] = self.seat
        width = 0.234
        self.chair = []

        def add(local_pos, halfsize, rgb):
            self.chair.append(
                (
                    self.chair_center + np.asarray(local_pos) @ self.R.T,
                    np.asarray(halfsize, dtype=float),
                    rgb,
                )
            )

        add([0, 0, -0.012], [0.17, width, 0.012], [0.51, 0.45, 0.37, 1])
        leg_length = self.seat - self.floor - 0.024
        for x in [-0.136, 0.136]:
            for y in [-width + 0.034, width - 0.034]:
                add(
                    [x, y, -0.024 - leg_length / 2],
                    [0.012, 0.012, leg_length / 2],
                    [0.33, 0.35, 0.35, 1],
                )
        for y in [-width + 0.041, width - 0.041]:
            add([self.back, y, 0.15], [0.012, 0.012, 0.165], [0.33, 0.35, 0.35, 1])
            if self.back < -0.17:
                add(
                    [(self.back - 0.14) / 2, y, -0.013],
                    [(-0.14 - self.back) / 2, 0.012, 0.011],
                    [0.33, 0.35, 0.35, 1],
                )
        add([self.back, 0, 0.282], [0.012, width - 0.009, 0.067], [0.51, 0.45, 0.37, 1])
        signs = np.stack(np.meshgrid(*[[-1.0, 1.0]] * 3), axis=-1).reshape(-1, 3)
        chair_box = np.vstack([pos + (half * signs) @ self.R.T for pos, half, _ in self.chair])
        bounds = []
        self.palm_rotation_deg = []
        self.projected_long_axis = []
        for p in q:
            self.setq(p)
            rot = self.data.geom_xmat[self.vis_ids].reshape(-1, 3, 3)
            palm = self.data.xmat[self.hand_body].reshape(3, 3)
            relative_rotation = self.palm0.T @ palm
            self.palm_rotation_deg.append(
                float(np.degrees(np.arccos(np.clip((np.trace(relative_rotation) - 1) / 2, -1, 1))))
            )
            self.projected_long_axis.append((self.handcam @ palm[:, 2])[:2])
            box = self.boxes @ rot.transpose(0, 2, 1) + self.data.geom_xpos[self.vis_ids, None, :]
            world = np.concatenate([box.reshape(-1, 3), chair_box]) @ self.bodycam.T
            hand = (box[self.hand].reshape(-1, 3) - self.data.xpos[self.hand_body]) @ self.handcam.T
            bounds.append([[world.min(0), world.max(0)], [hand.min(0), hand.max(0)]])
        bounds = np.array(bounds)
        longaxis = np.asarray(self.projected_long_axis)
        self.proj_long_span = []
        good = np.linalg.norm(longaxis, axis=1) > 0.18
        if good[0] and good.any():
            angles = np.rad2deg(np.unwrap(np.arctan2(longaxis[good, 1], longaxis[good, 0])))
            self.proj_long_span = [float(angles.min() - angles[0]), float(angles.max() - angles[0])]
        self.view = []
        for i, b in enumerate([self.bodycam, self.handcam]):
            lo = bounds[:, i, 0].min(0)
            hi = bounds[:, i, 1].max(0)
            boxaxis = hi - lo
            span = max(boxaxis[1], boxaxis[0] * H / W) * [1.23, 1.17][i]
            center = (hi + lo) / 2
            if i == 0:
                center[1] += 0.025 * span
            self.view.append((b, center @ b, float(span)))

    def image(self, view):
        r = self.renderer
        sc = r.scene
        m = self.model
        mujoco.mjv_updateScene(m, self.data, self.opt, None, self.cam, mujoco.mjtCatBit.mjCAT_ALL, sc)
        for g in sc.geoms[: sc.ngeom]:
            if g.objtype != mujoco.mjtObj.mjOBJ_GEOM or (view and g.objid not in self.hand_ids):
                g.rgba[3] = 0
            elif max(g.rgba[:3]) < 0.32:
                # Lift dark finishes for visual inspection of the original mesh profiles.
                g.rgba[:3] = [0.28, 0.33, 0.37] if g.objid in self.hand_ids else [0.29, 0.33, 0.36]
                g.specular = 0.30
                g.shininess = 0.28
        if not view:
            for pos, half, color in self.chair:
                g = sc.geoms[sc.ngeom]
                sc.ngeom += 1
                mujoco.mjv_initGeom(
                    g,
                    mujoco.mjtGeom.mjGEOM_BOX,
                    half,
                    pos,
                    self.R.reshape(-1),
                    np.asarray(color, np.float32),
                )
        b, center, span = self.view[view]
        if view:
            center = center + self.data.xpos[self.hand_body]
        for camera in sc.camera:
            camera.orthographic = 1
            camera.pos[:] = center - b[2] * 5
            camera.forward[:] = b[2]
            camera.up[:] = b[1]
            camera.frustum_near = 0.01
            camera.frustum_far = 12
            camera.frustum_center = 0
            camera.frustum_top = span * 0.5
            camera.frustum_bottom = -span * 0.5
            camera.frustum_width = 0
        if sc.nlight:
            sc.lights[0].pos[:] = center - b[2] * 3 + b[1] * 1.3 - b[0] * 1.2
            sc.lights[0].dir[:] = unit(center - sc.lights[0].pos)
        sc.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
        rgb = r.render()
        rgb[~rgb.any(axis=2)] = BG
        return rgb


def write_video(output, poses, title, side):
    display = Display(poses, side)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 25)
    except OSError:
        font = ImageFont.load_default(size=25)
    executable = imageio_ffmpeg.get_ffmpeg_exe()
    command = [
        executable,
        "-nostdin",
        "-y",
        "-v",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        "1920x1080",
        "-r",
        "50",
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-preset",
        "fast",
        "-threads",
        "4",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output),
    ]
    try:
        with subprocess.Popen(command, stdin=subprocess.PIPE) as encoder:
            try:
                for pose in poses:
                    display.setq(pose)
                    frame = Image.fromarray(np.hstack((display.image(0), display.image(1))))
                    ImageDraw.Draw(frame).text((31, 26), title, font=font, fill=(30, 38, 42))
                    encoder.stdin.write(frame.tobytes())
            finally:
                encoder.stdin.close()
            if encoder.wait():
                raise RuntimeError("FFmpeg failed to encode the video")
    finally:
        display.renderer.close()
