"""Evaluate the assembled robot at each original measured time."""

import mujoco
import numpy as np

from .body import basis, angle
from .model import DIGITS
from .hand import A, B


def evaluate(model, qpos, side, wrist, chains):
    prefix = "lh_" if side == "left" else "rh_"
    palm = model.body(prefix + "palm").id
    knuckles = [model.joint(prefix + name + "J4").id for name in ("FF", "MF", "RF", "LF")]
    tips = [model.body(prefix + digit + "distal").id for digit in DIGITS.values()]
    offsets = np.array([0.034996109, 0.032016930, 0.032016930, 0.032016930, 0.032016930])
    data = mujoco.MjData(model)
    source_normal = basis(wrist, chains[:, 1:, 0])[:, :, 2]
    source_distances = np.linalg.norm(chains[:, A, -1] - chains[:, B, -1], axis=2) * 1000
    pair_error = np.empty_like(source_distances)
    normal_error = np.empty(len(qpos))
    depth = np.zeros(len(qpos))
    # The little finger has a mobile metacarpal. Use the neutral MCP landmark
    # frame fixed to the palm when evaluating palm orientation.
    data.qpos[:] = model.qpos0
    data.qpos[7:] = 0
    mujoco.mj_kinematics(model, data)
    local_normal = data.xmat[palm].reshape(3, 3).T @ basis(data.xpos[palm], data.xanchor[knuckles])[:, 2]
    for i, pose in enumerate(qpos):
        data.qpos[:] = pose
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        tip_pos = data.xpos[tips] + data.xmat[tips].reshape(5, 3, 3)[:, :, 2] * offsets[:, None]
        separations = 1000 * np.linalg.norm(tip_pos[A] - tip_pos[B], axis=1)
        pair_error[i] = np.abs(separations - source_distances[i])
        normal_error[i] = angle(data.xmat[palm].reshape(3, 3) @ local_normal, source_normal[i])
        mujoco.mj_collision(model, data)
        if data.ncon:
            depth[i] = max(0.0, -1000 * data.contact.dist[: data.ncon].min())
    return pair_error, normal_error, depth
