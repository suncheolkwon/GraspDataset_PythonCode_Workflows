"""Depth-buffered triangle rasterization for the reference two-panel renderer."""

import math

import numpy as np
from numba import njit


@njit
def _paint(canvas, xy, depths, colors, faces):
    height, width = canvas.shape[:2]
    zbuffer = np.full((height, width), -np.inf, np.float32)
    half, one = np.float32(0.5), np.float32(1.0)
    for i, j, k in faces:
        x0, y0 = xy[i]
        x1, y1 = xy[j]
        x2, y2 = xy[k]
        determinant = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)
        if not np.isfinite(determinant) or abs(determinant) < 1e-7:
            continue
        left = max(0, math.ceil(min(x0, x1, x2) - half))
        right = min(width - 1, math.floor(max(x0, x1, x2) - half))
        top = max(0, math.ceil(min(y0, y1, y2) - half))
        bottom = min(height - 1, math.floor(max(y0, y1, y2) - half))
        inv = one / determinant
        dax, day = (y2 - y0) * inv, -(x2 - x0) * inv
        dbx, dby = -(y1 - y0) * inv, (x1 - x0) * inv
        for y in range(top, bottom + 1):
            yd = np.float32(y) + half - y0
            for x in range(left, right + 1):
                xd = np.float32(x) + half - x0
                a, b = xd * dax + yd * day, xd * dbx + yd * dby
                c = one - a - b
                if min(a, b, c) < -1e-5:
                    continue
                z = c * depths[i] + a * depths[j] + b * depths[k]
                if z <= zbuffer[y, x]:
                    continue
                zbuffer[y, x] = z
                for ch in range(3):
                    value = c * colors[i, ch] + a * colors[j, ch] + b * colors[k, ch]
                    canvas[y, x, ch] = min(255, max(0, value + half))
    return zbuffer


def render(canvas, xy, depth, colors, faces):
    return _paint(
        canvas,
        np.ascontiguousarray(xy, np.float32),
        np.ascontiguousarray(depth, np.float32),
        np.ascontiguousarray(colors, np.float32),
        np.ascontiguousarray(faces, np.int32),
    )
