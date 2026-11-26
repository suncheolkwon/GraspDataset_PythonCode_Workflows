"""Orthographic line rendering. Drag to orbit and scroll to zoom."""

import math

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QColor, QFont, QPainter, QPen, QPolygonF
from PyQt6.QtWidgets import QWidget


class SkeletonCanvas(QWidget):
    def __init__(self, close_up=False):
        super().__init__()
        self.close_up = close_up
        self.recording = None
        self.frame = 0
        self.drag = None
        self.setMinimumSize(220, 235)
        self.setToolTip("Drag to rotate · Scroll to zoom · Double-click to reset")
        self.setAccessibleName("Measured hand view" if close_up else "Full body view")
        self.reset_view()

    def reset_view(self):
        self.yaw = math.radians(8 if self.close_up else 46)
        self.pitch = math.radians(9 if self.close_up else 13)
        self.zoom = 1.0
        self.update()

    def set_recording(self, recording):
        self.recording, self.frame = recording, 0
        self.reset_view()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        box = QRectF(self.rect()).adjusted(1, 1, -1, -1)
        painter.setPen(QPen(QColor("#d8e1e3"), 1))
        painter.setBrush(QColor("#ffffff"))
        painter.drawRoundedRect(box, 10, 10)
        painter.setPen(QColor("#233a46"))
        font = QFont(self.font())
        font.setBold(True)
        painter.setFont(font)
        title = "Full body"
        if self.close_up:
            title = (
                "Measured hand"
                if not self.recording
                else {"R": "Right hand", "L": "Left hand"}[self.recording.trial.side]
            )
        painter.drawText(QPointF(17, 28), title)
        if self.recording is None:
            return
        r = self.recording
        points = r.points[self.frame]
        if self.close_up:
            center, radius = points[r.wrist] + r.hand_center, r.hand_radius
            basis = r.hand_basis
        else:
            center, radius, basis = r.body_center, r.body_radius, r.body_basis
        local = (points - center) @ basis.T
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        horizontal = local[:, 0] * c - local[:, 1] * s
        depth = local[:, 0] * s + local[:, 1] * c
        vertical = local[:, 2] * math.cos(self.pitch) - depth * math.sin(self.pitch)
        view = box.adjusted(10, 35, -10, -12)
        scale = min(view.width(), view.height()) * 0.465 / radius * self.zoom
        x = view.center().x() + horizontal * scale
        y = view.center().y() - vertical * scale
        painter.setClipRect(view)
        for ids, color, hand in r.lines:
            if self.close_up and not hand:
                continue
            pen = QPen(QColor(color), 3.3 if self.close_up else 2.1)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            painter.setPen(pen)
            painter.drawPolyline(QPolygonF([QPointF(x[i], y[i]) for i in ids]))

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.drag = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseMoveEvent(self, event):
        if self.drag is not None:
            delta = event.position() - self.drag
            self.yaw += delta.x() * 0.009
            self.pitch = max(-1.55, min(1.55, self.pitch - delta.y() * 0.009))
            self.drag = event.position()
            self.update()

    def mouseReleaseEvent(self, event):
        self.drag = None
        self.unsetCursor()

    def wheelEvent(self, event):
        self.zoom = min(
            7.0, max(0.4, self.zoom * 1.12 ** (event.angleDelta().y() / 120))
        )
        self.update()
        event.accept()

    def mouseDoubleClickEvent(self, event):
        self.reset_view()
