"""Play a released GRASP filtered recording as connected 3D lines."""

import argparse
from pathlib import Path
import sys
import time

import numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from viewer.canvas import SkeletonCanvas
from viewer.motion import DIGITS, DIGIT_COLORS, discover, load


class Viewer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GRASP | Skeleton viewer")
        self.resize(1120, 720)
        self.trials, self.recording = [], None
        self.frame = 0
        self.selecting = False
        self.clock = 0.0
        self.t0 = 0.0
        self.timer = QTimer(self)
        self.timer.setTimerType(Qt.TimerType.PreciseTimer)
        self.timer.setInterval(16)
        self.timer.timeout.connect(self.tick)
        self.setStyleSheet("""
        QWidget {background:#f4f7f8; color:#223b47; font-size:14px;}
        QComboBox,QPushButton {background:white; border:1px solid #b6c7cd; border-radius:5px; padding:6px 10px;}
        QPushButton:hover {background:#e4f1f1;} QPushButton:disabled,QLabel:disabled {color:#86959b;}
        QPushButton#play {background:#126a70; color:white; border:0; min-width:70px;}
        QSlider::groove:horizontal {height:5px; background:#cbdadd; border-radius:2px;}
        QSlider::sub-page:horizontal {background:#13777e;}
        QSlider::handle:horizontal {width:13px; margin:-5px 0; background:#126a70; border-radius:6px;}
        """)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(22, 18, 22, 17)
        outer.setSpacing(12)
        top = QHBoxLayout()
        title = QLabel("GRASP  /  Skeleton viewer")
        title.setStyleSheet("font-size:22px; font-weight:600; color:#203b47;")
        top.addWidget(title)
        top.addStretch()
        choose = QPushButton("Open dataset…")
        choose.clicked.connect(self.choose_folder)
        top.addWidget(choose)
        outer.addLayout(top)
        self.location = QLabel(
            "Open the extracted filtered dataset to choose a recording."
        )
        self.location.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        outer.addWidget(self.location)
        selectors = QHBoxLayout()
        self.grasp, self.side, self.take = QComboBox(), QComboBox(), QComboBox()
        for label, widget, stretch in (
            ("Grasp", self.grasp, 3),
            ("Hand", self.side, 1),
            ("Recording", self.take, 2),
        ):
            name = QLabel(label)
            name.setBuddy(widget)
            selectors.addWidget(name)
            selectors.addWidget(widget, stretch)
        outer.addLayout(selectors)
        self.grasp.currentIndexChanged.connect(self.change_grasp)
        self.side.currentIndexChanged.connect(self.change_side)
        self.take.currentIndexChanged.connect(self.change_trial)
        views = QHBoxLayout()
        views.setSpacing(12)
        self.body, self.hand = SkeletonCanvas(), SkeletonCanvas(True)
        views.addWidget(self.body, 1)
        views.addWidget(self.hand, 1)
        outer.addLayout(views, 1)
        legend = QHBoxLayout()
        legend.addStretch()
        for digit, color in zip(DIGITS, DIGIT_COLORS):
            line = QFrame()
            line.setFixedSize(19, 3)
            line.setStyleSheet(f"background:{color}")
            legend.addWidget(line)
            legend.addWidget(QLabel("Little" if digit == "Pinky" else digit))
            legend.addSpacing(6)
        legend.addStretch()
        outer.addLayout(legend)
        self.position = QSlider(Qt.Orientation.Horizontal)
        self.position.setAccessibleName("Recording timeline")
        self.position.valueChanged.connect(self.seek)
        outer.addWidget(self.position)
        controls = QHBoxLayout()
        self.play = QPushButton("Play")
        self.play.setObjectName("play")
        self.play.clicked.connect(self.toggle)
        self.loop = QCheckBox("Loop")
        self.speed = QComboBox()
        for speed in (0.25, 0.5, 1.0, 2.0):
            self.speed.addItem(f"{speed:g}×", speed)
        self.speed.setCurrentIndex(2)
        self.speed.setAccessibleName("Playback speed")
        self.speed.currentIndexChanged.connect(
            lambda: self.anchor_clock() if self.timer.isActive() else None
        )
        self.info = QLabel("—")
        self.info.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        controls.addWidget(self.play)
        controls.addWidget(self.speed)
        controls.addWidget(self.loop)
        controls.addStretch()
        controls.addWidget(self.info)
        outer.addLayout(controls)
        self.shortcuts = []
        for key, action in (
            ("Space", self.toggle),
            ("Left", lambda: self.step(-1)),
            ("Right", lambda: self.step(1)),
        ):
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.activated.connect(action)
            self.shortcuts.append(shortcut)
        self.set_ready(False)

    def set_ready(self, ready):
        for widget in (self.position, self.play, self.speed, self.loop):
            widget.setEnabled(ready)

    def choose_folder(self):
        path = QFileDialog.getExistingDirectory(
            self, "Select extracted filtered dataset"
        )
        if path:
            self.open_dataset(Path(path))

    @staticmethod
    def fill(combo, entries, prefer=None):
        combo.clear()
        for label, val in entries:
            combo.addItem(label, val)
        idx = combo.findData(prefer)
        combo.setCurrentIndex(max(idx, 0))

    def open_dataset(self, folder, initial=None):
        self.pause()
        try:
            trials = discover(folder)
            if initial and all(t.name != initial for t in trials):
                raise ValueError(f"Trial {initial} was not found in this dataset.")
        except (OSError, ValueError) as exc:
            self.location.setText(str(exc))
            return False
        self.trials = trials
        grps = sorted({(t.number, t.grasp) for t in trials})
        chosen = next((t for t in trials if t.name == initial), None)
        self.selecting = True
        self.fill(
            self.grasp,
            [(f"{n:02d}   {name.replace('_', ' ')}", name) for n, name in grps],
            chosen.grasp if chosen else grps[0][1],
        )
        self.selecting = False
        self.dataset_label = f"{len(trials):,} recordings   ·   {folder.name}"
        self.location.setToolTip(str(folder.resolve()))
        self.change_grasp(
            preferred_side=chosen.side if chosen else "R",
            preferred_take=chosen.take if chosen else 1,
        )
        return self.recording is not None

    def change_grasp(self, index=0, preferred_side=None, preferred_take=None):
        if self.selecting:
            return
        grasp = self.grasp.currentData()
        sides = sorted({t.side for t in self.trials if t.grasp == grasp})
        self.selecting = True
        self.fill(
            self.side,
            [({"R": "Right", "L": "Left"}[s], s) for s in sides],
            preferred_side or self.side.currentData(),
        )
        self.selecting = False
        self.change_side(preferred_take=preferred_take)

    def change_side(self, index=0, preferred_take=None):
        if self.selecting:
            return
        choices = [
            t.take
            for t in self.trials
            if t.grasp == self.grasp.currentData() and t.side == self.side.currentData()
        ]
        self.selecting = True
        self.fill(
            self.take,
            [
                (
                    f"{n:02d}   ·   S{(n - 1) // 10 + 1:02d} / rep {(n - 1) % 10 + 1:02d}",
                    n,
                )
                for n in choices
            ],
            preferred_take or self.take.currentData(),
        )
        self.selecting = False
        self.change_trial()

    def change_trial(self):
        if self.selecting:
            return
        self.pause()
        selected = (
            self.grasp.currentData(),
            self.side.currentData(),
            self.take.currentData(),
        )
        trial = next(
            (t for t in self.trials if (t.grasp, t.side, t.take) == selected), None
        )
        try:
            self.recording = load(trial) if trial else None
        except (OSError, ValueError, StopIteration) as exc:
            self.recording = None
            self.location.setText(f"Could not load recording: {exc}")
        for canvas in (self.body, self.hand):
            canvas.set_recording(self.recording)
        self.set_ready(self.recording is not None)
        if self.recording:
            self.position.setRange(0, len(self.recording.times) - 1)
            self.location.setText(self.dataset_label)
            self.show_frame(0)
        else:
            self.info.setText("—")

    def show_frame(self, index):
        self.frame = int(index)
        r = self.recording
        self.position.blockSignals(True)
        self.position.setValue(self.frame)
        self.position.blockSignals(False)
        for canvas in (self.body, self.hand):
            canvas.frame = self.frame
            canvas.update()
        self.info.setText(
            f"Source {r.times[self.frame]:.3f} s   /   {r.times[-1]:.3f} s"
            f"     ·     Source row {r.source_rows[self.frame]}"
        )

    def anchor_clock(self):
        self.clock = time.monotonic()
        self.t0 = self.recording.times[self.frame]

    def pause(self):
        self.timer.stop()
        self.play.setText("Play")

    def toggle(self):
        if self.recording is None:
            return
        if self.timer.isActive():
            self.pause()
        else:
            if self.frame == len(self.recording.times) - 1:
                self.show_frame(0)
            self.anchor_clock()
            self.timer.start()
            self.play.setText("Pause")

    def tick(self):
        times = self.recording.times
        t = self.t0 + (time.monotonic() - self.clock) * self.speed.currentData()
        if t >= times[-1]:
            if self.loop.isChecked() and times[-1] > times[0]:
                t = times[0] + (t - times[0]) % (times[-1] - times[0])
                self.t0, self.clock = t, time.monotonic()
            else:
                self.show_frame(len(times) - 1)
                self.pause()
                return
        self.show_frame(max(0, np.searchsorted(times, t, side="right") - 1))

    def seek(self, index):
        if self.recording is not None:
            self.show_frame(index)
            if self.timer.isActive():
                self.anchor_clock()

    def step(self, delta):
        if self.recording is not None:
            self.pause()
            self.show_frame(
                max(0, min(len(self.recording.times) - 1, self.frame + delta))
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        "--data-root",
        dest="input_dir",
        metavar="FILTERED",
        type=Path,
        help="Extracted filtered directory or its parent; optional.",
    )
    parser.add_argument(
        "--trial", help="Initial recording, e.g. Tripod_R_1 (requires --input-dir)."
    )
    args = parser.parse_args()
    if args.trial and not args.input_dir:
        parser.error("--trial requires --input-dir")
    app = QApplication([sys.argv[0]])
    app.setStyle("Fusion")
    window = Viewer()
    if args.input_dir and not window.open_dataset(args.input_dir, args.trial):
        print(window.location.text(), file=sys.stderr)
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
