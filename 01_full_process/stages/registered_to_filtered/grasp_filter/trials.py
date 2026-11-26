"""Release metadata, sample correspondence, and CSV I/O."""
import csv
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from .filter import filter_trial


@dataclass(frozen=True)
class Trial:
    name: str
    side: str
    relative: Path
    rows: int
    first: int


def catalog():
    with Path(__file__).with_name('trials.csv').open(newline='', encoding='utf-8') as f:
        trials = {}
        for rec in csv.DictReader(f):
            name = rec['trial']
            folder = rec['grasp_number'] + '.' + rec['grasp']
            leaf = rec['grasp'] + '_' + rec['hand']
            relative = Path(folder) / leaf / (name + '_registered.csv')
            trials[name] = Trial(name, rec['hand'], relative, int(rec['registered_rows']), int(rec['first_source_row']))
    return trials


@lru_cache(maxsize=2)
def skeleton(side):
    with Path(__file__).with_name('skeleton.csv').open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    names = [r['node'] for r in rows]
    key = {'L': 'left_parent', 'R': 'right_parent'}[side]
    parents = np.array([names.index(r[key]) if r[key] else -1 for r in rows])
    components = ['Pos' + a for a in 'XYZ'] + ['Rot' + a for a in 'XYZW']
    columns = ['Frame', 'Time'] + [node + '_' + field for node in names for field in components]
    indices = np.arange(2, len(columns)).reshape(len(names), 7)
    return names, parents, columns, indices[:, :3], indices[:, 3:]


def convert(path, trial):
    names, parents, columns, pc, qc = skeleton(trial.side)
    # The release pipeline used pandas' default float parser for its inputs.
    registered = pd.read_csv(path, dtype=np.float64, encoding='utf-8-sig')
    if list(registered.columns) != columns or len(registered) != trial.rows:
        raise ValueError(f'{path}: expected {trial.rows} rows and the 534 release columns in order')
    indices = np.arange(trial.rows)
    if not np.array_equal(registered.Frame, indices) or not np.array_equal(registered.Time, indices / 200):
        raise ValueError(f'{path}: incomplete or changed registration clock')
    source_rows = indices[trial.first:]
    source = registered.to_numpy()[source_rows].copy()
    if not np.isfinite(source).all():
        raise ValueError(f'{path}: non-finite retained measurement')
    times = source[:, 1].copy()
    data = dict(source=source, source_rows=source_rows, source_times=times, names=names, parents=parents,
                pc=pc, qc=qc, segments=[(0, len(source))])
    output = filter_trial(data)
    frame = pd.DataFrame(output, columns=columns)
    rowmap = pd.DataFrame(dict(output_row0=np.arange(len(frame)), source_row0=source_rows,
                               source_time_s=times, Frame=output[:, 0], Time=output[:, 1]))
    return frame, rowmap


def compare(table, path, *, rowmap=False):
    reference = pd.read_csv(path, dtype=np.float64, float_precision='round_trip')
    if not reference.columns.equals(table.columns) or reference.shape != table.shape:
        raise ValueError(f'Schema or row count differs in {path}')
    got = table.to_numpy()
    expected = reference.to_numpy()
    if not np.isfinite(expected).all():
        raise ValueError(f'Non-finite reference in {path}')
    if rowmap:
        if not np.array_equal(got, expected):
            raise ValueError(f'Different source-row correspondence in {path}')
        return 0., 0.
    error = np.abs(got - expected)
    nodes = error[:, 2:].reshape(len(error), -1, 7)
    position = float(nodes[:, :, :3].max())
    rotation = float(nodes[:, :, 3:].max())
    if error[:, :2].any() or position > 1e-8 or rotation > 1e-9:
        raise ValueError(f'{path}: clock, position (max {position:.9g} mm) or signed quaternion '
                         f'(max {rotation:.9g}) differs')
    return position, rotation
