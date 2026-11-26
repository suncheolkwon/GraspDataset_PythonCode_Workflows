"""Create the GRASP filtered tier from the distributed registered recordings."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

from grasp_filter.trials import catalog, compare, convert


def save_pair(frame, rowmap, destination):
    """Stage both outputs, publish the sidecar first, and publish the motion last.

    A retry verifies any existing member before completing an interrupted pair.
    No existing file is replaced, including during concurrent runs.
    """
    sidecar = destination.with_suffix('.rowmap.csv')
    tables = ((sidecar, rowmap, True), (destination, frame, False))
    destination.parent.mkdir(parents=True, exist_ok=True)
    for path, table, is_map in tables:
        if path.exists():
            compare(table, path, rowmap=is_map)
    if destination.exists() and sidecar.exists():
        return 'existing_verified'
    with TemporaryDirectory(dir=destination.parent, prefix='.filter-') as tmp:
        for path, table, is_map in tables:
            if not path.exists():
                table.to_csv(Path(tmp) / path.name, index=False, float_format='%.17g')
        for path, table, is_map in tables:
            stage = Path(tmp) / path.name
            if not stage.exists():
                continue
            try:
                os.link(stage, path)
            except FileExistsError:
                compare(table, path, rowmap=is_map)
    return 'written'


def run(job):
    trial, source, destination, verify = job
    try:
        frame, rowmap = convert(source, trial)
        if verify:
            compare(rowmap, destination.with_suffix('.rowmap.csv'), rowmap=True)
            pos, rot = compare(frame, destination)
            return 'verified', len(frame), pos, rot, None
        return save_pair(frame, rowmap, destination), len(frame), 0., 0., None
    except Exception as exc:
        return 'failed', 0, 0., 0., f'{trial.name}: {exc}'


def overlaps(first, second):
    return first.is_relative_to(second) or second.is_relative_to(first)


def discover(source, records, requested, require_complete):
    """Use only exact GRASP trial names; a repeated recording is ambiguous."""
    if not source.is_dir():
        raise ValueError(f'Registered input directory does not exist: {source}')
    if any(p.name.startswith('Registration_data_GRASP_filtered') for p in (source, *source.parents)):
        raise ValueError(f'This is the filtered tier. Choose the registered tier: {source}')
    selected = set(requested or ())
    if len(selected) != len(requested or ()):
        raise ValueError('A --trial name was supplied more than once.')
    unknown = selected - records.keys()
    if unknown:
        raise ValueError('Unknown trial ID: ' + ', '.join(sorted(unknown)))
    if require_complete and requested:
        raise ValueError('--require-complete checks all 1,320 recordings; omit --trial.')

    found = {}
    for path in source.rglob('*_registered.csv'):
        trial_name = path.name.removesuffix('_registered.csv')
        if selected and trial_name not in selected:
            continue
        if trial_name not in records:
            raise ValueError(f'Unknown registered recording: {path}')
        if path.with_suffix('.rowmap.csv').exists() or path.resolve().with_suffix('.rowmap.csv').exists():
            raise ValueError(f'This recording has a filtered row map. Select its registered input: {path}')
        if not path.is_file():
            raise ValueError(f'Input is not a regular file: {path}')
        if trial_name in found:
            raise ValueError(f'Ambiguous recording {trial_name}: {found[trial_name]} and {path}')
        found[trial_name] = path
    expected = records.keys() if require_complete else selected
    missing = sorted(set(expected) - found.keys())
    if missing:
        raise ValueError(f'Missing {len(missing)} requested recording(s); first: {missing[0]}')
    if not found:
        raise ValueError(f'No matching *_registered.csv files under {source}')
    return [(trial, found[trial.name]) for trial in records.values() if trial.name in found]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--data-root', type=Path, help='Directory containing Registration_data_GRASP.')
    inputs.add_argument('--input-dir', '--registered-dir', type=Path, dest='registered_dir',
                        help='Folder containing registered recordings, including a subset or one grasp.')
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--output-dir', '--output', dest='output', type=Path, help='Destination filtered tier directory.')
    mode.add_argument('--verify-against', type=Path, help='Compare with an extracted filtered tier; no files written.')
    parser.add_argument('--trial', action='append', metavar='NAME', help='Exact trial ID. Repeatable; default is every recording found in the input.')
    parser.add_argument('--require-complete', action='store_true', help='Require all 1,320 release recordings.')
    parser.add_argument('--workers', default=1, type=int, help='Number of recordings to process in parallel.')
    args = parser.parse_args()
    source = (args.registered_dir or (args.data_root / 'Registration_data_GRASP')).expanduser().resolve()
    output = (args.output or args.verify_against).expanduser().resolve()
    verify = args.verify_against is not None
    if args.workers < 1:
        parser.error('--workers must be positive.')
    if not verify:
        tier = next((p for p in (source, *source.parents) if p.name == 'Registration_data_GRASP'), source)
        protected = (tier, tier.parent / 'Manus_data_GRASP', tier.parent / 'Motive_data_GRASP')
        if any(overlaps(output, p.resolve()) for p in protected):
            parser.error('Choose an output folder separate from the input and the raw data tiers.')
        if output.exists() and not output.is_dir():
            parser.error(f'The output directory is a file: {output}')
    try:
        selected = discover(source, catalog(), args.trial, args.require_complete)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    jobs = []
    for trial, path in selected:
        target = output / trial.relative
        if verify and (not target.is_file() or not target.with_suffix('.rowmap.csv').is_file()):
            parser.error(f'Missing filtered reference or row map: {target}')
        if not verify and (target != target.resolve() or target.with_suffix('.rowmap.csv').is_symlink()):
            parser.error(f'Target must not be a symbolic link: {target}')
        jobs.append((trial, path, target, verify))
    counts = Counter()
    total = 0
    pos = rot = 0.
    workers = min(args.workers, len(jobs))
    pool = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
    try:
        results = pool.map(run, jobs) if pool else map(run, jobs)
        for status, rows, p, q, error in results:
            counts[status] += 1
            total += rows
            pos, rot = max(p, pos), max(q, rot)
            if error:
                print(error, file=sys.stderr)
    finally:
        if pool:
            pool.shutdown()
    summary = {'selected':len(jobs), 'failed':counts['failed'], 'output_rows':total, **counts}
    if verify:
        summary.update(position_component_max_mm=pos, quaternion_component_max_abs=rot)
    print(json.dumps(summary))
    return int(counts['failed'] != 0)


if __name__ == '__main__':
    sys.exit(main())
