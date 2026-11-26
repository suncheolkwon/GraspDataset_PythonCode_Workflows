"""Event-preserving, bounded distal-hand smoothing with exact fixed lengths.

Body/wrist and palm anchors use the registered length-only reconstruction.
Only distal bone directions, expressed in the shared wrist frame, are smoothed
in source-only low-motion (quiet) neighborhoods. Orientations transport the
baseline frame by the minimal swing from old to new outgoing bone, preserving
terminal relative rotation. The full centred window must be quiet at every sample;
moving samples, quiet boundaries, and protected events retain the length-only
baseline. This is an offline centred method, not a claim of nonlinear
zero-phase filtering or proof that every quiet variation is sensor noise.
Each source-quiet contiguous run uses one fixed-grid gain, selected to keep
both TIP second-difference RMS and maximum at or below its length-only baseline.
"""
from __future__ import annotations

import numpy as np
from . import kinematics

METHOD_VERSION = 'FILTERED_V3_EVENT_PRESERVING_DISTAL_V4'
DIGITS = ('Thumb', 'Index', 'Middle', 'Ring', 'Pinky')
QUIET_GAIN_GRID = [1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0]
QUIET_CONSTRAINT_TOLERANCE_MM = 1e-8
DEFAULT_CONFIG = {
    'joint_step_mm': 3.0,
    'tip_gap_step_mm': 3.0,
    'relative_rotation_step_deg': 3.0,
    'guard_samples': 2,
    'quiet_joint_step_mm': 0.5,
    'quiet_relative_rotation_step_deg': 1.0,
    'quiet_radius_edges': 10,
    'kernel': [1.0, 2.0, 3.0, 2.0, 1.0],
    'hand_correction_cap_mm': 0.25,
    'cap_bisection_iterations': 20,
}


def _unit(x, context):
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    if not np.isfinite(x).all() or np.any(n <= kinematics.EPS):
        raise ValueError('DEGENERATE_EVENT_DIRECTION: ' + context)
    return x / n


def _hand_indices(data):
    names = list(data['names'])
    wrist = names.index('Manus_Hand')
    nodes = np.array([i for i, name in enumerate(names)
                      if name.startswith('Manus_') and name != 'Manus_Hand'], dtype=int)
    if len(nodes) != 24:
        raise ValueError('EXPECTED_24_MANUS_CHILD_NODES')
    tips = np.array([names.index('Manus_' + digit + '_TIP') for digit in DIGITS])
    return wrist, nodes, tips


def _quiet_samples(data):
    """Mark low-motion samples outside protected events; edge k connects k-1 to k.

    Source coordinates and normalized source rotations are used, before any
    length projection. Only contiguous source rows separated by 0.005 seconds
    are valid edges. Protection does not cross a source-segment boundary.
    """
    cfg = DEFAULT_CONFIG
    source = np.asarray(data['source'], dtype=float)
    pc, qc = np.asarray(data['pc']), np.asarray(data['qc'])
    parents = np.asarray(data['parents'], dtype=int)
    source_rows = np.asarray(data['source_rows'], dtype=int)
    source_times = np.asarray(data['source_times'], dtype=float)
    segments = [(int(a), int(b)) for a, b in data['segments']]
    xyz = source[:, pc]
    q = kinematics.qnorm(source[:, qc], 'EVENT_SOURCE_QUATERNION')
    wrist, hand, tips = _hand_indices(data)
    local = kinematics.qapply(kinematics.qconj(q[:, wrist, None]), xyz[:, hand] - xyz[:, wrist, None])
    relative = kinematics.qmul(kinematics.qconj(q[:, parents[hand]]), q[:, hand])
    relative = kinematics.qnorm(relative, 'EVENT_SOURCE_RELATIVE_QUATERNION')
    gaps = np.stack([np.linalg.norm(xyz[:, tips[i]] - xyz[:, tips[j]], axis=-1)
                     for i in range(5) for j in range(i + 1, 5)], axis=-1)
    n = len(source)
    valid = np.zeros(n, dtype=bool)
    joint_step = np.zeros(n)
    gap_step = np.zeros(n)
    rotation_step = np.zeros(n)
    for a, b in segments:
        if b - a < 2:
            continue
        edge = np.arange(a + 1, b)
        good = ((source_rows[edge] - source_rows[edge - 1] == 1)
                & np.isclose(source_times[edge] - source_times[edge - 1], .005,
                             rtol=0, atol=1e-9))
        if not good.all():
            raise ValueError('SEGMENT_CONTAINS_NONCONTIGUOUS_SOURCE_EDGE')
        valid[edge] = True
        joint_step[edge] = np.max(np.linalg.norm(local[edge] - local[edge - 1], axis=-1), axis=-1)
        gap_step[edge] = np.max(np.abs(gaps[edge] - gaps[edge - 1]), axis=-1)
        delta = kinematics.qmul(kinematics.qconj(relative[edge - 1]), relative[edge])
        angles = np.degrees(2 * np.arctan2(np.linalg.norm(delta[..., :3], axis=-1),
                                         np.abs(delta[..., 3])))
        rotation_step[edge] = angles.max(axis=-1)
    event = valid & ((joint_step > cfg['joint_step_mm'])
                     | (gap_step > cfg['tip_gap_step_mm'])
                     | (rotation_step > cfg['relative_rotation_step_deg']))
    protected = np.zeros(n, dtype=bool)
    quiet = np.zeros(n, dtype=bool)
    radius, guard = cfg['quiet_radius_edges'], cfg['guard_samples']
    low = valid & (joint_step < cfg['quiet_joint_step_mm']) & (rotation_step < cfg['quiet_relative_rotation_step_deg'])
    for a, b in segments:
        for k in np.flatnonzero(event[a:b]) + a:
            protected[max(a, k - 1 - guard):min(b, k + 1 + guard)] = True
        # Complete 2*r+1 edge window; segment first valid edge is a+1.
        for k in range(a + 1 + radius, b - radius):
            quiet[k] = bool(low[k-radius:k+radius+1].all())
    return quiet & ~protected


def _centred_directions(direction, eligible, segments, kernel):
    weights = np.asarray(kernel, dtype=float)
    weights /= weights.sum()
    radius = len(weights) // 2
    smoothd = direction.copy()
    for a, b in segments:
        centers = np.flatnonzero(eligible[a:b]) + a
        if not len(centers):
            continue
        dsum = np.zeros_like(direction[centers])
        for offset, weight in zip(range(-radius, radius+1), weights):
            dsum += weight * direction[centers+offset]
        smoothd[centers] = _unit(dsum, 'CENTRED_HAND_DIRECTION')
    return smoothd


def _minimal_swing(source_direction, target_direction):
    """Unique shortest swing except antiparallel, which explicitly fails.

    A cap-constrained quiet correction should never approach the ambiguous
    antiparallel case; no arbitrary axis is silently inserted if it does.
    """
    a = _unit(source_direction, 'SWING_SOURCE')
    b = _unit(target_direction, 'SWING_TARGET')
    dot = np.clip(np.sum(a*b, axis=-1), -1., 1.)
    if np.any(dot < -1. + 1e-10):
        raise ValueError('AMBIGUOUS_ANTIPARALLEL_ORIENTATION_TRANSPORT')
    return kinematics.qnorm(np.concatenate((np.cross(a, b), (1.+dot)[..., None]), axis=-1),
                    'MINIMAL_OUTGOING_BONE_SWING')


def filter_trial(data):
    """Produce filtered poses on the retained sample grid."""
    cfg = DEFAULT_CONFIG
    baseline, diagnostics = kinematics.length_baseline(data)
    quiet = _quiet_samples(data)
    pc, qc = np.asarray(data['pc']), np.asarray(data['qc'])
    names = list(data['names'])
    parents = np.asarray(data['parents'], dtype=int)
    segments = [(int(a), int(b)) for a, b in data['segments']]
    bp, bq = baseline[:, pc].copy(), baseline[:, qc].copy()
    n = len(baseline)
    wrist, hand, tips = _hand_indices(data)
    eligible = np.zeros(n, dtype=bool)
    radius = len(cfg['kernel']) // 2
    for a, b in segments:
        for k in range(a + radius, b - radius):
            # Preserve genuine moving/held trajectories below event thresholds
            # as well: every member of the full window must be source-quiet.
            eligible[k] = quiet[k-radius:k+radius+1].all()

    # Palm positions and CMC orientations stay exactly at their baseline.
    # Use one wrist frame for all directions: independently filtering rotations
    # and their compensating parent-local offsets can create new physical jitter.
    actived = np.array([j for j in hand if not names[j].endswith(('_CMC', '_MCP'))], dtype=int)
    vectors = bp[:, actived] - bp[:, parents[actived]]
    wrist_direction = _unit(kinematics.qapply(kinematics.qconj(bq[:, wrist, None]), vectors),
                            'BASELINE_WRIST_FRAME_BONE_DIRECTION')
    sd = _centred_directions(wrist_direction, eligible, segments, cfg['kernel'])
    dindex = {int(j): i for i, j in enumerate(actived)}
    ref = {x['child_index']: x['reference_length_mm'] for x in diagnostics['links']}
    hand_order = [j for j in kinematics.topological_order(parents) if j in set(hand)]
    tip_relative = kinematics.qnorm(kinematics.qmul(kinematics.qconj(bq[:, parents[tips]]), bq[:, tips]),
                            'BASELINE_TERMINAL_RELATIVE_QUATERNION')

    def reconstruct(alpha):
        # One alpha per complete hand, shared by every distal bone direction.
        aq = alpha[:, None, None]
        blendd = _unit(wrist_direction + aq * (sd-wrist_direction), 'BLENDED_WRIST_FRAME_BONE_DIRECTION')
        worldd = kinematics.qapply(bq[:, wrist, None], blendd)
        xyz, q = bp.copy(), bq.copy()
        for child in hand_order:
            parent = parents[child]
            if child in dindex:
                xyz[:, child] = xyz[:, parent] + ref[child] * worldd[:, dindex[child]]
        # Each nonterminal digit node has exactly one outgoing distal bone.
        # This rotation transports its measured source frame without inventing
        # twist. Terminal nodes inherit the parent's swing, preserving relative q.
        swing = _minimal_swing(vectors, worldd)
        q[:, parents[actived]] = kinematics.qnorm(kinematics.qmul(swing, bq[:, parents[actived]]),
                                         'TRANSPORTED_HAND_QUATERNION')
        q[:, tips] = kinematics.qnorm(kinematics.qmul(q[:, parents[tips]], tip_relative),
                              'TRANSPORTED_TERMINAL_QUATERNION')
        # Enforce exact baseline identity for protected/noneligible/zero-alpha rows,
        # including quaternion signs, rather than merely floating-point closeness.
        inactive = alpha == 0
        xyz[inactive], q[inactive] = bp[inactive], bq[inactive]
        correction = np.linalg.norm(xyz[:, hand] - bp[:, hand], axis=-1).max(axis=-1)
        return xyz, q, correction

    alpha = eligible.astype(float)
    xyz, q, raw_correction = reconstruct(alpha)
    cap = cfg['hand_correction_cap_mm']
    capped = raw_correction > cap
    if capped.any():
        low = np.zeros(n)
        high = np.ones(n)
        for _ in range(cfg['cap_bisection_iterations']):
            trial_alpha = alpha.copy()
            trial_alpha[capped] = (low[capped]+high[capped])/2
            _, _, correction = reconstruct(trial_alpha)
            acceptable = correction <= cap - 1e-10
            low[capped & acceptable] = trial_alpha[capped & acceptable]
            high[capped & ~acceptable] = trial_alpha[capped & ~acceptable]
        alpha[capped] = low[capped]
        xyz, q, correction = reconstruct(alpha)
    else:
        correction = raw_correction
    if correction.max(initial=0) > cap + 1e-8:
        raise ValueError('HAND_CORRECTION_CAP_VIOLATION')

    # A cap alone does not rule out new jitter: even a small position correction
    # can raise a second-difference peak. Bound RMS AND peak for each complete
    # source-defined quiet run, keeping the original evaluation mask unchanged.
    # First/last kernel-radius rows of a run stay baseline, so runs are independent.
    cap_alpha = alpha.copy()
    quiet_runs = []
    baseline_tip_local = kinematics.qapply(kinematics.qconj(bq[:, wrist, None]), bp[:, tips]-bp[:, wrist, None])
    def _run_stats(tip_local, a, b):
        if b-a < 3:
            return 0.0, 0.0
        second = tip_local[a+2:b]-2*tip_local[a+1:b-1]+tip_local[a:b-2]
        values = np.linalg.norm(second, axis=-1)
        return float(np.sqrt(np.mean(values**2))), float(values.max(initial=0))
    for a, b in segments:
        edges = np.diff(np.r_[False, quiet[a:b], False].astype(int))
        starts = np.flatnonzero(edges == 1)+a
        stops = np.flatnonzero(edges == -1)+a
        for first, stop in zip(starts, stops):
            rms, maximum = _run_stats(baseline_tip_local, first, stop)
            quiet_runs.append(dict(first_output_row=int(first), last_output_row=int(stop-1),
                                   baseline_rms_mm=rms, baseline_max_mm=maximum))
    unresolved = list(range(len(quiet_runs)))
    for gain in QUIET_GAIN_GRID:
        if not unresolved:
            break
        for index in unresolved:
            row = quiet_runs[index]
            a, b = row['first_output_row'], row['last_output_row']+1
            alpha[a:b] = cap_alpha[a:b]*gain
        # The first candidate is already reconstructed and position-capped.
        if gain != 1.0:
            xyz, q, correction = reconstruct(alpha)
        candidate_tip_local = kinematics.qapply(kinematics.qconj(bq[:, wrist, None]), xyz[:, tips]-xyz[:, wrist, None])
        rejected = []
        for index in unresolved:
            row = quiet_runs[index]
            a, b = row['first_output_row'], row['last_output_row']+1
            rms, maximum = _run_stats(candidate_tip_local, a, b)
            run_cap = float(correction[a:b].max(initial=0))
            accepted = (rms <= row['baseline_rms_mm']+QUIET_CONSTRAINT_TOLERANCE_MM
                        and maximum <= row['baseline_max_mm']+QUIET_CONSTRAINT_TOLERANCE_MM
                        and run_cap <= cap+1e-8)
            if accepted:
                row['chosen_gain'] = gain
            else:
                rejected.append(index)
        unresolved = rejected
    if unresolved:
        raise ValueError('QUIET_RUN_BASELINE_FALLBACK_FAILED')
    # Final geometry is recomputed from the selected alphas, never post-edited.
    xyz, q, correction = reconstruct(alpha)
    if correction.max(initial=0) > cap+1e-8:
        raise ValueError('FINAL_HAND_CORRECTION_CAP_VIOLATION')
    final_tip_local = kinematics.qapply(kinematics.qconj(bq[:, wrist, None]), xyz[:, tips]-xyz[:, wrist, None])
    for row in quiet_runs:
        a, b = row['first_output_row'], row['last_output_row']+1
        rms, maximum = _run_stats(final_tip_local, a, b)
        if (rms > row['baseline_rms_mm']+QUIET_CONSTRAINT_TOLERANCE_MM
                or maximum > row['baseline_max_mm']+QUIET_CONSTRAINT_TOLERANCE_MM):
            raise ValueError('FINAL_QUIET_RUN_SECOND_DIFFERENCE_CONSTRAINT_VIOLATION')
        if row['chosen_gain'] == 0 and (not np.array_equal(xyz[a:b], bp[a:b])
                                         or not np.array_equal(q[a:b], bq[a:b])):
            raise ValueError('ZERO_GAIN_MUST_BE_EXACT_LENGTH_ONLY_BASELINE')
    # Independently check the geometric meaning of the transported rotations.
    baseline_parent_local_bones = kinematics.qapply(kinematics.qconj(bq[:, parents[actived]]), vectors)
    transported_bones = kinematics.qapply(q[:, parents[actived]], baseline_parent_local_bones)
    new_bones = xyz[:, actived] - xyz[:, parents[actived]]
    axis_error = np.linalg.norm(transported_bones-new_bones, axis=-1)
    terminal_relative_new = kinematics.qnorm(kinematics.qmul(kinematics.qconj(q[:, parents[tips]]), q[:, tips]))
    terminal_aligned = np.where(np.sum(terminal_relative_new*tip_relative, axis=-1, keepdims=True) < 0,
                                -terminal_relative_new, terminal_relative_new)
    terminal_error = np.degrees(4*np.arcsin(np.clip(np.linalg.norm(terminal_aligned-tip_relative, axis=-1)/2, 0, 1)))
    if axis_error.max(initial=0) > 1e-7 or terminal_error.max(initial=0) > 1e-7:
        raise ValueError('ORIENTATION_GEOMETRY_TRANSPORT_RESIDUAL')
    output = baseline.copy()
    output[:, pc], output[:, qc] = xyz, q
    if not np.isfinite(output).all():
        raise ValueError('NONFINITE_EVENT_V3_OUTPUT')
    fixed = [x for x in diagnostics['links'] if x['fixed']]
    residual = np.stack([np.abs(np.linalg.norm(xyz[:, x['child_index']] - xyz[:, x['parent_index']],
                                             axis=-1) - x['reference_length_mm']) for x in fixed], axis=-1)
    maxres = float(residual.max(initial=0))
    if maxres > .01:
        raise ValueError('FIXED_LENGTH_RESIDUAL_EXCEEDS_0P01_MM')
    return output
