#!/usr/bin/env python3
"""Why did the base sit still?  Dwell episodes per round, with a cause tally.

The base pose stream (gazebo ground truth, 5 Hz) is enough to say *that* the
robot stopped; the stack log says *why*.  A dwell episode is a window in which
the base moves less than DWELL_DIST metres for at least DWELL_MIN seconds.
Every log line stamped inside that window is classified; the tally is printed,
so "stuck for 20 s" becomes "9 x planner failed, 14 x collision ahead".

    tools/dwell_report.py --dir runs/<round> [--dir ...] [--csv out.csv]
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

DWELL_MIN = 2.5          # seconds of near-zero motion that count as a dwell
DWELL_DIST = 0.08        # metres of allowed drift inside a dwell window
WINDOW = 3.0             # local window used to decide "not moving"

STAMP_RE = re.compile(r'\[(\d{9,}\.\d+)\]')
START_RE = re.compile(r'Started (\S+) -> (\S+) attempt (\d+)')
DONE_RE = re.compile(r'DELIVERY COMPLETED: (\S+) placed in warehouse (\S+)')
TIMING_RE = re.compile(r'\[fixed_joint_pick_place\]: TIMING (\S+(?: \S+)*?) ([0-9.]+)$')

# Ordered: the first match wins, so put the specific causes before the generic.
CAUSES = [
    ('planner_no_path', re.compile(
        r'GridBased: failed to create plan|failed to generate a valid path')),
    ('collision_ahead', re.compile(r'detected collision ahead')),
    ('backup_failed', re.compile(r'Collision Ahead - Exiting DriveOnHeading|backup failed')),
    ('spin_failed', re.compile(r'Collision Ahead - Exiting Spin|spin failed')),
    ('contact_guard', re.compile(r'held at [0-9.]+ m from the base')),
    ('stale_odom', re.compile(r'stale odometry')),
    ('band_start', re.compile(r'inside an inscribed band')),
    ('yield_arc_pending', re.compile(r'yield arc pending')),
    ('hold_still', re.compile(r'holding still for a moving obstacle')),
    ('yield_hold', re.compile(r'Yield hold: speed limit')),
    ('yield_stop', re.compile(r'YIELD.*decision=stop')),
    ('yield_cap', re.compile(r'YIELD.*decision=cap')),
    ('yield_arc', re.compile(r'YIELD.*decision=arc')),
    ('yield_go', re.compile(r'YIELD.*decision=go')),
    ('monitor_flip', re.compile(r'to approach for .* away from collision|to continue normal operation')),
    ('behavior', re.compile(r'Running backup|Running spin|Running wait')),
    ('recovery', re.compile(r'recovery|relax|band retry|wedged')),
    ('vision', re.compile(r'vision_align|creep_to_grasp|detected cube|grasp offset')),
    ('arm', re.compile(r'arm: |follow_joint_trajectory|Executor result')),
    ('nav_retry', re.compile(r'navigate_to|Navigation to .* failed')),
]

CUBE_RADIUS = 0.32       # base footprint radius, for context in the report


def load_track(run_dir: Path):
    """[(wall_time, x, y)] from the recorder; ground truth, not AMCL."""
    ev = run_dir / 'events.jsonl'
    pts = []
    if not ev.exists():
        return pts
    with ev.open(encoding='utf-8', errors='replace') as handle:
        for line in handle:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            d = rec.get('d') or {}
            if rec.get('topic') == 'link_states':
                if 'base_footprint' in str(d.get('name', '')):
                    pts.append((rec['w'], d['x'], d['y']))
            elif rec.get('topic') == 'model_states':
                robot = d.get('robot')
                if robot and not pts:
                    pts.append((rec['w'], robot[0], robot[1]))
    return pts


def find_dwells(pts, min_s=DWELL_MIN, max_disp=DWELL_DIST, window=WINDOW):
    """Windows where the base makes no progress; merged, in wall time."""
    if len(pts) < 4:
        return []
    dwells = []
    i = 0
    while i < len(pts):
        t0, x0, y0 = pts[i]
        j = i
        while j + 1 < len(pts) and pts[j + 1][0] - t0 <= window:
            j += 1
        far = max(((pts[k][1] - x0) ** 2 + (pts[k][2] - y0) ** 2) ** 0.5
                  for k in range(i, j + 1))
        if far <= max_disp and j > i:
            # Extend forward while still inside the same stop.
            k = j
            while k + 1 < len(pts) and pts[k + 1][0] - t0 <= 600:
                if ((pts[k + 1][1] - x0) ** 2 + (pts[k + 1][2] - y0) ** 2) ** 0.5 > max_disp:
                    break
                k += 1
            dur = pts[k][0] - t0
            if dur >= min_s:
                dwells.append({'t0': t0, 't1': pts[k][0], 'dur': dur,
                               'x': round(x0, 2), 'y': round(y0, 2),
                               'drift': round(far, 3)})
            i = k + 1
        else:
            i += 1
    return dwells


def load_log(run_dir: Path):
    """[(wall_time, line)] and the object timeline from a round's stack.log."""
    path = run_dir / 'stack.log'
    lines = []
    objects = []
    if not path.exists():
        return lines, objects
    with path.open(encoding='utf-8', errors='replace') as handle:
        for line in handle:
            m = STAMP_RE.search(line)
            stamp = float(m.group(1)) if m else None
            if stamp is not None:
                lines.append((stamp, line))
            s = START_RE.search(line)
            if s and stamp is not None:
                objects.append({'t': stamp, 'object': s.group(1),
                                'zone': s.group(2), 'attempt': int(s.group(3))})
            d = DONE_RE.search(line)
            if d and objects:
                objects[-1]['done'] = True
    return lines, objects


def step_timeline(lines, objects):
    """[(start, end, object, label)] from Started + TIMING lines.

    TIMING is printed when a step *ends*, and steps run back to back, so the
    step covering instant t is the one whose end is the first after t.
    """
    steps = []
    for idx, obj in enumerate(objects):
        begin = obj['t']
        end_of_leg = objects[idx + 1]['t'] if idx + 1 < len(objects) else 1e18
        for stamp, line in lines:
            if not (begin <= stamp <= end_of_leg):
                continue
            m = TIMING_RE.search(line)
            if m:
                label = m.group(1).strip()
                steps.append({'t0': begin, 't1': stamp,
                              'object': obj['object'], 'zone': obj['zone'],
                              'label': label})
                begin = stamp
    return steps


def phase_of(steps, t):
    for step in steps:
        if step['t0'] <= t <= step['t1']:
            return step['label']
    return ''


def attribute(lines, dwells, objects, steps):
    out = []
    for dw in dwells:
        tally = Counter()
        for stamp, line in lines:
            if dw['t0'] <= stamp <= dw['t1']:
                for name, rx in CAUSES:
                    if rx.search(line):
                        tally[name] += 1
                        break
        obj = ''
        for o in objects:
            if o['t'] <= dw['t0']:
                obj = f"{o['object']}->{o['zone']}"
        ranked = [f'{k} x{v}' for k, v in tally.most_common(4)]
        phase = phase_of(steps, (dw['t0'] + dw['t1']) / 2)
        out.append({**dw, 'seg': obj, 'phase': phase, 'causes': dict(tally),
                    'top': ', '.join(ranked) if ranked else 'silent'})
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', action='append', required=True)
    parser.add_argument('--csv', default='')
    args = parser.parse_args()

    rows = []
    for name in args.dir:
        run_dir = Path(name)
        pts = load_track(run_dir)
        lines, objects = load_log(run_dir)
        dwells = find_dwells(pts)
        # Sitting on the spawn pose while the start gate is closed is not a
        # navigation dwell; drop anything parked within 0.4 m of the origin
        # during the first 20 s of tracking.
        dwells = [d for d in dwells
                  if not (abs(d['x']) < 0.4 and abs(d['y']) < 0.4
                          and d['t0'] - pts[0][0] < 20.0)]
        steps = step_timeline(lines, objects)
        if objects:
            last_done = max(o['t'] for o in objects if o.get('done')) \
                if any(o.get('done') for o in objects) else None
            if last_done:
                dwells = [d for d in dwells if d['t0'] <= last_done]
        ranked = attribute(lines, dwells, objects, steps)
        total = sum(d['dur'] for d in ranked)
        print(f'\n== {run_dir.name}  ({len(ranked)} dwells, {total:.1f} s '
              f'of {len(pts) / 5.0:.0f} s tracked)')
        for d in sorted(ranked, key=lambda r: -r['dur'])[:12]:
            print(f"  {d['dur']:6.1f}s at ({d['x']:+.2f},{d['y']:+.2f}) "
                  f"{d['seg']:<24} phase={d['phase']:<34} {d['top']}")
        for d in ranked:
            rows.append({'round': run_dir.name, 'seg': d['seg'],
                         'phase': d['phase'], 'dur_s': round(d['dur'], 1),
                         'x': d['x'], 'y': d['y'], 'drift_m': d['drift'],
                         'top_cause': d['top']})
    if args.csv and rows:
        Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, 'w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f'\ncsv -> {args.csv} ({len(rows)} rows)')


if __name__ == '__main__':
    main()
