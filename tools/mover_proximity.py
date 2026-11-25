#!/usr/bin/env python3
"""How close do the patrolling boxes come to the base, and what happens there?

The complaint is that while the robot dodges a moving obstacle the box squeezes
past it ("擦着车过去").  Ground truth is in the recorder: model_states carries
every mover, link_states carries the base.  This tool reports, per round and
per mover:

  * the closest centre-to-centre distance ever reached;
  * every encounter window inside NEAR metres (default 0.55 m, i.e. just above
    the 0.42 m contact guard), with duration, minimum distance, how fast the
    base was going, and how fast the gap was closing.

Centre distance is the number to trust here: both poses are Gazebo world poses.
Subtract the base footprint radius (~0.32 m) and a mover half-box (~0.15 m) to
get a rough clearance, which is why < 0.55 m centre distance already means
"paint touching".
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

NEAR = 0.55      # metres, centre to centre
TOUCH = 0.47     # below this the guard is usually already holding the box off


def load(path: Path):
    """Return base samples and mover samples from events.jsonl."""
    base, movers = [], {}
    if not path.exists():
        return base, movers
    with path.open(encoding='utf-8', errors='replace') as handle:
        for line in handle:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            topic, d = rec.get('topic'), rec.get('d') or {}
            if topic == 'link_states' and 'base_footprint' in str(d.get('name', '')):
                base.append((rec['w'], rec['t'], d['x'], d['y']))
            elif topic == 'model_states':
                for name, pose in d.items():
                    if name.startswith('mover'):
                        movers.setdefault(name, []).append(
                            (rec['w'], rec['t'], pose[0], pose[1]))
    return base, movers


def nearest(samples, t):
    """Sample closest in time to t (samples are time-ordered)."""
    lo, hi = 0, len(samples) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if samples[mid][0] < t:
            lo = mid + 1
        else:
            hi = mid
    best = samples[lo]
    if lo and abs(samples[lo - 1][0] - t) < abs(best[0] - t):
        best = samples[lo - 1]
    return best


def base_speed(base, t, half=0.6):
    """Ground-truth speed of the base around t, m/s."""
    before = nearest(base, t - half)
    after = nearest(base, t + half)
    dt = after[0] - before[0]
    if dt <= 0:
        return 0.0
    return ((after[2] - before[2]) ** 2 + (after[3] - before[3]) ** 2) ** 0.5 / dt


def episodes(series, near=NEAR):
    """Contiguous windows where the gap is inside `near` metres."""
    out, cur = [], None
    for w, trel, dist, closing in series:
        if dist <= near:
            if cur is None:
                cur = {'w0': w, 't0': trel, 'min': dist, 'closing': closing}
            else:
                cur['min'] = min(cur['min'], dist)
                cur['closing'] = max(cur['closing'], closing)
            cur['w1'], cur['t1'] = w, trel
        elif cur is not None:
            out.append(cur)
            cur = None
    if cur is not None:
        out.append(cur)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', action='append', required=True)
    parser.add_argument('--near', type=float, default=NEAR)
    parser.add_argument('--csv', default='')
    args = parser.parse_args()

    rows = []
    for name in args.dir:
        run_dir = Path(name)
        base, movers = load(run_dir / 'events.jsonl')
        if not base:
            print(f'{run_dir.name}: no base samples')
            continue
        print(f'\n== {run_dir.name}')
        for mover, samples in sorted(movers.items()):
            series = []
            for w, trel, mx, my in samples:
                b = nearest(base, w)
                dist = ((b[2] - mx) ** 2 + (b[3] - my) ** 2) ** 0.5
                prev = series[-1][2] if series else dist
                dt = w - series[-1][0] if series else 0.2
                closing = (prev - dist) / dt if dt > 0 else 0.0
                series.append((w, trel, dist, closing))
            eps = episodes(series, args.near)
            if not series:
                continue
            closest = min(s[2] for s in series)
            inside = sum(1 for s in series if s[2] <= TOUCH)
            print(f'  {mover}: closest {closest:.2f} m, '
                  f'{len(eps)} encounters < {args.near:.2f} m, '
                  f'{inside} samples inside {TOUCH:.2f} m')
            for ep in eps:
                speed = base_speed(base, (ep['w0'] + ep['w1']) / 2)
                dur = ep['w1'] - ep['w0']
                print(f"    {ep['t0']:7.1f}s  {dur:5.1f}s  min {ep['min']:.2f} m  "
                      f"closing {ep['closing']:.2f} m/s  base speed {speed:.2f} m/s")
                rows.append({'round': run_dir.name, 'mover': mover,
                             't_rel_s': round(ep['t0'], 1),
                             'dur_s': round(dur, 1),
                             'min_m': round(ep['min'], 3),
                             'closing_mps': round(ep['closing'], 2),
                             'base_speed_mps': round(speed, 2)})
    if args.csv and rows:
        Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, 'w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f'\ncsv -> {args.csv} ({len(rows)} rows)')


if __name__ == '__main__':
    main()
