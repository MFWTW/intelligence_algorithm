#!/usr/bin/env python3
"""How far does the base stray from the route Nav2 planned for a leg?

The 2026-11-26 14:39 rule 2 round was lost to a corner cut: the leg's route bent
27 deg about 1 m ahead, the base carried 1.3 m/s through it, and by the time the
controller cut the speed the base stood 0.35 m to the inside of that route - in
a cell the planner treats as inscribed (0.38 m), so NavFn refused to start from
there and the leg died. The retreat threshold was blamed first; this tool
measures the tracking error itself, so a controller change can be judged on a
distribution instead of one more anecdote.

Route, not plan: Nav2 republishes a plan about every 1.1 s from the *current*
pose, so comparing the base against the plan that is active right now always
says "on the line" - a drifting base simply gets a new line through itself.
The quantity that matters is the distance from the route the leg started with,
which is why a gap of more than --leg-gap seconds between plans is read as a new
leg and its first plan becomes that leg's route.

Frames: plans are in map, /odom starts at (0, 0). Applying the constant offset
between the two at the moment of the plan would be right, but the offset is
what AMCL drift changes; instead the deviation is measured in the plan's own
frame after aligning the plan's start with the base, so only the *shape* of the
execution is scored.

Usage:
    python3 tools/tracking_error.py runs/<round> [...] [--csv out]
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys


def load_probe(run_dir: pathlib.Path):
    path = run_dir / 'probe.jsonl'
    if not path.exists():
        return None
    plans, odom = [], []
    with path.open() as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            stamp = row.get('t')
            if stamp is None:
                continue
            payload = row.get('d') or {}
            if row.get('topic') == 'plan':
                pts = payload.get('pts') or []
                if len(pts) >= 2:
                    plans.append((stamp, pts))
            elif row.get('topic') == 'odom' and 'x' in payload:
                plans_ok = True
                odom.append((stamp, payload['x'], payload['y']))
    if not plans or not odom:
        return None
    plans.sort(key=lambda item: item[0])
    odom.sort()
    return plans, odom


def distance_to_polyline(pts, x, y):
    best = 1e9
    for px, py in pts:
        d = (px - x) ** 2 + (py - y) ** 2
        if d < best:
            best = d
    return math.sqrt(best)


def legs_of(plans, gap):
    """Split plans into legs: a gap larger than `gap` starts a new route."""
    legs = []
    for stamp, pts in plans:
        if not legs or stamp - legs[-1][-1][0] > gap:
            legs.append([])
        legs[-1].append((stamp, pts))
    return legs


def analyse(run_dir: pathlib.Path, leg_gap: float):
    loaded = load_probe(run_dir)
    if loaded is None:
        return None
    plans, odom = loaded
    legs = legs_of(plans, leg_gap)
    samples = []
    for index, leg in enumerate(legs):
        begin = leg[0][0]
        end = legs[index + 1][0][0] if index + 1 < len(legs) else odom[-1][0]
        route = leg[0][1]
        # align the route start with the base at the moment the leg was planned
        base = min(odom, key=lambda row: abs(row[0] - begin))
        ox, oy = route[0][0] - base[1], route[0][1] - base[2]
        shifted = [(px - ox, py - oy) for px, py in route]
        window = [(t, x, y) for t, x, y in odom if begin <= t < end]
        for k, (stamp, x, y) in enumerate(window):
            prev = window[k - 1] if k else None
            speed = 0.0
            if prev is not None and stamp > prev[0]:
                speed = math.hypot(x - prev[1], y - prev[2]) / (stamp - prev[0])
            samples.append((stamp, speed, distance_to_polyline(shifted, x, y)))
    if not samples:
        return None
    errors = sorted(s[2] for s in samples)
    moving = [s for s in samples if s[1] >= 0.30]
    worst = max(samples, key=lambda s: s[2])
    return {
        'legs': len(legs),
        'samples': len(samples),
        'p50': errors[len(errors) // 2],
        'p95': errors[int(len(errors) * 0.95)],
        'max': errors[-1],
        'over_020': sum(1 for s in samples if s[2] > 0.20),
        'moving_over_020': sum(1 for s in moving if s[2] > 0.20),
        'moving': len(moving),
        'worst': worst,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs='+')
    parser.add_argument('--leg-gap', type=float, default=3.0,
                        help='a plan silence longer than this starts a new leg (s)')
    parser.add_argument('--csv', default='')
    args = parser.parse_args(argv)

    results = []
    for name in args.runs:
        run_dir = pathlib.Path(name)
        row = analyse(run_dir, args.leg_gap)
        if row is None:
            print(f'{run_dir}: no probe.jsonl with plan+odom')
            continue
        row['run'] = run_dir.name
        results.append(row)
        stamp, speed, err = row['worst']
        print(f"{row['run']}: legs={row['legs']} n={row['samples']} "
              f"p50={row['p50']:.2f} p95={row['p95']:.2f} max={row['max']:.2f} "
              f">0.20m: {row['over_020']}/{row['moving_over_020']} moving  "
              f"worst t={stamp:.0f}s v={speed:.2f}")
    if args.csv and results:
        out = pathlib.Path(args.csv)
        out.write_text('run,legs,samples,p50,p95,max,over_0.20,moving_over_0.20,moving\n'
                       + ''.join(
                           f"{r['run']},{r['legs']},{r['samples']},{r['p50']:.2f},"
                           f"{r['p95']:.2f},{r['max']:.2f},{r['over_020']},"
                           f"{r['moving_over_020']},{r['moving']}\n" for r in results))
        print(f'wrote {out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
