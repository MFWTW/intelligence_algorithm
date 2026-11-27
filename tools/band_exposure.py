#!/usr/bin/env python3
"""How much time does the base spend inside the planner's inscribed circle?

The failure this measures: the base is driven within 0.42 m (empty) or 0.38 m
(loaded) of a wall, which is the radius NavFn calls inscribed. The cell the
base stands on then costs 99, NavFn refuses to start a plan from it and Nav2
refuses to reverse out of it - the leg is dead until something retreats the
base by hand. 2026-11-26 lost three rounds that way.

Clearance is measured on the static map against the Gazebo-truth pose, so the
number is "how close did the body actually get to a wall", not how close the
localiser thought it was. map == odom on this stack (static map->odom, AMCL
does not broadcast), so odom coordinates index the map directly.

The interesting split is by speed: a base parked 0.38 m from the fridge during
a grasp is expected - it has to reach in - while a base carrying 1.3 m/s into a
0.38 m corner is the controller cutting the turn. Report both.

Usage: python3 tools/band_exposure.py runs/<round> [runs/<round> ...] [--csv out.csv]
"""
import argparse
import csv
import json
import pathlib

import numpy as np
from PIL import Image

# PGMs are stored top row first; the map frame counts y upward from the bottom,
# so the array has to be flipped or every clearance is read off the wrong row.
DEFAULT_MAP = 'src/yzbot/bot_navigation/maps/room_from_world.pgm'
RES = 0.05
ORIGIN_X = -15.050
ORIGIN_Y = -16.000
EMPTY_RADIUS = 0.42      # inscribed radius while empty (0.32 body + 0.10 margin)
LOADED_RADIUS = 0.38     # same, loaded (0.28 + 0.10)
FAST = 0.60              # m/s, "this is navigation, not docking"


def load_clearance(path):
    image = np.array(Image.open(path).convert('L'))[::-1]
    occupied = image < 50
    # Distance from every cell to the nearest occupied cell, in metres.
    from scipy.ndimage import distance_transform_edt
    return distance_transform_edt(~occupied) * RES


def clearance_at(field, x, y):
    mx = int(round((x - ORIGIN_X) / RES))
    my = int(round((y - ORIGIN_Y) / RES))
    if not (0 <= mx < field.shape[1] and 0 <= my < field.shape[0]):
        return None
    return float(field[my, mx])


def episodes(samples):
    """Contiguous stretches of (sample, clearance) below the band radius."""
    out, current = [], []
    for sample in samples:
        if sample['clearance'] <= EMPTY_RADIUS:
            current.append(sample)
        elif current:
            out.append(current)
            current = []
    if current:
        out.append(current)
    return out


def analyse(run_dir, field):
    events = run_dir / 'events.jsonl'
    track = []
    for line in open(events):
        record = json.loads(line)
        if record['topic'] != 'odom':
            continue
        d = record['d']
        value = clearance_at(field, d['x'], d['y'])
        if value is None:
            continue
        track.append({'t': record['t'], 'x': d['x'], 'y': d['y'],
                      'v': abs(d.get('vx', 0.0)), 'clearance': value})
    if not track:
        return None
    clearances = np.array([s['clearance'] for s in track])
    fast = [s for s in track if s['v'] >= FAST]
    tight = [s for s in track if s['clearance'] <= EMPTY_RADIUS]
    # odom runs at ~30 Hz, so counting samples and multiplying by a nominal
    # period overstated every duration six-fold the first time this was run.
    # Integrate the actual stamps instead.
    def spanned(samples):
        if not samples:
            return 0.0
        return float(samples[-1]['t'] - samples[0]['t'])
    tight_fast = [s for s in tight if s['v'] >= FAST]
    spans = episodes(track)
    longest = max(spans, key=len) if spans else []
    return {
        'run': run_dir.name,
        'samples': len(track),
        'min': float(clearances.min()),
        'p1': float(np.percentile(clearances, 1)),
        'p5': float(np.percentile(clearances, 5)),
        'p50': float(np.percentile(clearances, 50)),
        'samples_in_band': len(tight),
        'seconds_in_band': spanned(tight),
        'band_while_fast': spanned(tight_fast),
        'fast_samples': len(fast),
        'fast_span': spanned(fast),
        'loads_in_band': len(episodes(track)),
        'longest_band_s': round(spanned(longest), 1),
        'longest_band_at': (f"({longest[0]['x']:.2f},{longest[0]['y']:.2f})"
                            if longest else ''),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('runs', nargs='+')
    parser.add_argument('--map', default=DEFAULT_MAP)
    parser.add_argument('--csv', default='')
    args = parser.parse_args(argv)

    field = load_clearance(args.map)
    rows = []
    for name in args.runs:
        row = analyse(pathlib.Path(name), field)
        if row is None:
            print(f'{name}: no odom samples')
            continue
        rows.append(row)
        print(f"{row['run']}: n={row['samples']} "
              f"clearance p1 {row['p1']:.2f} p5 {row['p5']:.2f} p50 {row['p50']:.2f} "
              f"min {row['min']:.2f} | in-band {row['seconds_in_band']:.1f}s "
              f"(fast {row['band_while_fast']:.1f}s of {row['fast_span']:.0f}s) "
              f"longest {row['longest_band_s']}s at {row['longest_band_at']}")

    if args.csv and rows:
        with open(args.csv, 'w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f'wrote {args.csv}')


if __name__ == '__main__':
    main()
