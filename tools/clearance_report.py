#!/usr/bin/env python3
"""Did the base ever stand inside its own inscribed radius, and for how long?

NavFn cannot start a plan from a cell that sits inside the global inscribed
band: every GridBased request aborts with "failed to create plan", the behaviour
server answers "Collision Ahead" to the backup and the spin, and the leg dies as
``planner_band`` -> "no verified recovery".  Whether that happens is not decided
by the route, it is decided by *where the base happened to stop*: the route runs
0.40-0.47 m from the mapped walls in several places, the loaded inscribed radius
is 0.38 m and the empty one 0.42 m (both logged by the executor as "Nav2
geometry confirmed for ..."), and one map cell is 0.05 m.  Three runs can stop on
the free side of that line and the fourth on the other side.

This tool measures the quantity that line is drawn on: the distance from the
base centre to the nearest occupied cell of the static map, sampled at every
recorded pose.  It needs no ROS and no costmap dump - ``events.jsonl`` (odom)
and ``stack.log`` (which radius was in force) are enough, so it can be run over
a whole batch of rounds after the fact.

    python3 tools/clearance_report.py runs/<round> [...] [--csv out.csv]

Reading the output:

* ``min clearance``  - the closest the base centre ever came to a black cell.
  Below the radius in force, the planner had no legal start cell there.
* ``under radius``   - seconds spent inside the band, and the longest episode.
  This is the deadlock exposure, not just a near miss.
* ``borderline``     - seconds inside radius + 0.04 m.  A run with a large
  borderline time is one coin flip away from the failure; a run with none was
  never at risk.  That is the difference to look for between "3 ok" and "the
  4th stuck", not a difference in the route.

Caveat: ``odom`` is the pose stream used (3000+ samples, vs ~180 amcl poses),
and odom is not corrected by AMCL.  ``tools/map_alignment.py`` measures the
map-vs-world offset on the same run (0.07 m mean residual, ~0.04 m systematic on
2026-11-26), so treat the absolute clearance as +/- one cell; the *shape* of the
series and which cells it crosses are what matter.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import re
import sys

import numpy as np
from scipy import ndimage

ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_MAP = ROOT / 'src/yzbot/bot_navigation/maps/room_from_world.yaml'

RADIUS_RE = re.compile(
    r'\[(\d{9,}\.\d+)\].*Nav2 geometry confirmed for [^:]*: '
    r'local radius=([0-9.]+) m, global radius=([0-9.]+) m')
DEFAULT_RADIUS = 0.42


def read_pgm(path: pathlib.Path):
    data = path.read_bytes()
    fields, i = [], 0
    while len(fields) < 4:
        while data[i:i + 1].isspace():
            i += 1
        if data[i:i + 1] == b'#':
            while data[i:i + 1] not in (b'\n', b''):
                i += 1
            continue
        j = i
        while data[j:j + 1] and not data[j:j + 1].isspace():
            j += 1
        fields.append(data[i:j])
        i = j
    w, h, maxv = int(fields[1]), int(fields[2]), int(fields[3])
    i += 1
    pix = np.frombuffer(data[i:i + w * h], dtype=np.uint8).reshape(h, w)
    return pix, w, h, maxv


def load_map(path: pathlib.Path):
    info = {}
    for line in path.read_text(encoding='utf-8').splitlines():
        if ':' in line:
            k, v = line.split(':', 1)
            info[k.strip()] = v.strip()
    res = float(info['resolution'])
    ox, oy = [float(v) for v in info['origin'].strip('[]').split(',')[:2]]
    pix, w, h, maxv = read_pgm(path.parent / info['image'])
    occ = pix.astype(np.float32) / maxv < 0.35
    # Distance from a free cell centre to the nearest occupied cell centre, in
    # metres.  This is the same quantity the inflation layer works from, so it
    # is directly comparable to the inscribed radius.
    dist = ndimage.distance_transform_edt(~occ) * res
    return dict(occ=occ, dist=dist.astype(np.float32), res=res, ox=ox, oy=oy,
                w=w, h=h), path


def load_poses(run: pathlib.Path):
    """(t, x, y) in the odom frame, from events.jsonl (probe as fallback)."""
    out = []
    for name in ('events.jsonl', 'probe.jsonl'):
        path = run / name
        if not path.exists():
            continue
        with path.open('r', encoding='utf-8') as f:
            for line in f:
                if '"odom"' not in line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get('topic') != 'odom':
                    continue
                d = rec['d']
                out.append((rec['w'], d['x'], d['y']))
        if out:
            return out
    return out


def load_radii(run: pathlib.Path):
    """[(t, radius)] from the executor's "Nav2 geometry confirmed" lines."""
    path = run / 'stack.log'
    if not path.exists():
        return []
    steps = []
    with path.open('r', encoding='utf-8', errors='replace') as f:
        for line in f:
            m = RADIUS_RE.search(line)
            if m:
                steps.append((float(m.group(1)), float(m.group(3))))
    steps.sort()
    return steps


def radius_at(steps, t):
    value = DEFAULT_RADIUS
    for stamp, radius in steps:
        if stamp > t:
            break
        value = radius
    return value


def clearance_at(sm, x, y):
    col = int((x - sm['ox']) / sm['res'])
    row = sm['h'] - 1 - int((y - sm['oy']) / sm['res'])
    if not (0 <= row < sm['h'] and 0 <= col < sm['w']):
        return float('nan')
    return float(sm['dist'][row, col])


def episodes(t, under, min_len=0.3):
    """Contiguous True stretches of ``under`` as (t0, t1, len_samples)."""
    out, start = [], None
    for i, flag in enumerate(under):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            if t[i - 1] - t[start] >= min_len:
                out.append((t[start], t[i - 1], i - start))
            start = None
    if start is not None and t[-1] - t[start] >= min_len:
        out.append((t[start], t[-1], len(t) - start))
    return out


def analyse(run: pathlib.Path, sm, sm_path):
    poses = load_poses(run)
    if len(poses) < 20:
        return None
    steps = load_radii(run)
    w = np.array([p[0] for p in poses])
    x = np.array([p[1] for p in poses])
    y = np.array([p[2] for p in poses])
    clr = np.array([clearance_at(sm, px, py) for px, py in zip(x, y)])
    limit = np.array([radius_at(steps, wt) for wt in w])
    ok = np.isfinite(clr)
    if not ok.any():
        return None
    w, x, y, clr, limit = w[ok], x[ok], y[ok], clr[ok], limit[ok]
    i_min = int(np.argmin(clr))
    under = clr < limit
    border = (clr < limit + 0.04) & ~under
    eps = episodes(w, under)
    longest = max(eps, key=lambda e: e[1] - e[0], default=None)
    return dict(
        run=run.name, map=sm_path.name, n=len(w), t0=w[0], t1=w[-1],
        radius_first=float(limit[0]), radius_last=float(limit[-1]),
        min_clear=float(clr[i_min]), min_t=float(w[i_min]),
        min_x=float(x[i_min]), min_y=float(y[i_min]),
        margin=float(clr[i_min] - limit[i_min]),
        under_sec=float(np.sum(np.diff(w, prepend=w[0])[under])),
        under_frac=float(under.mean()),
        n_under=int(under.sum()), n_eps=len(eps),
        longest=(float(longest[0]), float(longest[1])) if longest else None,
        border_sec=float(np.sum(np.diff(w, prepend=w[0])[border])),
    )


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('runs', nargs='+', help='run directories')
    ap.add_argument('--map', default=str(DEFAULT_MAP), help='map yaml')
    ap.add_argument('--csv', help='write the per-run rows here')
    args = ap.parse_args(argv)

    sm, sm_path = load_map(pathlib.Path(args.map))
    print(f'map {sm_path}: {sm["w"]}x{sm["h"]} @ {sm["res"]} m, '
          f'{int(sm["occ"].sum())} occupied cells')

    rows = []
    for name in args.runs:
        run = pathlib.Path(name)
        r = analyse(run, sm, sm_path)
        if r is None:
            print(f'{run.name}: no pose stream')
            continue
        rows.append(r)

    rows.sort(key=lambda r: r['min_clear'])
    print(f'\n{"run":36s} {"min":>5s} {"margin":>6s} {"r0->r1":>9s} {"@t":>7s} '
          f'{"pos":>17s} {"under":>7s} {"%":>5s} {"eps":>4s} '
          f'{"longest":>15s} {"border":>6s}')
    for r in rows:
        pos = f'({r["min_x"]:5.2f},{r["min_y"]:6.2f})'
        longest = (f'{r["longest"][0]-r["t0"]:.0f}-{r["longest"][1]-r["t0"]:.0f}s'
                   if r['longest'] else '-')
        radii = f'{r["radius_first"]:.2f}->{r["radius_last"]:.2f}'
        print(f'{r["run"]:36s} {r["min_clear"]:5.2f} {r["margin"]:+6.2f} '
              f'{radii:>9s} {r["min_t"]-r["t0"]:7.1f} {pos:>17s} '
              f'{r["under_sec"]:6.1f}s {100*r["under_frac"]:4.1f} '
              f'{r["n_eps"]:4d} {longest:>15s} {r["border_sec"]:5.1f}s')

    if args.csv:
        with open(args.csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f'\nwrote {args.csv}')

    print(f'\nradius in force is read per sample from "Nav2 geometry confirmed" '
          f'(empty 0.42 m, loaded 0.38 m); {DEFAULT_RADIUS:.2f} m when the log '
          f'has no line.  odom is not AMCL-corrected: read the margin as '
          f'+/- one 0.05 m cell.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
