#!/usr/bin/env python3
"""Is room_from_world.pgm aligned with the world the robot actually drives in?

Two questions kept coming back: the planner refuses to start from cells that
the body fits through, and at one stall the map claimed 0.65 m of clearance
where the laser measured 0.50 m. Either the map is offset from the world, or
something unmapped is standing there. This tool answers the first question by
scan matching: the laser returns of a run, carried into the map frame with the
Gazebo ground-truth pose, are correlated against the map's occupied cells.

Because map == odom here (the launch publishes an identity map->odom and AMCL
only reports), the ground-truth pose is also what the planner believed, so the
offset measured is the offset the planner suffers from.

Usage:
    python3 tools/map_alignment.py runs/<round> [--map ...]

Reading: `before` is the mean distance from laser points to the nearest mapped
wall with no correction, `after` is the same after shifting the map by the
reported offset. A map built from this world should sit within one cell
(0.05 m); a systematic offset of 0.1 m or more means the planner and the world
are disagreeing about where the walls are.
"""
from __future__ import annotations

import argparse
import base64
import json
import math
import pathlib
import sys
import zlib

import numpy as np
from scipy import ndimage


def read_pgm(path: pathlib.Path):
    data = path.read_bytes()
    fields, index = [], 0
    while len(fields) < 4:
        while data[index:index + 1].isspace():
            index += 1
        if data[index:index + 1] == b'#':
            while data[index:index + 1] not in (b'\n', b''):
                index += 1
            continue
        end = index
        while not data[end:end + 1].isspace():
            end += 1
        fields.append(data[index:end])
        index = end
    width, height = int(fields[1]), int(fields[2])
    index += 1
    pixels = np.frombuffer(data[index:index + width * height], dtype=np.uint8)
    return pixels.reshape(height, width)


def load_map(meta_path: pathlib.Path):
    meta = {}
    for line in meta_path.read_text().splitlines():
        if ':' not in line or line.strip().startswith('#'):
            continue
        key, value = line.split(':', 1)
        meta[key.strip()] = value.strip()
    pgm = meta_path.with_name(meta['image'])
    grid = read_pgm(pgm)
    resolution = float(meta['resolution'])
    (ox, oy, _) = [float(v) for v in meta['origin'].strip('[]').split(',')]
    return grid, resolution, ox, oy


def load_run(run_dir: pathlib.Path, limit=400):
    truths, scans = [], []
    for line in (run_dir / 'events.jsonl').open():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (row.get('topic') == 'link_states'
                and 'base_footprint' in row['d'].get('name', '')):
            truths.append((row['t'], row['d']['x'], row['d']['y'], row['d']['yaw']))
    truths.sort()
    for line in (run_dir / 'probe.jsonl').open():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get('topic') == 'scan' and len(scans) < limit:
            scans.append((row['t'], row['d']))
    return truths, scans


def scan_points(payload, pose):
    """Laser returns in the map frame, anchored on the ground-truth pose."""
    stamp, x, y, yaw = pose
    data = payload
    raw = zlib.decompress(base64.b64decode(data['ranges']))
    ranges = np.frombuffer(raw[:len(raw) // 4 * 4], dtype=np.float32)
    angles = data['amin'] + np.arange(len(ranges)) * data['ainc']
    good = (ranges > 0.20) & (ranges < 12.0) & np.isfinite(ranges)
    rng, ang = ranges[good], angles[good]
    px = rng * np.cos(ang)
    py = rng * np.sin(ang)
    cos, sin = math.cos(yaw), math.sin(yaw)
    return np.stack([x + px * cos - py * sin, y + px * sin + py * cos], axis=1)


def gather_points(truths, scans, stride=4):
    times = np.array([t[0] for t in truths])
    points = []
    for stamp, payload in scans[::stride]:
        index = int(np.argmin(np.abs(times - stamp)))
        if abs(times[index] - stamp) > 0.5:
            continue
        points.append(scan_points(payload, truths[index]))
    if not points:
        return np.empty((0, 2))
    return np.concatenate(points, axis=0)


def edt_of(grid, resolution, unknown_free=True):
    occupied = grid < 50
    if unknown_free:
        occupied |= False
    return ndimage.distance_transform_edt(~occupied) * resolution


def cost_of(points, edt, resolution, ox, oy, dx=0.0, dy=0.0, dtheta=0.0):
    pts = points.copy()
    if dtheta:
        cos, sin = math.cos(math.radians(dtheta)), math.sin(math.radians(dtheta))
        pts = pts @ np.array([[cos, sin], [-sin, cos]])
    cols = np.rint((pts[:, 0] + dx - ox) / resolution).astype(int)
    rows = np.rint((pts[:, 1] + dy - oy) / resolution).astype(int)
    rows = edt.shape[0] - 1 - rows          # PGM rows run top-down
    ok = (cols >= 0) & (cols < edt.shape[1]) & (rows >= 0) & (rows < edt.shape[0])
    if not ok.any():
        return float('inf'), 0
    return float(edt[rows[ok], cols[ok]].mean()), int(ok.sum())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs='+')
    parser.add_argument('--map', default='src/yzbot/bot_navigation/maps/room_from_world.yaml')
    parser.add_argument('--span', type=float, default=0.6)
    parser.add_argument('--step', type=float, default=0.01)
    args = parser.parse_args(argv)

    grid, resolution, ox, oy = load_map(pathlib.Path(args.map))
    edt = edt_of(grid, resolution)
    print(f'map {args.map}: {grid.shape[1]}x{grid.shape[0]} @ {resolution} m, '
          f'occupied cells {int((grid < 50).sum())}')

    for name in args.runs:
        run_dir = pathlib.Path(name)
        truths, scans = load_run(run_dir)
        points = gather_points(truths, scans)
        if len(points) < 500:
            print(f'{run_dir.name}: not enough laser points ({len(points)})')
            continue
        before, used = cost_of(points, edt, resolution, ox, oy)
        best = (before, 0.0, 0.0, 0.0)
        span = np.arange(-args.span, args.span + 1e-9, args.step)
        for dx in span:
            for dy in span:
                value, _ = cost_of(points, edt, resolution, ox, oy, dx, dy)
                if value < best[0]:
                    best = (value, dx, dy, 0.0)
        dx, dy = best[1], best[2]
        best_rot = best
        for theta in np.arange(-1.5, 1.51, 0.25):
            value, _ = cost_of(points, edt, resolution, ox, oy, dx, dy, theta)
            if value < best_rot[0]:
                best_rot = (value, dx, dy, theta)
        print(f'{run_dir.name}: points={len(points)} used={used} '
              f'before={before:.3f} m  best={best_rot[0]:.3f} m at '
              f'dx={best_rot[1]:+.2f} dy={best_rot[2]:+.2f} '
              f'dtheta={best_rot[3]:+.2f} deg')
    return 0


if __name__ == '__main__':
    sys.exit(main())
