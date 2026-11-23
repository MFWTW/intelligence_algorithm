#!/usr/bin/env python3
"""Turn a probe.jsonl into evidence about "detected collision ahead" episodes.

Inputs (all produced by ``tools/run_round.sh <rule> <out_dir>``):
    <out>/probe.jsonl         streams recorded by tools/collision_probe.py
    <out>/stack.log           controller log (optional, used for line counts)
    <out>/driver.out          runner log (optional)
    tools/collision_episodes.md     markdown report (per episode table)
    tools/collision_episodes.png    costmap+arc+scan snapshots

Method
------
An episode is a stretch of >=0.3 s where ``/cmd_vel_nav`` is exactly zero
(Regulated Pure Pursuit publishes a zero twist when its collision check trips)
while the robot was recently moving. For each episode the script answers three
questions, each in its own column:

* geometry?  nearest LETHAL (>=253) cell to the arc RPP actually checked,
             plus the closest distance from the arc to a static-map wall.
* dynamic?   nearest laser return to the arc, and the count of lethal cells
             that have a laser return within 0.12 m.
* phantom?   lethal cells with NO laser return near them and NO static-map
             support -- i.e. stale inflation / voxel marks.

Sums over the whole round tell whether the refusals are dominated by real
geometry, by live movers, or by unsupported marks.
"""
import argparse
import base64
import json
import math
import os
import re
import struct
import sys
import zlib
from collections import defaultdict

import numpy as np

MAP_YAML = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..',
                        'src', 'yzbot', 'bot_navigation', 'maps',
                        'room_from_world.yaml')


# --------------------------------------------------------------------- loading
def decode_costmap(d):
    raw = zlib.decompress(base64.b64decode(d['data']))
    # /local_costmap/costmap carries the 0..100 view of Costmap2D: 100 is
    # LETHAL_OBSTACLE (254 internally), 99 INSCRIBED_INFLATED_OBSTACLE (253),
    # 1..98 inflation, 0 free, -1 unknown.
    arr = np.frombuffer(raw, dtype=np.uint8).astype(np.int16)
    arr[arr > 127] = arr[arr > 127] - 256
    return arr.reshape(d['h'], d['w'])  # row 0 = lowest y


LETHAL = 99      # published costmap: 100 lethal, 99 inscribed
INFLATED = 90    # comfortably inside the inflation cone
INFLATION_RADIUS = 0.36  # local_costmap inflation_layer.inflation_radius (m)


def decode_floats(d):
    raw = zlib.decompress(base64.b64decode(d['ranges']))
    return np.frombuffer(raw, dtype='<f4')


def load(path):
    streams = defaultdict(list)
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            streams[rec['topic']].append(rec)
    return streams


def read_pgm(path):
    with open(path, 'rb') as f:
        data = f.read()
    # P5 header: magic, width height, maxval, then binary
    parts = []
    i = 0
    tokens = 0
    while tokens < 4:
        # skip whitespace and comments
        while i < len(data) and data[i:i + 1].isspace():
            i += 1
        if data[i:i + 1] == b'#':
            while i < len(data) and data[i:i + 1] != b'\n':
                i += 1
            continue
        j = i
        while j < len(data) and not data[j:j + 1].isspace():
            j += 1
        parts.append(data[i:j])
        i = j
        tokens += 1
    assert parts[0] == b'P5', parts[0]
    w, h, maxv = int(parts[1]), int(parts[2]), int(parts[3])
    i += 1  # single whitespace after maxval
    pix = np.frombuffer(data[i:i + w * h], dtype=np.uint8).reshape(h, w)
    return pix, w, h, maxv


def load_static_map(path=MAP_YAML):
    info = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if ':' not in line:
                continue
            k, v = line.split(':', 1)
            info[k.strip()] = v.strip()
    res = float(info['resolution'])
    ox, oy = [float(v) for v in info['origin'].strip('[]').split(',')[:2]]
    pgm = os.path.join(os.path.dirname(path), info['image'])
    pix, w, h, maxv = read_pgm(pgm)
    # trinary: occupied > 0.65, free < 0.25 (negate=0 means white=free)
    occ = pix.astype(np.float32) / maxv < 0.35  # dark pixels = occupied
    return dict(occ=occ, res=res, ox=ox, oy=oy, w=w, h=h)


def map_index(sm, x, y):
    """Return (row, col) into sm['occ'] for world (x, y)."""
    col = ((x - sm['ox']) / sm['res']).astype(int)
    row = sm['h'] - 1 - ((y - sm['oy']) / sm['res']).astype(int)
    return row, col


# -------------------------------------------------------------------- episodes
def find_episodes(streams, min_len=0.3, speed_gate=0.3):
    nav = streams.get('cmd_vel_nav', [])
    odom = streams.get('odom', [])
    if not nav or not odom:
        return []
    t_nav = np.array([r['t'] for r in nav])
    v_nav = np.array([abs(r['d']['vx']) + abs(r['d']['wz']) for r in nav])
    t_od = np.array([r['t'] for r in odom])
    v_od = np.array([abs(r['d']['vx']) for r in odom])
    x_od = np.array([r['d']['x'] for r in odom])
    y_od = np.array([r['d']['y'] for r in odom])

    zero = v_nav < 1e-3
    episodes = []
    i = 0
    while i < len(zero):
        if not zero[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(zero) and zero[j + 1]:
            j += 1
        t0, t1 = t_nav[i], t_nav[j]
        if t1 - t0 < min_len:
            i = j + 1
            continue
        # speed in the second before the refusal
        m = (t_od >= t0 - 1.0) & (t_od <= t0 + 0.2)
        v_before = float(np.max(v_od[m])) if m.any() else 0.0
        if v_before < speed_gate:
            i = j + 1
            continue
        m2 = (t_od >= t0) & (t_od <= t1)
        caught = (m2.sum() > 0)
        dist = 0.0
        if caught:
            dist = float(np.hypot(x_od[m2][-1] - x_od[m2][0],
                                  y_od[m2][-1] - y_od[m2][0]))
        episodes.append(dict(t0=float(t0), t1=float(t1),
                             dur=float(t1 - t0), v_before=v_before,
                             coast=dist))
        i = j + 1
    return episodes


COLLISION_LINE = re.compile(
    r'\[(?P<stamp>\d+\.\d+)\] \[controller_server\]: '
    r'RegulatedPurePursuitController detected collision ahead')


def load_log_episodes(stack_log, epoch0, tol_gap=0.5):
    """Episodes straight from the controller log: the ground truth for RPP
    refusals, independent of whatever the twist stream shows."""
    if not os.path.exists(stack_log):
        return []
    stamps = []
    with open(stack_log, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            m = COLLISION_LINE.search(line)
            if m:
                stamps.append(float(m.group('stamp')) - epoch0)
    eps, run = [], []
    for t in stamps:
        if run and t - run[-1] > tol_gap:
            eps.append((run[0], run[-1]))
            run = []
        run.append(t)
    if run:
        eps.append((run[0], run[-1]))
    return [dict(t0=a, t1=b) for a, b in eps]


def probe_epoch0(streams):
    first = min((r for v in streams.values() for r in v[:1]),
                key=lambda r: r['t'], default=None)
    if first is None:
        return None
    return first['w'] - first['t']


def nearest(streams, topic, t, tol):
    rows = streams.get(topic, [])
    best = None
    for r in rows:
        dt = abs(r['t'] - t)
        if best is None or dt < best[0]:
            best = (dt, r)
    if best is None or best[0] > tol:
        return None
    return best[1]


def cell_centers(d):
    """World coordinates of every cell in a costmap snapshot (row 0 = low y)."""
    w, h, res = d['w'], d['h'], d['res']
    xs = d['ox'] + (np.arange(w) + 0.5) * res
    ys = d['oy'] + (np.arange(h) + 0.5) * res
    return xs, ys


def static_support(sm, xs, ys):
    """Bool array: does the static map mark this world point as occupied?"""
    if len(xs) == 0:
        return np.zeros(0, dtype=bool)
    row, col = map_index(sm, np.asarray(xs), np.asarray(ys))
    ok = (row >= 0) & (row < sm['h']) & (col >= 0) & (col < sm['w'])
    out = np.zeros(len(xs), dtype=bool)
    out[ok] = sm['occ'][row[ok], col[ok]]
    return out


def ep_metrics(streams, ep, sm, r_in=0.32):
    arc_rec = nearest(streams, 'collision_arc', ep['t0'] + 0.15, 1.0)
    cm_rec = nearest(streams, 'costmap', ep['t0'] + 0.15, 0.8)
    od_rec = nearest(streams, 'odom', ep['t0'], 0.3)
    scan_rec = nearest(streams, 'scan', ep['t0'] + 0.1, 0.5)
    out = dict(ep, n_arc=0, gap=float('nan'), blocked=0, zero_gap=0,
               scan_gap=float('nan'), phantom=0, inflated=0, stat=0,
               laser=0,
               arc=None, costmap=None, scan_pts=None,
               odom=od_rec['d'] if od_rec else None,
               x=float('nan'), y=float('nan'))
    if od_rec:
        out['x'], out['y'] = od_rec['d']['x'], od_rec['d']['y']
    odom_rows = streams.get('odom', [])
    if odom_rows and math.isnan(out['v_before']):
        win = [r for r in odom_rows if ep['t0'] - 1.0 <= r['t'] <= ep['t0']]
        out['v_before'] = max(
            [abs(r['d']['vx']) for r in win], default=float('nan'))
    if odom_rows and math.isnan(out['coast']):
        win = [r for r in odom_rows if ep['t0'] <= r['t'] <= ep['t1']]
        if len(win) >= 2:
            out['coast'] = math.hypot(win[-1]['d']['x'] - win[0]['d']['x'],
                                      win[-1]['d']['y'] - win[0]['d']['y'])
    if arc_rec is None or cm_rec is None:
        return out
    pts = np.array(arc_rec['d']['pts'], dtype=float)
    if len(pts) == 0:
        return out
    out['n_arc'] = len(pts)
    d = cm_rec['d']
    grid = decode_costmap(d)
    xs, ys = cell_centers(d)
    lethal = np.argwhere(grid >= LETHAL)  # (row, col)
    lethal_xy = (np.stack([xs[lethal[:, 1]], ys[lethal[:, 0]]], axis=1)
                 if len(lethal) else np.zeros((0, 2)))

    if len(lethal_xy):
        # Arc clearance: the smallest distance between any tested arc pose and
        # any lethal cell. RPP's FootprintCollisionChecker refuses the arc when
        # a lethal cell falls under the footprint polygon (inscribed radius r_in).
        dd = np.linalg.norm(pts[:, None, :] - lethal_xy[None, :, :], axis=2)
        per_pose = dd.min(axis=1)
        out['gap'] = float(per_pose.min())
        out['blocked'] = int(np.sum(per_pose <= r_in + 1e-6))
        out['zero_gap'] = int(np.sum(per_pose <= 0.02))
        # classify the lethal cells that block the arc (within r_in of a pose)
        near = np.unique(np.where(dd <= r_in)[1])
        if len(near):
            cells = lethal_xy[near]
            out['stat'] = int(np.sum(static_support(sm, cells[:, 0], cells[:, 1])))
    # scan support handled below once scan points are known
    if scan_rec is not None:
        sd = scan_rec['d']
        rng = decode_floats(sd)
        ang = sd['amin'] + np.arange(len(rng)) * sd['ainc']
        good = np.isfinite(rng) & (rng >= sd['rmin']) & (rng <= sd['rmax'])
        if od_rec:
            bx, by, yaw = od_rec['d']['x'], od_rec['d']['y'], od_rec['d']['yaw']
            lx = bx + rng[good] * np.cos(yaw + ang[good])
            ly = by + rng[good] * np.sin(yaw + ang[good])
            out['scan_pts'] = np.stack([lx, ly], axis=1)
            dd = np.linalg.norm(pts[:, None, :] - out['scan_pts'][None, :, :],
                                axis=2)
            out['scan_gap'] = float(np.min(dd))
            if len(lethal_xy):
                # scan support for the lethal cells near the arc
                cells = lethal_xy[
                    np.min(np.linalg.norm(
                        lethal_xy[:, None, :] - pts[None, :, :], axis=2),
                        axis=1) <= r_in]
                if len(cells):
                    sd2 = np.linalg.norm(
                        cells[:, None, :] - out['scan_pts'][None, :, :], axis=2)
                    sup_laser = sd2.min(axis=1) <= 0.15
                    sup_static = static_support(sm, cells[:, 0], cells[:, 1])
                    out['laser'] = int(np.sum(sup_laser))
                    out['stat'] = int(np.sum(sup_static))
                    # The 99 cells are the inscribed band, not obstacle marks:
                    # they are legitimate whenever a *supported* obstacle cell
                    # sits within the inflation radius. Only the cells no live
                    # obstacle can explain are the stale "shadow" marks.
                    costs = grid[lethal[:, 0], lethal[:, 1]]
                    marker = costs >= 100
                    sup_marker = marker & static_support(
                        sm, lethal_xy[:, 0], lethal_xy[:, 1])
                    ml = np.min(np.linalg.norm(
                        lethal_xy[:, None, :] - out['scan_pts'][None, :, :],
                        axis=2), axis=1) <= 0.15
                    sup_marker = sup_marker | (marker & ml)
                    if sup_marker.any():
                        dm = np.min(np.linalg.norm(
                            cells[:, None, :] - lethal_xy[sup_marker][None, :, :],
                            axis=2), axis=1)
                        inflated = (dm <= INFLATION_RADIUS) & ~sup_laser \
                            & ~sup_static
                    else:
                        inflated = np.zeros(len(cells), dtype=bool)
                    out['inflated'] = int(np.sum(inflated))
                    out['phantom'] = int(np.sum(
                        ~sup_laser & ~sup_static & ~inflated))
    return out


# --------------------------------------------------------------------- plotting
def plot_episode(path, m, sm):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MplPolygon

    fig, ax = plt.subplots(figsize=(9, 9), dpi=110)
    d = m['costmap']
    grid = decode_costmap(d)
    w, h, res = d['w'], d['h'], d['res']
    xs = d['ox'] + (np.arange(w) + 0.5) * res
    ys = d['oy'] + (np.arange(h) + 0.5) * res
    img = np.where(grid >= LETHAL, 3, np.where(grid >= INFLATED, 2,
                    np.where(grid > 0, 1, np.where(grid < 0, -1, 0))))
    ax.imshow(img, origin='lower', extent=[xs[0], xs[-1], ys[0], ys[-1]],
              cmap='Greys', vmin=-1, vmax=3)
    pts = np.array(m['arc']['d']['pts'])
    ax.plot(pts[:, 0], pts[:, 1], 'b-', lw=2, label='collision arc')
    if m['scan_pts'] is not None and len(m['scan_pts']):
        ax.scatter(m['scan_pts'][:, 0], m['scan_pts'][:, 1], s=3, c='tab:green',
                   label='laser returns', alpha=0.7)
    fp = nearest_stream_any(m, 'footprint')
    if fp:
        ax.add_patch(MplPolygon(np.array(fp['d']['pts']), closed=True,
                                fill=False, edgecolor='tab:red', lw=1.5,
                                label='footprint'))
    ax.plot(m['x'], m['y'], 'ko', ms=5)
    ax.set_title(f"t={m['t0']:.1f}s dur={m['dur']:.1f}s "
                 f"v_before={m['v_before']:.2f} gap={m['gap']:.2f} "
                 f"phantom={m['phantom']}")
    ax.set_aspect('equal')
    ax.legend(loc='upper right')
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def nearest_stream_any(m, key):
    return m.get('_fp')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in-dir', required=True)
    ap.add_argument('--inscribed', type=float, default=0.32,
                    help='inscribed radius of the local footprint (m)')
    args = ap.parse_args()

    streams = load(os.path.join(args.in_dir, 'probe.jsonl'))
    sm = load_static_map()
    print('streams: ' + ', '.join(f'{k}={len(v)}' for k, v in sorted(
        streams.items())))
    stack_log = os.path.join(args.in_dir, 'stack.log')
    epoch0 = probe_epoch0(streams)
    log_eps = load_log_episodes(stack_log, epoch0) if epoch0 else []
    if log_eps:
        print(f'episodes: {len(log_eps)} from "detected collision ahead" log')
        eps = [dict(t0=e['t0'], t1=e['t1'], dur=e['t1'] - e['t0'],
                    v_before=float('nan'), coast=float('nan'),
                    source='log') for e in log_eps]
    else:
        eps = [dict(e, source='nav') for e in find_episodes(streams)]
        print(f'episodes (nav=0 while moving): {len(eps)}')

    rows = [ep_metrics(streams, e, sm, r_in=args.inscribed) for e in eps]
    if not rows:
        print('no episodes; nothing to analyse')
        return

    report = [f"# Collision-ahead episodes ({os.path.basename(os.path.abspath(args.in_dir))})",
              '',
              f"Episodes: {len(rows)}; total refusal time "
              f"{sum(r['dur'] for r in rows):.1f}s",
              '',
              '| t0 | dur | v_before | coast | arc | gap | blocked | on_cell | scan_gap | static | laser | inflated | phantom | pos |',
              '|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|']
    for r in rows:
        report.append(
            f"| {r['t0']:.1f} | {r['dur']:.1f} | {r['v_before']:.2f} | "
            f"{r['coast']:.2f} | {r['n_arc']} | {r['gap']:.2f} | "
            f"{r['blocked']} | {r['zero_gap']} | {r['scan_gap']:.2f} | "
            f"{r['stat']} | {r['laser']} | {r['inflated']} | {r['phantom']} | "
            f"({r['x']:.2f},{r['y']:.2f}) |")
    report.append('')
    blocked = [r for r in rows if r['n_arc'] and r['blocked'] > 0]
    clear = [r for r in rows if r['n_arc'] and r['blocked'] == 0]
    phantom = [r for r in blocked if r['phantom'] > 0]
    laser = [r for r in blocked if r['laser'] > 0 and r['phantom'] == 0]
    static = [r for r in blocked if r['stat'] > 0 and r['phantom'] == 0]
    report.append('## Reading the table')
    report.append('')
    report.append(f"- `gap` = closest distance from an arc pose to a lethal cell;")
    report.append(f"  with an inscribed footprint of {args.inscribed:.2f} m, RPP's")
    report.append(f"  footprint check trips when a lethal cell falls inside that")
    report.append(f"  radius. `blocked` = arc poses that are inside that radius.")
    report.append(f"- `on_cell` = arc poses that sit exactly on a lethal cell.")
    report.append(f"- `scan_gap` = closest laser return to any arc pose.")
    report.append(f"- grid cells blocking the arc within r_in: `static` supported")
    report.append(f"  by the static map, `laser` supported by a live return,")
    report.append(f"  `inflated` the 99 band around such a supported obstacle, and")
    report.append(f"  `phantom` none of the above (stale voxel marks: the shadow).")
    report.append('')
    report.append(f"- episodes with a footprint collision: {len(blocked)} / {len(rows)};")
    report.append(f"- of those: phantom-only {len(phantom)}, laser-only {len(laser)},")
    report.append(f"  static-only {len(static)}; arc clear yet refused: {len(clear)}.")
    report.append('')
    report.append('_The decisive case is an episode with `blocked > 0` and')
    report.append('`phantom > 0` (cells nobody currently supports) — that is the')
    report.append('"stuck in the inflation shadow" the user describes._')
    out_md = os.path.join(args.in_dir, 'collision_episodes.md')
    with open(out_md, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report) + '\n')
    print('\n'.join(report))
    print(f'wrote {out_md}')

    drawn = 0
    for r in rows:
        if drawn >= 4:
            break
        if not r['n_arc']:
            continue
        r['arc'] = nearest(streams, 'collision_arc', r['t0'] + 0.15, 1.0)
        cms = nearest(streams, 'costmap', r['t0'] + 0.15, 0.8)
        scan = nearest(streams, 'scan', r['t0'] + 0.1, 0.5)
        if not r['arc'] or not cms:
            continue
        r['costmap'] = cms['d']
        r['_fp'] = nearest(streams, 'footprint', r['t0'] + 0.15, 0.8)
        if scan:
            sd = scan['d']
            rng = decode_floats(sd)
            ang = sd['amin'] + np.arange(len(rng)) * sd['ainc']
            good = np.isfinite(rng) & (rng >= sd['rmin']) & (rng <= sd['rmax'])
            od = nearest(streams, 'odom', scan['t'], 0.4)
            if od:
                bx, by, yaw = od['d']['x'], od['d']['y'], od['d']['yaw']
                r['scan_pts'] = np.stack(
                    [bx + rng[good] * np.cos(yaw + ang[good]),
                     by + rng[good] * np.sin(yaw + ang[good])], axis=1)
        path = os.path.join(args.in_dir, f'episode_{drawn}.png')
        try:
            plot_episode(path, r, sm)
            print(f'wrote {path}')
            drawn += 1
        except Exception as exc:  # noqa: BLE001 - plotting is best effort
            print(f'plot failed: {exc}')


if __name__ == '__main__':
    sys.exit(main())
