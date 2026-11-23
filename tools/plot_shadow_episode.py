#!/usr/bin/env python3
"""Plot one costmap-shadow episode with every lethal cell coloured by support.

Green lethal cells have a laser return within 0.15 m (real geometry). Black
ones sit on the static map (walls/furniture). Red ones have neither: they are
marks nothing in the current sensor view justifies, i.e. the "shadow" the base
refuses to drive into. The patrol boxes and their recent trail are drawn from
the round recorder's /gazebo/model_states so a red trail can be matched to the
box that left it behind.

    python3 tools/plot_shadow_episode.py --dir runs/<round> --t 57.7 \
        --out runs/<round>/shadow.png
"""
import argparse
import json
import math
import os

import numpy as np

from analyze_collision_episodes import (
    INFLATION_RADIUS, LETHAL, cell_centers, decode_costmap, decode_floats,
    load, load_log_episodes, probe_epoch0, static_support, load_static_map)


def load_events(path):
    rows = []
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def nearest(rows, t):
    return min(rows, key=lambda r: abs(r['t'] - t))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True)
    ap.add_argument('--t', type=float, default=None,
                    help='episode start, seconds in probe time; default first')
    ap.add_argument('--half', type=float, default=2.2, help='half window (m)')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    streams = load(os.path.join(args.dir, 'probe.jsonl'))
    epoch0 = probe_epoch0(streams)
    episodes = load_log_episodes(os.path.join(args.dir, 'stack.log'), epoch0)
    if not episodes:
        raise SystemExit('no RPP collision episodes in stack.log')
    ep = episodes[0]
    if args.t is not None:
        ep = min(episodes, key=lambda e: abs(e['t0'] - args.t))
    t0 = ep['t0'] + 0.5

    sm = load_static_map()
    od = nearest(streams['odom'], t0)['d']
    cm = nearest(streams['costmap'], t0)['d']
    arc = nearest(streams['collision_arc'], t0)['d']

    grid = decode_costmap(cm)
    xs, ys = cell_centers(cm)
    lethal = np.argwhere(grid >= LETHAL)
    lx = xs[lethal[:, 1]]
    ly = ys[lethal[:, 0]]

    scan = nearest(streams['scan'], t0)['d']
    rng = decode_floats(scan)
    ang = scan['amin'] + np.arange(len(rng)) * scan['ainc']
    ok = np.isfinite(rng) & (rng >= scan['rmin']) & (rng <= scan['rmax'])
    sx = od['x'] + rng[ok] * np.cos(od['yaw'] + ang[ok])
    sy = od['y'] + rng[ok] * np.sin(od['yaw'] + ang[ok])
    sp = np.stack([sx, sy], axis=1)

    costs = grid[lethal[:, 0], lethal[:, 1]] if len(lx) else np.zeros(0, int)
    if len(lx):
        d = np.linalg.norm(np.stack([lx, ly], axis=1)[:, None, :] - sp[None, :, :],
                           axis=2)
        laser = d.min(axis=1) <= 0.15
        stat = static_support(sm, lx, ly)
        marker = costs >= 100
        sup_marker = marker & (stat | laser)
        inflated = np.zeros(len(lx), dtype=bool)
        if sup_marker.any():
            dm = np.min(np.linalg.norm(
                np.stack([lx, ly], axis=1)[:, None, :]
                - np.stack([lx[sup_marker], ly[sup_marker]], axis=1)[None, :, :],
                axis=2), axis=1)
            inflated = (dm <= INFLATION_RADIUS) & ~laser & ~stat
    else:
        laser = np.zeros(0, dtype=bool)
        stat = np.zeros(0, dtype=bool)
        inflated = np.zeros(0, dtype=bool)
    phantom = ~laser & ~stat & ~inflated

    events = load_events(os.path.join(args.dir, 'events.jsonl'))
    msgs = [r for r in events if r['topic'] == 'model_states']
    now_models = nearest(msgs, t0 - 0.5)['d']
    trail = [r for r in msgs if t0 - 12.0 <= r['t'] <= t0 - 0.5]

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Polygon, Rectangle

    fig, ax = plt.subplots(figsize=(9.5, 8.5), dpi=120)
    half = args.half
    ax.set_xlim(od['x'] - half, od['x'] + half)
    ax.set_ylim(od['y'] - half, od['y'] + half)

    # costmap as a faint background (free/unknown/inflated)
    img = np.where(grid >= LETHAL, 10, np.where(grid >= 1, grid, 0))
    ax.imshow(img, origin='lower',
              extent=[xs[0] - cm['res'] / 2, xs[-1] + cm['res'] / 2,
                      ys[0] - cm['res'] / 2, ys[-1] + cm['res'] / 2],
              cmap='Greys', vmin=0, vmax=2, alpha=0.25, zorder=0)

    if len(lx):
        if stat.any():
            ax.scatter(lx[stat], ly[stat], s=26, c='k', marker='s',
                       label=f'lethal, static map ({int(stat.sum())})', zorder=2)
        if laser.any():
            ax.scatter(lx[laser], ly[laser], s=26, c='tab:green', marker='s',
                       label=f'lethal, laser ({int(laser.sum())})', zorder=3)
        if inflated.any():
            ax.scatter(lx[inflated], ly[inflated], s=26, c='salmon', marker='s',
                       label=f'lethal, legit inflation ({int(inflated.sum())})',
                       zorder=3)
        if phantom.any():
            ax.scatter(lx[phantom], ly[phantom], s=26, c='tab:red', marker='s',
                       label=f'lethal, stale shadow ({int(phantom.sum())})',
                       zorder=4)

    ax.scatter(sp[:, 0], sp[:, 1], s=4, c='tab:green', alpha=0.5,
               label='laser returns', zorder=1)
    pts = np.array(arc['pts'])
    ax.plot(pts[:, 0], pts[:, 1], '-', c='tab:blue', lw=2.2,
            label='RPP collision arc', zorder=5)

    for r in trail[::2]:
        m = r['d'].get('mover_2')
        if m:
            ax.plot(m[0], m[1], '.', c='tab:orange', ms=3, alpha=0.55,
                    zorder=6)
    for name, v in now_models.items():
        if name.startswith('mover'):
            ax.add_patch(Rectangle((v[0] - 0.25, v[1] - 0.25), 0.5, 0.5,
                                   fill=False, edgecolor='tab:orange', lw=1.4,
                                   zorder=6))
            ax.annotate(name, (v[0], v[1] + 0.3), color='tab:orange', fontsize=8)
    ax.plot([], [], '.', c='tab:orange', ms=3,
            label='mover_2 trail, last 12 s')

    ax.add_patch(Circle((od['x'], od['y']), 0.2, fill=False, edgecolor='k',
                        lw=1.2, zorder=7))
    ax.add_patch(Circle((od['x'], od['y']), 0.32, fill=False, ls='--',
                        edgecolor='k', lw=1.0, zorder=7,
                        label='inscribed radius 0.32 m'))
    ax.annotate('', xy=(od['x'] + 0.45 * math.cos(od['yaw']),
                        od['y'] + 0.45 * math.sin(od['yaw'])),
                xytext=(od['x'], od['y']),
                arrowprops=dict(arrowstyle='-|>', color='k', lw=1.4), zorder=7)

    ax.set_aspect('equal')
    ax.set_xlabel('x [m, odom]')
    ax.set_ylabel('y [m, odom]')
    ax.set_title(f'RPP refused at t={t0 - 0.5:.1f}s  pose=({od["x"]:.2f},'
                 f'{od["y"]:.2f})  phantom lethal={int(phantom.sum())}',
                 fontsize=11)
    ax.legend(loc='upper right', fontsize=8, framealpha=0.9)
    fig.tight_layout()
    out = args.out or os.path.join(args.dir, 'shadow.png')
    fig.savefig(out)
    print('wrote', out)
    print(f'phantom={int(phantom.sum())} laser={int(laser.sum())} '
          f'static={int(stat.sum())}')


if __name__ == '__main__':
    main()
