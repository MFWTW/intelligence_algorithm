#!/usr/bin/env python3
"""Draw one competition round on the static map.

    tools/plot_round.py runs/<round> [--out <png>]

Shows: the occupancy grid, the driven odometry path (coloured per carried
object with direction arrows), cube start positions, the warehouse/zone poses
that are actually configured for this build, the moving-obstacle tracks, every
close encounter and every yield hold.  The PNG is written next to
analysis.json.
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', str(Path.home() / '.cache' / 'mpl'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import round_analysis as ra

MAP_YAML = 'src/yzbot/bot_navigation/maps/room_from_world.yaml'
EXEC_YAML = 'src/yzbot/mybot/config/profiles/executor_fast.yaml'


def read_pose_table(path, keys):
    """Pull `key:\n  - x\n  - y\n  - yaw` pairs out of the executor profile."""
    text = Path(path).read_text(encoding='utf-8')
    out = {}
    for key in keys:
        m = re.search(rf'^\s*{key}:\s*\n\s*-\s*([-\d.]+)\s*\n\s*-\s*([-\d.]+)'
                      rf'\s*(?:\n\s*-\s*([-\d.]+))?', text, re.M)
        if m:
            out[key] = [float(m.group(1)), float(m.group(2))]
    return out


def arrows(ax, x, y, every=25, colour='k'):
    for k in range(every, len(x), every):
        dx, dy = x[k] - x[k - 1], y[k] - y[k - 1]
        if np.hypot(dx, dy) < 1e-6:
            continue
        ax.arrow(x[k - 1], y[k - 1], dx * 0.9, dy * 0.9,
                 head_width=0.25, head_length=0.45, fc=colour, ec=colour,
                 alpha=0.75, length_includes_head=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('round_dir')
    ap.add_argument('--out')
    ap.add_argument('--map', default=None)
    ap.add_argument('--exec-config', default=None)
    args = ap.parse_args()

    rd = Path(args.round_dir)
    analysis = json.loads((rd / 'analysis.json').read_text())
    events = ra.load_events(str(rd / 'events.jsonl'))

    here = Path(__file__).resolve().parent.parent
    yaml_path = args.map or str(here / MAP_YAML)
    exec_path = args.exec_config or str(here / EXEC_YAML)
    occ, res, ox, oy, h = ra.read_map(yaml_path)
    w = occ.shape[1]

    poses = read_pose_table(exec_path, [
        'zone_a_centre', 'zone_b_centre', 'zone_c_centre',
        'warehouse_a_pose', 'warehouse_b_pose', 'warehouse_c_pose',
    ])

    odom = ra.odom_arrays(events)
    fit = analysis.get('map_fit', {})
    if 'odom_to_map_r' in fit:
        r = np.array(fit['odom_to_map_r'])
        tvec = np.array(fit['odom_to_map_t'])
    else:
        r, tvec, _ = ra.align_to_map(odom, events)
    yaw_off = ra.yaw_from_r(r)
    px = odom['x'] * r[0, 0] + odom['y'] * r[1, 0] + tvec[0]
    py = odom['x'] * r[0, 1] + odom['y'] * r[1, 1] + tvec[1]

    r2, t2, yaw2, _ = ra.world_to_map(events, odom, r, tvec)
    movers = ra.mover_tracks(events, r2, t2, yaw2)

    cubes = {}
    for _, d in ra.series(events, 'model_all'):
        cubes = d
        break
    if not cubes:
        for _, d in ra.series(events, 'model_states'):
            cubes = {k: v[:2] for k, v in d.items() if 'cube' in k}
            break

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(occ, cmap='gray_r', vmin=0, vmax=1,
              extent=[ox, ox + w * res, oy, oy + h * res], origin='upper',
              alpha=0.75)

    # --- moving obstacle tracks (world == map, verified per round)
    for name, tr in sorted(movers.items()):
        if not name.startswith('mover'):
            continue
        ax.plot(tr['x'], tr['y'], '-', lw=0.9, alpha=0.5, color='tab:orange')
        ax.annotate(name, (tr['x'][0], tr['y'][0]), fontsize=6,
                    color='darkorange')

    # --- cubes that were still on the floor when the round started
    for name, xy in sorted(cubes.items()):
        if 'cube' not in name:
            continue
        col = 'tab:red' if name.startswith('red') else 'tab:blue'
        ax.plot(xy[0], xy[1], 's', ms=6, color=col, alpha=0.85)
        ax.annotate(name.replace('_cube_', ''), xy, fontsize=6, color=col,
                    xytext=(3, 3), textcoords='offset points')

    for key, xy in poses.items():
        if 'zone' in key:
            ax.plot(xy[0], xy[1], 'k^', ms=9, mfc='0.85')
            ax.annotate(key.split('_')[1].upper(), xy, fontsize=9,
                        fontweight='bold', xytext=(5, -10),
                        textcoords='offset points')
        else:
            ax.plot(xy[0], xy[1], 'k*', ms=11)

    colours = plt.cm.tab10(np.linspace(0, 1, 10))
    for i, o in enumerate(analysis['objects']):
        m = (odom['t'] >= o['start_ts']) & (odom['t'] <= o['end_ts'])
        if m.sum() < 2:
            continue
        c = colours[i % 10]
        ax.plot(px[m], py[m], '-', lw=2.0, color=c,
                label=f"#{o['index']} {o['cube']}->{o['zone']}")
        ax.plot(px[m][0], py[m][0], 'o', ms=7, mfc='none', mec=c, mew=1.6)
        arrows(ax, px[m], py[m], every=max(12, int(0.12 * m.sum())), colour=c)
        ax.annotate(f"#{o['index']}", (px[m][-1], py[m][-1]), fontsize=9,
                    color=c, fontweight='bold', xytext=(5, 5),
                    textcoords='offset points')

    ax.plot(px[0], py[0], 'k*', ms=16, label='start (map frame origin)')

    for e in analysis['encounters']:
        k = min(int(np.searchsorted(odom['t'], e['min_t'])), len(px) - 1)
        ax.plot(px[k], py[k], 'o', ms=14, mfc='none', mec='crimson', mew=1.8)
        ax.annotate(f"{e['mover']} {e['min_dist']:.2f} m", (px[k], py[k]),
                    fontsize=7, color='crimson', xytext=(6, 6),
                    textcoords='offset points')

    holds = [y for y in analysis['yields'] if y['action'] == 'stop']
    for i, y in enumerate(holds):
        k = min(int(np.searchsorted(odom['t'], y['ts'])), len(px) - 1)
        ax.plot(px[k], py[k], 'x', ms=4, color='red', alpha=0.55,
                label='yield stop' if i == 0 else None)

    rows = np.flatnonzero(occ.any(axis=1))
    cols = np.flatnonzero(occ.any(axis=0))
    ax.set_xlim(ox + cols[0] * res - 1.5, ox + cols[-1] * res + 1.5)
    ax.set_ylim(oy + (h - 1 - rows[-1]) * res - 1.5,
                oy + (h - 1 - rows[0]) * res + 1.5)
    ax.set_aspect('equal')
    ax.set_xlabel('map x [m]')
    ax.set_ylabel('map y [m]')
    ax.set_title(f"{analysis['round']}: driven route, zones and obstacles")
    ax.legend(loc='upper left', fontsize=8, framealpha=0.9)
    ax.grid(alpha=0.2, ls=':')

    out = args.out or str(rd / 'route.png')
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print(out)


if __name__ == '__main__':
    main()
