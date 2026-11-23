#!/usr/bin/env python3
"""Timeline: speed, heading and every yield decision of one round.

    tools/plot_timeline.py runs/2026-11-21_rule1 [--out .../timeline.png]

Panel 1: linear speed (m/s) with the per-object spans shaded.
Panel 2: heading in degrees (0 = +x / east, 90 = +y / north).
Panel 3: yield decisions, one marker per YIELD line, coloured by action and
         placed on the reason lane so the decision sequence is readable.
"""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', os.path.expanduser('~/.cache/mpl'))
import matplotlib                                        # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt                          # noqa: E402
import numpy as np                                       # noqa: E402
import sys                                               # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
import round_analysis as ra                              # noqa: E402

plt.rcParams['font.family'] = ['Noto Sans CJK JP', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

ACTION_COLOR = {'go': '#2ca02c', 'cap': '#ff7f0e', 'arc': '#1f77b4',
                'stop': '#d62728'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('round_dir')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    rd = Path(args.round_dir)
    analysis = json.loads((rd / 'analysis.json').read_text())
    objs = analysis['objects']
    t00 = objs[0]['start_ts']

    events = ra.load_events(rd / 'events.jsonl')
    odom = ra.odom_arrays(events)
    ts = odom['t'] - t00
    step = np.hypot(np.diff(odom['x']), np.diff(odom['y']))
    dt = np.maximum(np.diff(odom['t']), 1e-3)
    spd = np.r_[0.0, step / dt]
    yaw = np.rad2deg(np.unwrap(odom['yaw']))

    binw = 0.5
    edges = np.arange(ts[0], ts[-1] + binw, binw)
    idx = np.digitize(ts, edges) - 1
    keep = (idx >= 0) & (idx < len(edges) - 1)
    nb = len(edges) - 1
    cnt = np.bincount(idx[keep], minlength=nb)
    sum_s = np.bincount(idx[keep], weights=spd[keep], minlength=nb)
    sum_y = np.bincount(idx[keep], weights=yaw[keep], minlength=nb)
    good = cnt > 0
    ts_b = 0.5 * (edges[:-1] + edges[1:])[good]
    spd_b = sum_s[good] / cnt[good]
    yaw_b = sum_y[good] / cnt[good]

    fig, axes = plt.subplots(3, 1, figsize=(11, 7.2), sharex=True,
                             gridspec_kw={'height_ratios': [2, 2, 2.6]})
    ax0, ax1, ax2 = axes

    for i, o in enumerate(objs):
        a, b = o['start_ts'] - t00, (o['end_ts'] or o['start_ts']) - t00
        for ax in axes:
            ax.axvspan(a, b, color='#1f77b4' if i % 2 else '#8c8c8c',
                       alpha=0.06, lw=0)

    ax0.plot(ts_b, spd_b, lw=0.9, color='#1f77b4')
    ax0.set_ylabel('线速度 m/s')
    ax0.grid(alpha=0.25)
    ax0.set_ylim(0, max(0.4, float(spd_b.max()) * 1.1))

    ax1.plot(ts_b, yaw_b, lw=0.9, color='#7f4f9b')
    ax1.set_ylabel('航向 °（0=东 90=北）')
    ax1.set_yticks([-180, -90, 0, 90, 180, 270, 360])
    ax1.set_yticklabels(['180', '270', '0', '90', '180', '270', '360'])
    ax1.grid(alpha=0.25)

    reasons = sorted({y['reason'] for y in analysis['yields']})
    lane = {r: i for i, r in enumerate(reasons)}
    for y in analysis['yields']:
        if y.get('detail') == 'diag':
            continue
        x = y['ts'] - t00
        ax2.scatter([x], [lane[y['reason']]], s=26,
                    color=ACTION_COLOR.get(y['action'], 'gray'),
                    marker='o' if y['action'] != 'stop' else 'X', zorder=3)
    ax2.set_yticks(range(len(reasons)))
    ax2.set_yticklabels(reasons, fontsize=8)
    ax2.set_ylabel('避障决策原因')
    ax2.set_xlabel('时间 s（第一块物块开始时为 0）')
    ax2.grid(alpha=0.25, axis='x')

    for o in objs:
        a, b = o['start_ts'] - t00, (o['end_ts'] or o['start_ts']) - t00
        ax0.text((a + b) / 2, ax0.get_ylim()[1] * 0.92,
                 f"#{o['index']} {o['cube']}→{o['zone']}",
                 ha='center', va='top', fontsize=8,
                 bbox=dict(fc='white', ec='none', alpha=0.75, pad=1.2))

    handles = [plt.Line2D([], [], color=c, marker='o', ls='', label=a)
               for a, c in ACTION_COLOR.items()]
    ax2.legend(handles=handles, ncol=4, fontsize=8, loc='upper right',
               title='YIELD 动作', title_fontsize=8)

    fig.suptitle(f'{rd.name}: speed / heading / yield decisions')
    fig.tight_layout()
    out = Path(args.out) if args.out else rd / 'timeline.png'
    fig.savefig(out, dpi=150)
    print(out)


if __name__ == '__main__':
    main()
