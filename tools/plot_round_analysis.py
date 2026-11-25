#!/usr/bin/env python3
"""Figures for a 3-round batch: per-object split, dwell causes, movers, paths.

    tools/plot_round_analysis.py --dir runs/<rule1> --dir runs/<rule2> \
        --dir runs/<rule3> --out analysis

Writes PNGs into --out and prints the figures it wrote.  Labels are English on
purpose: the container has no CJK font, so Chinese glyphs would come out as
boxes; the Chinese wording belongs in the markdown report that embeds them.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dwell_report import attribute, find_dwells, load_log, load_track, step_timeline  # noqa: E402
from mover_proximity import TOUCH, load as load_mover  # noqa: E402

BUCKETS = [('nav', '#3b6ea5'), ('creep', '#7fb3d5'), ('vision', '#e0a33e'),
           ('arm', '#c0504d'), ('gripper_io', '#8c8c8c'),
           ('overhead', '#b0b0b0'), ('other', '#d9d9d9')]

ROUND_COLOURS = ['#3b6ea5', '#e0a33e', '#c0504d']


def load_objects(run_dir):
    source = run_dir / 'objects.json'
    if not source.exists():
        source = run_dir / 'analysis.json'
    try:
        return json.loads(source.read_text())
    except Exception:
        return []


def fig_objects(dirs, out):
    fig, axes = plt.subplots(1, len(dirs), figsize=(5.2 * len(dirs), 5.0),
                             sharey=False)
    if len(dirs) == 1:
        axes = [axes]
    for ax, run_dir in zip(axes, dirs):
        source = run_dir / 'objects.json'
        if not source.exists():
            source = run_dir / 'analysis.json'
        rows = json.loads(source.read_text())
        rows = [r for r in rows if r['delivered'] or r['total'] > 5]
        names = [f"{r['object']}\n->{r['zone']}" for r in rows]
        left = [0.0] * len(rows)
        for key, colour in BUCKETS:
            vals = [r.get(key, 0.0) for r in rows]
            ax.barh(names, vals, left=left, color=colour, label=key,
                    edgecolor='white', linewidth=0.5)
            left = [a + b for a, b in zip(left, vals)]
        for i, value in enumerate(left):
            ax.text(value + 0.6, i, f'{value:.0f}s', va='center', fontsize=8)
        ax.set_xlabel('seconds')
        ax.set_title(run_dir.name, fontsize=9)
        ax.invert_yaxis()
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(labels), fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle('Per-object time split (nav / creep / vision / arm)', fontsize=11)
    fig.tight_layout()
    path = out / 'fig1_per_object.png'
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def fig_dwell(dirs, out):
    """One bar per standstill: dock work (blue) vs mid-route stop (red)."""
    rows = []
    for run_dir in dirs:
        pts = load_track(run_dir)
        lines, objects = load_log(run_dir)
        if not pts or not lines:
            continue
        dwells = [d for d in find_dwells(pts)
                  if not (abs(d['x']) < 0.4 and abs(d['y']) < 0.4
                          and d['t0'] - pts[0][0] < 20.0)]
        last_done = [o['t'] for o in objects if o.get('done')]
        if last_done:
            dwells = [d for d in dwells if d['t0'] <= max(last_done)]
        steps = step_timeline(lines, objects)
        for dw in attribute(lines, dwells, objects, steps):
            rows.append((run_dir.name, dw))
    if not rows:
        return None
    fig, ax = plt.subplots(figsize=(9.5, 0.32 * len(rows) + 1.3))
    labels = []
    for i, (name, dw) in enumerate(rows):
        phase = dw.get('phase') or ''
        dock = phase.startswith(('arm:', 'vision:'))
        ax.barh(i, dw['dur'], color='#4C72B0' if dock else '#C44E52',
                height=0.72)
        ax.text(dw['dur'] + 0.25, i, f"{dw['dur']:.1f}s  {dw.get('top', '')}",
                va='center', fontsize=6.2)
        import re as _re
        m = _re.search(r'rule(\d)', name)
        short = f'r{m.group(1)}' if m else 'r?'
        labels.append(f"{short} {phase[:26]}")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(labels, fontsize=6.2)
    ax.invert_yaxis()
    ax.set_xlabel('standstill duration (s)')
    ax.set_title('Every standstill  (blue = dock work, red = mid-route / obstacle)',
                 fontsize=9)
    ax.grid(axis='x', alpha=0.3)
    ax.set_xlim(0, max(dw['dur'] for _n, dw in rows) * 1.7)
    fig.tight_layout()
    path = out / 'fig2_dwell.png'
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def fig_movers(dirs, out):
    fig, axes = plt.subplots(1, len(dirs), figsize=(5.2 * len(dirs), 3.8),
                             sharey=True)
    if len(dirs) == 1:
        axes = [axes]
    for ax, run_dir in zip(axes, dirs):
        base, movers = load_mover(run_dir / 'events.jsonl')
        if not base:
            continue
        for i, (name, samples) in enumerate(sorted(movers.items())):
            xs, ys = [], []
            for w, _t, mx, my in samples:
                best = min(base, key=lambda b: abs(b[0] - w))
                xs.append(best[1])
                ys.append(((best[2] - mx) ** 2 + (best[3] - my) ** 2) ** 0.5)
            colour = plt.cm.tab10(i % 10)
            ax.scatter(xs, ys, s=3, color=colour, label=name, alpha=0.8)
            low = [k for k, y in enumerate(ys) if y < 1.0]
            if low:
                k = min(low, key=lambda j: ys[j])
                ax.annotate(f'{name} {ys[k]:.2f} m', (xs[k], ys[k]),
                            textcoords='offset points', xytext=(4, -10),
                            fontsize=6, color=colour)
        ax.axhline(TOUCH, color='crimson', lw=0.9, ls='--',
                   label=f'shell touch {TOUCH:.2f} m')
        ax.axhline(0.42, color='darkorange', lw=0.8, ls=':', label='guard 0.42 m')
        ax.set_xlabel('round time (s)')
        ax.set_title(run_dir.name, fontsize=9)
        ax.set_ylim(0, 1.6)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel('base <-> mover centre distance (m)')
    axes[-1].legend(fontsize=6, ncol=2, loc='upper right')
    fig.suptitle('Dynamic obstacle clearance (only dips matter)', fontsize=11)
    fig.tight_layout()
    path = out / 'fig3_movers.png'
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def fig_paths(dirs, out):
    import math
    fig, axes = plt.subplots(1, len(dirs), figsize=(5.2 * len(dirs), 4.6))
    if len(dirs) == 1:
        axes = [axes]
    for ax, run_dir in zip(axes, dirs):
        pts = load_track(run_dir)
        dwells = find_dwells(pts)
        if pts:
            xs = [p[1] for p in pts]
            ys = [p[2] for p in pts]
            # Colour by speed so stops are visible as dark segments.
            speeds = []
            for i in range(len(pts)):
                a = pts[max(0, i - 2)]
                b = pts[min(len(pts) - 1, i + 2)]
                dt = b[0] - a[0]
                speeds.append(math.hypot(b[1] - a[1], b[2] - a[2]) / dt
                              if dt > 0 else 0.0)
            ax.scatter(xs, ys, c=speeds, s=4, cmap='viridis', vmin=0, vmax=1.0)
            for d in dwells:
                if d['dur'] >= 4:
                    ax.plot(d['x'], d['y'], 'o', ms=9, mfc='none',
                            mec='crimson', mew=1.4)
                    ax.annotate(f"{d['dur']:.0f}s", (d['x'], d['y']),
                                textcoords='offset points', xytext=(6, 4),
                                fontsize=7, color='crimson')
        ax.set_title(run_dir.name, fontsize=9)
        ax.set_aspect('equal', adjustable='datalim')
        ax.grid(alpha=0.25)
        ax.set_xlabel('x (m)')
    axes[0].set_ylabel('y (m)')
    fig.suptitle('Ground-truth path; rings mark dwells >= 4 s', fontsize=11)
    fig.tight_layout()
    path = out / 'fig4_paths.png'
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', action='append', required=True)
    parser.add_argument('--out', default='analysis')
    parser.add_argument('--skip', default='', help='comma list: objects,dwell,movers,paths')
    args = parser.parse_args()
    dirs = [Path(d) for d in args.dir]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    skip = {s.strip() for s in args.skip.split(',') if s.strip()}
    made = []
    for name, fn in (('objects', fig_objects), ('dwell', fig_dwell),
                     ('movers', fig_movers), ('paths', fig_paths)):
        if name in skip:
            continue
        try:
            made.append(fn(dirs, out))
        except Exception as exc:  # keep the other figures coming
            print(f'{name}: FAILED {type(exc).__name__}: {exc}')
    for path in made:
        print(f'wrote {path}')


if __name__ == '__main__':
    main()
