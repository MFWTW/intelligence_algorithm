#!/usr/bin/env python3
"""Stacked timing chart: navigation / vision / manipulation per object.

    tools/plot_timings.py runs/2026-11-21_rule1 runs/... --out runs/timings.png
"""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', os.path.expanduser('~/.cache/mpl'))
import matplotlib                                        # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt                          # noqa: E402
plt.rcParams['font.family'] = ['Noto Sans CJK JP', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

COLORS = {'nav': '#1f77b4', 'vision': '#ff7f0e', 'manip': '#2ca02c',
          'other': '#9e9e9e'}
LABELS = {'nav': '导航 nav', 'vision': '视觉调整 vision',
          'manip': '拿放 arm', 'other': '空隙/其他'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('rounds', nargs='+')
    ap.add_argument('--out', default='runs/timings.png')
    args = ap.parse_args()

    rounds = []
    for r in args.rounds:
        a = json.loads((Path(r) / 'analysis.json').read_text())
        rule = None
        pf = Path(r) / 'rule.txt'
        if pf.exists():
            for ln in pf.read_text(errors='replace').splitlines():
                if 'mapping_rule:' in ln:
                    rule = ln.split('mapping_rule:')[1].strip()
        rounds.append((Path(r).name, rule, a))

    rows = [(ri, name, rule, o) for ri, (name, rule, a) in enumerate(rounds, 1)
            for o in a['objects']]
    n = len(rows)
    fig, ax = plt.subplots(figsize=(max(9.0, 0.62 * n + 3.0), 4.8))
    xs = range(n)
    bottom = [0.0] * n
    for key in ('nav', 'vision', 'manip', 'other'):
        vals = []
        for _, _, _, o in rows:
            if key == 'nav':
                v = o['nav_s']
            elif key == 'vision':
                v = o['vision_s']
            elif key == 'manip':
                v = o['manip_s']
            else:
                v = max(0.0, o['wall_s'] - o['nav_s'] - o['vision_s'] -
                        o['manip_s'])
            vals.append(v)
        ax.bar(xs, vals, bottom=bottom, color=COLORS[key],
               label=LABELS[key], width=0.72, edgecolor='white', linewidth=0.4)
        bottom = [b + v for b, v in zip(bottom, vals)]

    for i, (ri, name, rule, o) in enumerate(rows):
        ax.text(i, bottom[i] + 0.8, f'{o["wall_s"]:.0f}', ha='center',
                va='bottom', fontsize=7.5)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([f'R{ri}#{o["index"]}\n'
                        f'{o["cube"].replace("_cube_", "")}'
                        for ri, name, r, o in rows], fontsize=7.5,
                       rotation=45, ha='right')
    ax.set_ylabel('秒')
    ax.set_title('每块物块耗时分解（导航 / 视觉调整 / 拿放 / 空隙）')
    ax.legend(loc='upper left', fontsize=8, framealpha=0.9)
    ax.grid(axis='y', alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout()
    out = Path(args.out)
    fig.savefig(out, dpi=150)
    print(out, f'{n} bars')


if __name__ == '__main__':
    main()
