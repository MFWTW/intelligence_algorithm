#!/usr/bin/env python3
"""One markdown block per round: completion, per-object timings, nav health.

Groups the three things the task asked for (navigation time, vision time,
grasp/place time) per object, and adds the navigation/obstacle figures the
round produced: controller refusals ("detected collision ahead"), the episode
table written by analyze_collision_episodes.py, and the failure summary.

    python3 tools/round_report.py runs/2026-11-23_map1 runs/2026-11-23_map2
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import per_object_timings as timings

COLLISION_RE = re.compile(r'detected collision ahead')
ESCAPE_RE = re.compile(r'leaving the band at the start pose')
ESCAPED_RE = re.compile(r'leaving the band at the start pose: guarded retreat measured '
                        r'[0-9.]+ m; stop: left planner band')
FAILED_RE = re.compile(r'FAILED SUMMARY: (.*)$', re.M)
EPISODES_RE = re.compile(r'^Episodes: (\d+); total refusal time ([0-9.]+)s', re.M)
ROW_RE = re.compile(r'^\| (?:[0-9]+\.?[0-9]*) \|', re.M)


def summary_line(log: Path) -> str:
    text = log.read_text(errors='ignore')
    failed = FAILED_RE.search(text)
    done = 'State -> COMPLETED' in text
    if failed:
        return f'FAILED — {failed.group(1).strip()}'
    return 'COMPLETED' if done else 'no terminal state in stack.log'


def round_stats(run: Path) -> dict:
    log = run / 'stack.log'
    text = log.read_text(errors='ignore') if log.exists() else ''
    stats = {
        'collision_lines': len(COLLISION_RE.findall(text)),
        'episodes': None,
        'refusal_s': None,
        'phantom_cells': None,
        'band_escapes': len(ESCAPE_RE.findall(text)),
        'band_escaped': len(ESCAPED_RE.findall(text)),
    }
    episodes_md = run / 'collision_episodes.md'
    if episodes_md.exists():
        md = episodes_md.read_text(errors='ignore')
        found = EPISODES_RE.search(md)
        if found:
            stats['episodes'] = int(found.group(1))
            stats['refusal_s'] = float(found.group(2))
        phantom = 0
        for line in md.splitlines():
            cells = [c.strip() for c in line.strip().strip('|').split('|')]
            if len(cells) >= 14 and line.startswith('|'):
                try:
                    phantom += int(cells[12])
                except ValueError:
                    continue
        stats['phantom_cells'] = phantom
    return stats


def report(run: Path) -> str:
    objects = timings.parse(run / 'stack.log')
    rows = []
    totals = {k: 0.0 for k in ('total', 'nav', 'vision', 'grasp')}
    for obj in objects:
        summary = timings.summarise(obj)
        buckets = summary['buckets']
        row = {
            'object': obj['object'],
            'zone': obj['zone'],
            'delivered': obj['done'] is not None,
            'total': summary['total'],
            # The three numbers the operator asked for: driving (nav + the
            # slow creep approach), the visual alignment, and grasp/place
            # (arm motion + gripper service calls).
            'nav': buckets['nav'] + buckets['creep'],
            'vision': buckets['vision'],
            'grasp': buckets['arm'] + buckets['gripper_io'],
        }
        rows.append(row)
        for key in totals:
            totals[key] += row[key]
    stats = round_stats(run)
    delivered = sum(1 for row in rows if row['delivered'])
    lines = [f'## {run.name}', '', summary_line(run / 'stack.log'), '']
    lines.append(f'- objects finished: **{delivered}/{len(rows)}**')
    lines.append(f'- controller refusals: {stats["collision_lines"]}')
    lines.append(f'- band escapes at leg start: {stats["band_escaped"]}/'
                 f'{stats["band_escapes"]} cleared')
    if stats['episodes'] is not None:
        lines.append(f'- refusal episodes: {stats["episodes"]} '
                     f'({stats["refusal_s"]:.1f} s), phantom cells '
                     f'{stats["phantom_cells"]}')
    lines += ['', '| object | zone | result | total | nav | vision | grasp/place |',
              '|---|---|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append(
            f"| {row['object']} | {row['zone']} | "
            f"{'delivered' if row['delivered'] else 'FAILED'} | "
            f"{row['total']:.1f} | {row['nav']:.1f} | {row['vision']:.1f} | "
            f"{row['grasp']:.1f} |")
    lines.append(
        f"| **total** | | | **{totals['total']:.1f}** | **{totals['nav']:.1f}** "
        f"| **{totals['vision']:.1f}** | **{totals['grasp']:.1f}** |")
    lines.append('')
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('runs', nargs='+')
    parser.add_argument('--out', help='write the markdown here as well')
    args = parser.parse_args()
    blocks = [report(Path(run)) for run in args.runs]
    text = '\n'.join(blocks)
    print(text)
    if args.out:
        Path(args.out).write_text(text)
        print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
