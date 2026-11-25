#!/usr/bin/env python3
"""Collect watcher stall reports across rounds into one table.

`tools/watcher.py` writes `<run>/watcher/stall_NN.md` whenever the base stands
still for longer than the stall threshold. Those files are the evidence for
"where exactly did the car lose time, and what was it waiting for"; this script
turns them into one CSV plus a markdown table so the rounds can be compared.

    python3 tools/stall_table.py runs/2026-11-25_1335_rule1_watch [more runs]
    python3 tools/stall_table.py --csv analysis/stalls.csv runs/*
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

FIELD = {
    'wall time': 'at',
    'state': 'state',
    'base pose': 'pose',
    'stood still for': 'still_s',
    'displacement in window': 'disp',
    'mean odom speed': 'speed',
    'verdict': 'verdict',
    'dominant log reasons': 'reasons',
    'last goal': 'goal',
}
POSE_RE = re.compile(r'\((-?\d+\.\d+),\s*(-?\d+\.\d+)\)')
NUM_RE = re.compile(r'(-?\d+(?:\.\d+)?)')


def parse_stall(path):
    row = {'file': path.name, 'run': path.parent.parent.name}
    text = path.read_text(encoding='utf-8', errors='replace')
    for line in text.splitlines():
        if not line.startswith('- '):
            continue
        key, _, value = line[2:].partition(':')
        key = key.strip().lower()
        value = value.strip()
        field = FIELD.get(key)
        if field is None:
            continue
        if field == 'pose':
            match = POSE_RE.search(value)
            if match:
                row['x'] = float(match.group(1))
                row['y'] = float(match.group(2))
        elif field in {'still_s', 'disp', 'speed'}:
            match = NUM_RE.search(value)
            if match:
                row[field] = float(match.group(1))
        elif field == 'verdict':
            row['verdict'] = 'JAM' if value.upper().startswith('JAM') else 'WAIT'
        else:
            row[field] = value
    row.setdefault('x', float('nan'))
    row.setdefault('y', float('nan'))
    row.setdefault('still_s', float('nan'))
    row.setdefault('disp', float('nan'))
    row.setdefault('speed', float('nan'))
    row.setdefault('verdict', '?')
    row.setdefault('reasons', '')
    row.setdefault('state', '')
    row.setdefault('at', '')
    row.setdefault('goal', '')
    return row


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('runs', nargs='+')
    parser.add_argument('--csv', default='')
    args = parser.parse_args(argv)

    rows = []
    for run in args.runs:
        run_dir = Path(run)
        if not run_dir.is_dir():
            continue
        for path in sorted((run_dir / 'watcher').glob('stall_*.md')):
            rows.append(parse_stall(path))
    rows.sort(key=lambda r: (r['run'], r['file']))

    columns = ['run', 'file', 'at', 'state', 'x', 'y', 'still_s', 'disp',
               'speed', 'verdict', 'reasons', 'goal']
    if args.csv:
        Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, 'w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            for row in rows:
                writer.writerow({k: row.get(k, '') for k in columns})
        print(f'wrote {args.csv} ({len(rows)} stalls)')

    print()
    print('| run | at | state | pose | still | verdict | why |')
    print('|---|---|---|---|---|---|---|')
    for row in rows:
        pose = f'({row["x"]:.2f}, {row["y"]:.2f})'
        print(f'| {row["run"]} | {row["at"]} | {row["state"]} | {pose} | '
              f'{row["still_s"]:.1f} s | {row["verdict"]} | {row["reasons"]} |')
    if not rows:
        print('| - | - | - | - | - | - | no stall reports |')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
