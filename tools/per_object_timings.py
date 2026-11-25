#!/usr/bin/env python3
"""Break one round's stack.log down per object: nav, vision, arm, overhead.

Every step the executor measures is printed as
    [fixed_joint_pick_place]: TIMING <label> <seconds>
and the object a step belongs to is bracketed by
    Started <object> -> <zone> attempt N.
    DELIVERY COMPLETED: <object> placed in warehouse <zone>.
The arm steps carry no object name, so the bracket is what attributes them.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

TIMING_RE = re.compile(
    r'\[fixed_joint_pick_place\]: TIMING (\S+(?: \S+)*?) ([0-9.]+)$')
START_RE = re.compile(r'\[task_state_machine\]: Started (\S+) -> (\S+) attempt (\d+)')
DONE_RE = re.compile(r'DELIVERY COMPLETED: (\S+) placed in warehouse (\S+)')

CATEGORIES = {
    'nav': ('nav:',),
    'creep': ('creep:',),
    'vision': ('vision_align', 'creep_to_grasp'),
    'arm': ('arm:',),
    'gripper_io': ('svc:attach', 'svc:detach'),
    'overhead': ('svc:', 'deps:'),
}


def category(label: str) -> str:
    for name, prefixes in CATEGORIES.items():
        if name == 'vision' and label in prefixes:
            return 'vision'
        if name != 'vision' and label.startswith(prefixes):
            return name
    return 'other'


def parse(path: Path):
    objects = []
    current = None
    with path.open(encoding='utf-8', errors='replace') as handle:
        for line in handle:
            m = START_RE.search(line)
            if m:
                current = {'object': m.group(1), 'zone': m.group(2),
                           'attempt': int(m.group(3)), 'steps': [], 'done': None}
                objects.append(current)
                continue
            d = DONE_RE.search(line)
            if d and current and current['object'] == d.group(1):
                current['done'] = d.group(2)
                continue
            t = TIMING_RE.search(line)
            if t and current is not None:
                label = t.group(1).strip()
                current['steps'].append((label, float(t.group(2))))
    return objects


def summarise(obj) -> dict:
    buckets = {name: 0.0 for name in CATEGORIES}
    buckets['other'] = 0.0
    labels = {}
    for label, seconds in obj['steps']:
        cat = category(label)
        buckets[cat] += seconds
        labels[label] = labels.get(label, 0.0) + seconds
    total = sum(buckets.values())
    return {'buckets': buckets, 'total': total, 'labels': labels}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', required=True, help='round directory or stack.log')
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args()
    path = Path(args.dir)
    if path.is_dir():
        path = path / 'stack.log'
    objects = parse(path)
    rows = []
    for obj in objects:
        summary = summarise(obj)
        row = {
            'object': obj['object'],
            'zone': obj['zone'],
            'attempt': obj['attempt'],
            'delivered': obj['done'] is not None,
            'total': round(summary['total'], 2),
            **{k: round(v, 2) for k, v in summary['buckets'].items()},
        }
        rows.append((row, summary['labels']))
    if args.json:
        print(json.dumps([r for r, _ in rows], indent=2))
        return
    header = (f"{'object':<12}{'zone':<5}{'result':<9}{'total':>7}{'nav':>7}"
              f"{'creep':>7}{'vision':>8}{'arm':>7}{'grip':>6}{'setup':>7}"
              f"{'other':>7}")
    print(header)
    print('-' * len(header))
    totals = {k: 0.0 for k in ('total', 'nav', 'creep', 'vision', 'arm',
                               'gripper_io', 'overhead', 'other')}
    for row, labels in rows:
        print(f"{row['object']:<12}{row['zone']:<5}"
              f"{'delivered' if row['delivered'] else 'FAILED':<9}"
              f"{row['total']:>7.1f}{row['nav']:>7.1f}{row['creep']:>7.1f}"
              f"{row['vision']:>8.1f}{row['arm']:>7.1f}"
              f"{row['gripper_io']:>6.1f}{row['overhead']:>7.1f}"
              f"{row['other']:>7.1f}")
        for key in totals:
            totals[key] += row[key]
        print('   steps: ' + ', '.join(
            f'{label} {seconds:.2f}' for label, seconds in labels.items()))
    print('-' * len(header))
    print(f"{'TOTAL':<26}{totals['total']:>7.1f}{totals['nav']:>7.1f}"
          f"{totals['creep']:>7.1f}{totals['vision']:>8.1f}{totals['arm']:>7.1f}"
          f"{totals['gripper_io']:>6.1f}{totals['overhead']:>7.1f}"
          f"{totals['other']:>7.1f}")


if __name__ == '__main__':
    main()
