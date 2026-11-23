#!/usr/bin/env python3
"""Flatten several analysed rounds into one CSV of per-object timings.

    tools/timings_csv.py runs/2026-11-22_rule1 runs/2026-11-22_rule2 ... \
        --out runs/timings.csv
"""
import argparse
import csv
import json
import re
from pathlib import Path

RULE_RE = re.compile(r'映射规则 (\d) 生效')
SEED_RE = re.compile(r'使用种子[:：]\s*(\d+)')
STATE_RE = re.compile(r'State -> (COMPLETED|FAILED|ABORTED)')


def round_meta(d: Path):
    rule = seed = status = None
    prob = d / 'problem.txt'
    if prob.exists():
        m = SEED_RE.search(prob.read_text(encoding='utf-8', errors='replace'))
        if m:
            seed = int(m.group(1))
    log = d / 'stack.log'
    if log.exists():
        txt = log.read_text(encoding='utf-8', errors='replace')
        m = RULE_RE.search(txt)
        if m:
            rule = int(m.group(1))
        states = STATE_RE.findall(txt)
        if states:
            status = states[-1]
    if status is None:
        try:
            a = json.loads((d / 'analysis.json').read_text(encoding='utf-8'))
            status = 'COMPLETED' if len(a['objects']) == 5 else 'INCOMPLETE'
        except Exception:
            status = 'UNKNOWN'
    return rule, seed, status


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('rounds', nargs='+')
    ap.add_argument('--out', default='runs/timings.csv')
    args = ap.parse_args()

    rows = []
    for rd in args.rounds:
        d = Path(rd)
        a = json.loads((d / 'analysis.json').read_text(encoding='utf-8'))
        rule, seed, status = round_meta(d)
        for o in a['objects']:
            placed = any(t['label'].startswith('arm:Lower to place')
                         for t in o['timings'])
            gap = o['wall_s'] - (o['nav_s'] + o['vision_s'] + o['manip_s'])
            rows.append({
                'round': d.name,
                'rule': rule if rule is not None else '',
                'seed': seed if seed is not None else '',
                'object_id': o['object_id'],
                'cube': o['cube'],
                'zone': o['zone'],
                'wall_s': round(o['wall_s'], 2),
                'nav_s': round(o['nav_s'], 2),
                'vision_s': round(o['vision_s'], 2),
                'manip_s': round(o['manip_s'], 2),
                'gap_s': round(gap, 2),
                'placed': int(placed),
                'round_status': status,
            })

    out = Path(args.out)
    with out.open('w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f'wrote {out} ({len(rows)} rows)')


if __name__ == '__main__':
    main()
