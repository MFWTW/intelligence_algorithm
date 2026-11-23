#!/usr/bin/env python3
"""Replay a recorded round through the yield policy, old rules vs prediction.

The unit tests show the new planner reacts correctly to hand-built observations.
This replays the observations the robot actually saw: base pose from
/gazebo/link_states, mover pose from /gazebo/model_states, base velocity from
/odom. Each frame becomes a YieldObservation in the base frame, and decide() is
run twice - once with the prediction enabled, once with the old distance rules -
so the two can be compared on the same geometry.

Lane fields are not reconstructed (the patrol segments are not in the recording),
so the lane-end distrust and lane escape paths are inert in both runs. What this
does verify is the part the 2026-11-21 review was about: when does each policy
first refuse to keep driving, and what would it have done at the moment the old
policy stopped.

    python3 tools/replay_yield.py runs/2026-11-21_r53s_rule2/events.jsonl \
        --mover mover_1 --t0 30 --t1 40
"""
import argparse
import json
import math
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src/yzbot/mybot/scripts'))

import obstacle_yield as oy  # noqa: E402


def _load(path):
    base, boxes, odom = [], [], []
    for line in path.open():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        topic = event.get('topic')
        data = event.get('d') or {}
        t = event.get('t')
        if t is None:
            continue
        if topic == 'link_states':
            base.append((t, data.get('x'), data.get('y'), data.get('yaw')))
        elif topic == 'model_states':
            boxes.append((t, {k: v for k, v in data.items()
                              if k.startswith('mover')}))
        elif topic == 'odom':
            odom.append((t, data.get('vx', 0.0), data.get('wz', 0.0)))
    return base, boxes, odom


def _before(samples, t):
    best = None
    for sample in samples:
        if sample[0] <= t:
            best = sample
        else:
            break
    return best


def _velocity(samples, t, name, window=0.8):
    """Median of pairwise velocities around t.

    A single finite difference over the window is dragged around by the patrol
    reversal, where the recorded pose jumps direction between two samples. The
    median of the pairwise velocities ignores those outlier pairs.
    """
    near = [s for s in samples if abs(s[0] - t) <= window and name in s[1]]
    vx, vy = [], []
    for first, last in zip(near, near[1:]):
        dt = last[0] - first[0]
        if dt <= 1e-6:
            continue
        dx = last[1][name][0] - first[1][name][0]
        dy = last[1][name][1] - first[1][name][1]
        if math.hypot(dx, dy) > 0.6:
            continue
        vx.append(dx / dt)
        vy.append(dy / dt)
    if not vx:
        return 0.0, 0.0
    vx.sort()
    vy.sort()
    mid = len(vx) // 2
    return vx[mid], vy[mid]


def observations(events, mover):
    base, boxes, odom = _load(events)
    for t, bx, by, yaw in base:
        if bx is None or yaw is None:
            continue
        sample = _before(boxes, t)
        if sample is None or t - sample[0] > 0.25 or mover not in sample[1]:
            continue
        mx, my, myaw = sample[1][mover]
        dx, dy = mx - bx, my - by
        box_x = math.cos(yaw) * dx + math.sin(yaw) * dy
        box_y = -math.sin(yaw) * dx + math.cos(yaw) * dy
        wx, wy = _velocity(boxes, t, mover)
        motion = _before(odom, t)
        vx, wz = (motion[1], motion[2]) if motion else (0.0, 0.0)
        rel_vx = (math.cos(yaw) * wx + math.sin(yaw) * wy - vx
                  + wz * box_y)
        rel_vy = (-math.sin(yaw) * wx + math.cos(yaw) * wy - wz * box_x)
        yield t, oy.YieldObservation(
            box_x=box_x, box_y=box_y, box_vx=rel_vx, box_vy=rel_vy,
            vx=vx, wz=wz, scan_age_sec=0.1,
            box_world_speed_mps=math.hypot(wx, wy))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('events', type=pathlib.Path)
    parser.add_argument('--mover', default='mover_1')
    parser.add_argument('--t0', type=float, default=0.0)
    parser.add_argument('--t1', type=float, default=1e9)
    args = parser.parse_args(argv)

    new = oy.YieldConfig()
    old = new.with_overrides(predict_enabled=False)
    rows = [(t, obs, oy.decide(obs, old), oy.decide(obs, new))
            for t, obs in observations(args.events, args.mover)
            if args.t0 <= t <= args.t1]
    if not rows:
        print('no frames in the window')
        return 0
    t0 = rows[0][0]
    print(f'{len(rows)} frames with {args.mover} in range, '
          f't={rows[0][0]:.1f}..{rows[-1][0]:.1f}')
    print('| t (s) | range (m) | old rules | with prediction |')
    print('|---|---|---|---|')
    shown, last = 0, None
    for t, obs, before, after in rows:
        key = (before.action, before.reason, after.action, after.reason)
        if key == last:
            continue
        last = key
        print(f'| {t - t0:.1f} | {obs.range_m():.2f} | '
              f'{before.action}/{before.reason} | '
              f'{after.action}/{after.reason} |')
        shown += 1
    print(f'{shown} decision changes')


if __name__ == '__main__':
    sys.exit(main())
