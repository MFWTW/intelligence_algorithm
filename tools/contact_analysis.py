#!/usr/bin/env python3
"""Measure Gazebo contact, shell clearance and push-while-stopped for one round.

The round recorder already logs the ground truth (`/gazebo/model_states` and
`/gazebo/link_states`, ~5 Hz) next to the commanded velocity. That is enough to
answer the three questions the 2026-11-21 review asked for, without trusting any
number the report script wrote by hand:

* contact count - frames where the physical hull overlaps a mover box;
* minimum shell clearance - the smallest SAT gap over the whole round;
* pushed while stopped - the base was commanded zero velocity and its Gazebo
  pose still moved, i.e. a stop order did not stop it.

Two shells are reported, because they answer different questions:

* `hull` is what the Gazebo collision shapes actually are: chassis 0.32 m square
  (large_base.xacro base_length/base_width), wheels 0.05 m wide at
  y=+-0.185 and radius 0.065 at x=+-0.14, so the box is 0.41 m x 0.37 m. Only
  a hull overlap means the round touched something.
* `policy` is the shell the yield policy plans with (base_half_length 0.25,
  base_half_width 0.35, i.e. 0.50 m x 0.70 m). Its margin is what stops an
  approach before the hull does, so it is the number to watch when asking
  whether a stop was comfortable, not the number that decides contact.

Clearance is the largest gap along the four face normals (the same conservative
measure the predicted-occupancy planner uses): 0.0 is touching, negative is
penetration.

Usage:
    python3 tools/contact_analysis.py runs/<round>/events.jsonl [-o out.json]
"""
import argparse
import json
import math
import pathlib
import sys

HULL_HALF_LENGTH = 0.205     # wheel outer edge: 0.14 + 0.065
HULL_HALF_WIDTH = 0.185      # wheel outer face: (0.32 + 0.05) / 2
POLICY_HALF_LENGTH = 0.25    # obstacle_yield.YieldConfig.base_half_length_m
POLICY_HALF_WIDTH = 0.35     # obstacle_yield.YieldConfig.base_half_width_m
MOVER_HALF = 0.11            # moving_obstacles box: 0.22 m square
# A command below this is "stopped" for the push test. RPP reports exactly 0.0
# when it refuses to move, and the executor publishes a zero Twist on a hold.
STOP_CMD_MPS = 0.02
STOP_CMD_RAD_S = 0.05
# Movement of the base while stopped that counts as being pushed.
PUSH_MIN_M = 0.02
# How far apart a model_states and a link_states sample may be to be compared.
PAIR_WINDOW_S = 0.25


def _corners(x, y, yaw, half_length, half_width):
    c, s = math.cos(yaw), math.sin(yaw)
    out = []
    for dx, dy in ((half_length, half_width), (half_length, -half_width),
                   (-half_length, -half_width), (-half_length, half_width)):
        out.append((x + c * dx - s * dy, y + s * dx + c * dy))
    return out


def sat_gap(base, mover):
    """Largest gap along the separating axes of two oriented boxes.

    Returns 0.0 for a touch and a negative value for overlap, which is exactly
    the convention the planner's _box_clearance uses.
    """
    best = -math.inf
    for ax, ay in _axes(base) + _axes(mover):
        b_proj = [px * ax + py * ay for px, py in base]
        m_proj = [px * ax + py * ay for px, py in mover]
        gap = max(min(m_proj) - max(b_proj), min(b_proj) - max(m_proj))
        best = max(best, gap)
    return best


def _axes(corners):
    axes = []
    for (x0, y0), (x1, y1) in zip(corners, corners[1:] + corners[:1]):
        dx, dy = x1 - x0, y1 - y0
        length = math.hypot(dx, dy)
        if length > 1e-9:
            axes.append((-dy / length, dx / length))
    return axes


def _read(path):
    models, base, cmd = [], [], []
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
        if topic == 'model_states':
            models.append((t, data))
        elif topic == 'link_states':
            base.append((t, data.get('x'), data.get('y'), data.get('yaw')))
        elif topic == 'cmd_vel':
            cmd.append((t, data.get('vx', 0.0), data.get('wz', 0.0)))
    return models, base, cmd


def _nearest_before(samples, t):
    best = None
    for sample in samples:
        if sample[0] <= t:
            best = sample
        else:
            break
    return best


def analyze(path, hull=(HULL_HALF_LENGTH, HULL_HALF_WIDTH)):
    models, base, cmd = _read(path)
    movers = sorted({name for _, data in models for name in data
                     if name.startswith('mover')})
    contacts = []
    per_mover_min = {name: math.inf for name in movers}
    per_mover_policy = {name: math.inf for name in movers}
    min_hull = math.inf
    min_hull_at = None
    min_hull_mover = None
    min_policy = math.inf
    min_policy_at = None
    min_policy_mover = None
    seen = 0
    for t, x, y, yaw in base:
        if x is None or y is None or yaw is None:
            continue
        sample = _nearest_before(models, t)
        if sample is None or t - sample[0] > PAIR_WINDOW_S:
            continue
        seen += 1
        hull_box = _corners(x, y, yaw, *hull)
        policy_box = _corners(x, y, yaw, POLICY_HALF_LENGTH, POLICY_HALF_WIDTH)
        for name in movers:
            pose = sample[1].get(name)
            if not pose:
                continue
            mover = _corners(pose[0], pose[1], pose[2], MOVER_HALF, MOVER_HALF)
            gap = sat_gap(hull_box, mover)
            policy = sat_gap(policy_box, mover)
            per_mover_min[name] = min(per_mover_min[name], gap)
            per_mover_policy[name] = min(per_mover_policy[name], policy)
            if gap < min_hull:
                min_hull, min_hull_at, min_hull_mover = gap, t, name
            if policy < min_policy:
                min_policy, min_policy_at, min_policy_mover = policy, t, name
            if gap <= 0.0:
                contacts.append({'t': round(t, 2), 'mover': name,
                                 'clearance': round(gap, 3)})
    return {
        'frames': seen,
        'movers': movers,
        'contacts': _episodes(contacts),
        'contact_frames': len(contacts),
        'min_clearance': None if min_hull is math.inf else round(min_hull, 3),
        'min_clearance_at': min_hull_at,
        'min_clearance_mover': min_hull_mover,
        'per_mover_min': {k: (None if v is math.inf else round(v, 3))
                          for k, v in per_mover_min.items()},
        'min_policy_clearance': (None if min_policy is math.inf
                                 else round(min_policy, 3)),
        'min_policy_clearance_at': min_policy_at,
        'min_policy_clearance_mover': min_policy_mover,
        'per_mover_policy': {k: (None if v is math.inf else round(v, 3))
                             for k, v in per_mover_policy.items()},
        'hull': {'half_length': hull[0], 'half_width': hull[1]},
        'pushed_while_stopped': _pushed(base, cmd),
    }


def _episodes(contacts):
    """Fold contact frames into episodes so one 0.8 s overlap is not 4 events."""
    episodes = []
    for contact in contacts:
        if (episodes and contact['mover'] == episodes[-1]['mover']
                and contact['t'] - episodes[-1]['end'] <= 1.0):
            episode = episodes[-1]
            episode['end'] = contact['t']
            episode['frames'] += 1
            episode['clearance'] = min(episode['clearance'],
                                       contact['clearance'])
        else:
            episodes.append({'t': contact['t'], 'end': contact['t'],
                             'mover': contact['mover'], 'frames': 1,
                             'clearance': contact['clearance']})
    for episode in episodes:
        episode['seconds'] = round(episode['end'] - episode['t'], 2)
        del episode['end']
    return episodes


def _pushed(base, cmd):
    """Episodes where the base sat still at zero command and then still moved.

    Two guards keep ordinary motion out of the count. The base must already have
    been flat for 0.35 s *before* the frame (otherwise the ramp-down after a stop
    order looks like a push), and the command must stay zero for the whole
    look-ahead too (otherwise a normal pull-away after a yield looks like one).
    """
    episodes = []
    for i, (t, x, y, _yaw) in enumerate(base):
        if x is None:
            continue
        recent = [f for f in base[max(0, i - 4):i + 1]
                  if f[1] is not None and t - f[0] <= 0.35]
        if len(recent) < 2:
            continue
        if max(math.hypot(f[1] - x, f[2] - y) for f in recent) > PUSH_MIN_M:
            continue
        commands = [c for c in cmd if t - 0.35 <= c[0] <= t]
        if not commands:
            continue
        if any(abs(c[1]) > STOP_CMD_MPS or abs(c[2]) > STOP_CMD_RAD_S
               for c in commands):
            continue
        command = commands[-1]
        for later in base[i + 1:]:
            if later[0] - t > 0.6:
                break
            if later[1] is None:
                continue
            if any(abs(c[1]) > STOP_CMD_MPS or abs(c[2]) > STOP_CMD_RAD_S
                   for c in cmd if t < c[0] <= later[0]):
                break          # the base was told to move: not a push
            moved = math.hypot(later[1] - x, later[2] - y)
            if moved > PUSH_MIN_M:
                episodes.append({'t': round(t, 2),
                                 'moved': round(moved, 3),
                                 'cmd': [command[1], command[2]]})
                break
    # One episode per second: the same push is sampled several times.
    deduped = []
    for episode in episodes:
        if deduped and episode['t'] - deduped[-1]['t'] < 1.0:
            deduped[-1]['moved'] = max(deduped[-1]['moved'], episode['moved'])
        else:
            deduped.append(episode)
    return deduped


def report(result, path):
    print(f'\n=== contact / clearance: {path} ===')
    if not result['frames']:
        print('no paired base/mover samples: nothing measured')
        return
    hull = result['hull']
    print(f"frames: {result['frames']}  movers: {', '.join(result['movers'])}")
    print(f"hull: {2*hull['half_length']:.2f} x {2*hull['half_width']:.2f} m")
    if result['min_clearance'] is None:
        print('clearance: no mover samples')
    else:
        print(f"min hull clearance: {result['min_clearance']:+.3f} m "
              f"({result['min_clearance_mover']} at {result['min_clearance_at']:.1f} s)")
        for name, value in result['per_mover_min'].items():
            print(f'  {name}: {value if value is None else f"{value:+.3f}"} m')
        print(f"min policy clearance: {result['min_policy_clearance']:+.3f} m "
              f"({result['min_policy_clearance_mover']} at "
              f"{result['min_policy_clearance_at']:.1f} s)")
    if result['contacts']:
        print(f"contacts: {len(result['contacts'])} episode(s)")
        for episode in result['contacts']:
            print(f"  t={episode['t']:.1f} s {episode['mover']} "
                  f"{episode['seconds']:.1f} s min {episode['clearance']:+.3f} m")
    else:
        print('contacts: none')
    if result['pushed_while_stopped']:
        print(f"pushed while stopped: {len(result['pushed_while_stopped'])} episode(s)")
        for episode in result['pushed_while_stopped']:
            print(f"  t={episode['t']:.1f} s moved {episode['moved']:.3f} m "
                  f"with cmd {episode['cmd'][0]:.2f} m/s {episode['cmd'][1]:.2f} rad/s")
    else:
        print('pushed while stopped: none')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('events', type=pathlib.Path,
                        help='round events.jsonl')
    parser.add_argument('-o', '--out', type=pathlib.Path,
                        help='write the full result as JSON here')
    parser.add_argument('--base-length', type=float,
                        help='override the hull length in metres')
    parser.add_argument('--base-width', type=float,
                        help='override the hull width in metres')
    args = parser.parse_args(argv)
    if not args.events.is_file():
        parser.error(f'{args.events} is not a file')
    hull = (HULL_HALF_LENGTH, HULL_HALF_WIDTH)
    if args.base_length or args.base_width:
        hull = ((args.base_length or 2 * HULL_HALF_LENGTH) / 2.0,
                (args.base_width or 2 * HULL_HALF_WIDTH) / 2.0)
    result = analyze(args.events, hull)
    report(result, args.events)
    if args.out:
        args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
