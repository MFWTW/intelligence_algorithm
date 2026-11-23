#!/usr/bin/env python3
"""Read-only final XY placement check from one newly received LinkStates frame.

Example: python3 verify_final_placement.py --params config/fixed_joint_pick_place.yaml
  --expected-count 2 --item red_cube_1=A --item blue_cube_1=B --timeout 5
The explicitly supplied YAML must contain the effective parameters for this
round; profiles are not merged and live ROS parameters are not queried. A
volatile subscription takes one frame delivered after it is created; LinkStates
has no timestamp, so publisher-side freshness cannot be independently proven.
Exit codes: 0 all inside, 1 at least one outside, 2 indeterminate/input error.
"""
import argparse
import json
import math
import re
import sys
import time

import yaml


class InputError(ValueError):
    pass


class UniqueLoader(yaml.SafeLoader):
    pass


def _mapping(loader, node, deep=False):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if not isinstance(key, (str, int, float, bool, type(None))):
            raise InputError('invalid YAML mapping key')
        if key in result:
            raise InputError(f'duplicate YAML key: {key}')
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)


def number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise InputError(f'{name} must be a finite number')
    return float(value)


def parse_items(items, count):
    if count <= 0 or len(items) != count:
        raise InputError('expected count must be positive and equal item count')
    parsed, seen = [], set()
    for item in items:
        parts = item.split('=')
        if len(parts) != 2 or not re.fullmatch(r'[A-Za-z0-9_]+', parts[0]) or not re.fullmatch(r'[A-Za-z0-9_]+', parts[1]):
            raise InputError(f'invalid item: {item!r}; use cube=zone')
        cube, zone = parts[0], parts[1].lower()
        if cube in seen:
            raise InputError(f'duplicate cube: {cube}')
        seen.add(cube)
        parsed.append((cube, zone))
    return parsed


def load_config(path, items):
    with open(path, encoding='utf-8') as stream:
        data = yaml.load(stream, Loader=UniqueLoader)
    try:
        params = data['fixed_joint_pick_place']['ros__parameters']
        if not isinstance(params, dict):
            raise TypeError('ros__parameters must be a mapping')
        link = params['cube_link']
        if not isinstance(link, str) or not re.fullmatch(r'[A-Za-z0-9_]+', link):
            raise InputError('invalid cube_link')
        length = number(params['zone_half_length_m'], 'zone_half_length_m')
        width = number(params['zone_half_width_m'], 'zone_half_width_m')
        tolerance = number(params['placement_tolerance_m'], 'placement_tolerance_m')
        if length <= 0 or width <= 0 or tolerance < 0 or tolerance > min(length, width):
            raise InputError('invalid dimensions or tolerance')
        centres = {}
        for _, zone in items:
            key = f'zone_{zone}_centre'
            centre = params[key]
            if not isinstance(centre, list) or len(centre) != 2:
                raise InputError(f'{key} must contain exactly two numbers')
            centres[zone] = [number(v, key) for v in centre]
        return link, length - tolerance, width - tolerance, centres
    except (KeyError, TypeError) as exc:
        raise InputError(f'missing or invalid parameter: {exc}') from exc


def evaluate(items, config, names, poses):
    link, xlimit, ylimit, centres = config
    results = []
    malformed = len(names) != len(poses)
    for cube, zone in items:
        name = f'{cube}::{link}'
        row = {'cube': cube, 'zone': zone.upper(), 'link': name, 'status': 'unknown'}
        indices = [i for i, candidate in enumerate(names) if candidate == name]
        if malformed:
            row['reason'] = 'name/pose length mismatch'
        elif len(indices) != 1:
            row['reason'] = 'missing link' if not indices else 'multiple matching links'
        else:
            try:
                position = poses[indices[0]].position
                x, y = number(position.x, 'x'), number(position.y, 'y')
                cx, cy = centres[zone]
                dx, dy = x - cx, y - cy
                if not math.isfinite(dx) or not math.isfinite(dy):
                    raise InputError('non-finite offset')
                inside = abs(dx) <= xlimit and abs(dy) <= ylimit
                row.update(status='inside' if inside else 'outside', xy=[x, y],
                           centre=[cx, cy], offset=[dx, dy], limits=[xlimit, ylimit])
            except (AttributeError, TypeError, InputError) as exc:
                row['reason'] = str(exc)
        results.append(row)
    code = 2 if any(r['status'] == 'unknown' for r in results) else (1 if any(r['status'] == 'outside' for r in results) else 0)
    return code, results


def receive_frame(timeout):
    # Imports are delayed so offline logic and --help do not require ROS.
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
    from gazebo_msgs.msg import LinkStates

    rclpy.init(args=[])
    node = None
    try:
        node = Node('final_placement_verifier')
        frames = []
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT,
                         durability=DurabilityPolicy.VOLATILE)
        node.create_subscription(LinkStates, '/gazebo/link_states', frames.append, qos)
        deadline = time.monotonic() + timeout
        while not frames and rclpy.ok():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            rclpy.spin_once(node, timeout_sec=remaining)
        if not frames:
            raise InputError('timed out waiting for a new /gazebo/link_states frame')
        return frames[0]
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


class Parser(argparse.ArgumentParser):
    def error(self, message):
        raise InputError(message)


def main(argv=None):
    parser = Parser(description=__doc__)
    parser.add_argument('--params', required=True, help='explicit fixed_joint_pick_place ROS parameter YAML')
    parser.add_argument('--expected-count', type=int, required=True)
    parser.add_argument('--item', action='append', default=[], help='cube=zone; repeat once per cube')
    parser.add_argument('--timeout', type=float, default=5.0, help='wall-clock seconds')
    items = []
    try:
        args = parser.parse_args(argv)
        items = parse_items(args.item, args.expected_count)
        timeout = number(args.timeout, 'timeout')
        if timeout <= 0:
            raise InputError('timeout must be positive')
        config = load_config(args.params, items)
        frame = receive_frame(timeout)
        code, results = evaluate(items, config, frame.name, frame.pose)
        output = {'status': ('pass', 'outside', 'unknown')[code], 'items': results}
    except (Exception, KeyboardInterrupt) as exc:
        code = 2
        output = {'status': 'unknown', 'error': str(exc) or type(exc).__name__,
                  'items': [{'cube': cube, 'zone': zone.upper(), 'status': 'unknown'} for cube, zone in items]}
    print(json.dumps(output, ensure_ascii=False, allow_nan=False))
    return code


if __name__ == '__main__':
    sys.exit(main())
