"""Static-map guard for the loaded blue_cube_3 -> warehouse A third waypoint.

A map check cannot certify the actual Nav2 trajectory or dynamic costmap. In
particular, via_2 may be accepted anywhere in its 0.35 m disk; this test checks
the nominal waypoint and observed handoff, NOT every possible early handoff.
Run a full loaded 5/5 round before keeping any waypoint change.
"""
import ast
import math
from pathlib import Path
import unittest

import numpy as np
import yaml

BOT = Path(__file__).resolve().parents[1]
MAP = BOT.parent / 'bot_navigation' / 'maps'
PARAMS = BOT / 'config' / 'fixed_joint_pick_place.yaml'
SCRIPT = BOT / 'scripts' / 'fixed_joint_pick_place.py'
WAYPOINT = 'blue_cube_3_warehouse_a_via_3'


def load_occupied_boxes():
    metadata = yaml.safe_load((MAP / 'room_from_world.yaml').read_text())
    with (MAP / metadata['image']).open('rb') as f:
        assert f.readline().strip() == b'P5'
        header = f.readline()
        while header.startswith(b'#'):
            header = f.readline()
        width, height = map(int, header.split())
        assert int(f.readline()) == 255
        pixels = np.frombuffer(f.read(), dtype=np.uint8).reshape(height, width)
    # OccupancyGrid.origin is the lower-left CORNER of its first cell, while
    # image row zero represents the north edge of the map.
    rows, cols = np.nonzero(pixels <= 89)  # PGM <= (1-0.65)*255 is occupied
    resolution = float(metadata['resolution'])
    ox, oy = metadata['origin'][:2]
    xs = ox + (cols + 0.5) * resolution
    ys = oy + (height - rows - 0.5) * resolution
    return xs, ys, resolution


class LoadedDetourClearanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.xs, cls.ys, cls.resolution = load_occupied_boxes()
        cls.params = yaml.safe_load(PARAMS.read_text())['fixed_joint_pick_place']['ros__parameters']

    @classmethod
    def clearance(cls, x, y):
        """Euclidean distance to nearest occupied *cell rectangle*, not centre."""
        half = cls.resolution / 2
        dx = np.maximum(np.abs(cls.xs - x) - half, 0.0)
        dy = np.maximum(np.abs(cls.ys - y) - half, 0.0)
        return float(np.hypot(dx, dy).min())

    @classmethod
    def chord_lower_bound(cls, start, end, spacing=0.005):
        length = math.dist(start[:2], end[:2])
        segments = max(1, math.ceil(length / spacing))
        sampled = min(
            cls.clearance(start[0] + (end[0] - start[0]) * i / segments,
                          start[1] + (end[1] - start[1]) * i / segments)
            for i in range(segments + 1)
        )
        # Distance to a fixed obstacle set is 1-Lipschitz. Every point on the
        # chord is <= half a sample interval from one evaluated point.
        return sampled - length / (2 * segments)

    def test_nominal_segment_and_observed_handoff_are_clear(self):
        loaded_radius = float(self.params['loaded_robot_radius_m'])
        self.assertEqual(loaded_radius, 0.35)
        self.assertEqual(float(self.params['loaded_inflation_radius_m']), 0.60)
        via2 = self.params['blue_cube_3_warehouse_a_via_2']
        target = self.params[WAYPOINT]
        self.assertEqual(target, [12.7, -2.8, 2.422763])
        self.assertLess(
            self.chord_lower_bound(via2, [12.475, -3.175]), loaded_radius,
            'the historical diagonal cut through the loaded footprint'
        )
        self.assertGreater(
            self.chord_lower_bound(via2, target), loaded_radius,
            'the new nominal segment must stay outside occupied cell boxes'
        )
        # Both prior failures began this leg near (13.17,-3.66), because
        # via_2 had already been accepted at 0.35 m rather than reached exactly.
        self.assertGreater(
            self.chord_lower_bound([13.17, -3.66], target),
            loaded_radius + 0.10,
            'the observed handoff must have extra loaded-payload clearance'
        )
        # Read the actual detour handoff limit from the executor: raising it
        # silently would void the clearance bound on the new goal's disk.
        tree = ast.parse(SCRIPT.read_text())
        handoffs = [
            ast.literal_eval(kw.value)
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and node.args
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == '_navigate_to'
            and isinstance(node.args[0], ast.JoinedStr)
            and any(isinstance(part, ast.Constant)
                    and 'detour waypoint' in str(part.value)
                    for part in node.args[0].values)
            for kw in node.keywords if kw.arg == 'completion_radius_m'
        ]
        self.assertEqual(handoffs, [0.35])
        # Even the nearest point on the waypoint 3 acceptance disk should
        # retain at least a 5 cm margin beyond the carried-payload radius.
        handoff_radius = handoffs[0]
        self.assertGreaterEqual(
            self.clearance(*target[:2]), loaded_radius + handoff_radius + 0.05
        )

    def test_standalone_default_agrees_with_competition_yaml(self):
        tree = ast.parse(SCRIPT.read_text())
        defaults = [
            ast.literal_eval(node.args[1]) for node in ast.walk(tree)
            if isinstance(node, ast.Call) and len(node.args) >= 2
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == 'declare_parameter'
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == WAYPOINT
        ]
        self.assertEqual(defaults, [self.params[WAYPOINT]])


if __name__ == '__main__':
    unittest.main()
