"""Near-path clearance is measured along the cached route, not the nose."""
import math
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'scripts'))
from obstacle_yield import near_path_clearance


class NearPathClearanceTests(unittest.TestCase):
    def test_cutoff_is_arc_length_even_when_the_route_turns(self):
        path = [(0, 0), (0.6, 0), (0.6, 2)]
        self.assertAlmostEqual(near_path_clearance(path, [(0.6, 0.7)]), 0.0)
        self.assertAlmostEqual(near_path_clearance(path, [(0.6, 1.5)]), 0.6)

    def test_ignores_traveled_segment_and_counts_the_join(self):
        path = [(-2, 0), (3, 0)]
        self.assertAlmostEqual(near_path_clearance(path, [(-1, 0)]), 1.0)
        path_offset = [(0, 0.3), (3, 0.3)]
        self.assertAlmostEqual(near_path_clearance(path_offset, [(0, 0.15)]), 0.0)
        self.assertAlmostEqual(near_path_clearance(path_offset, [(1.5, 0.3)]), 0.3)

    def test_missing_or_unjoinable_path_is_not_clear(self):
        self.assertIsNone(near_path_clearance(None, []))
        self.assertIsNone(near_path_clearance([(0, 0), (2, 0)], None))
        self.assertIsNone(near_path_clearance([(0, 0.6), (2, 0.6)], []))
        self.assertEqual(near_path_clearance([(0, 0), (2, 0)], []), math.inf)

    def test_minimum_uses_all_footprint_points(self):
        path = [(0, 0), (2, 0)]
        self.assertAlmostEqual(near_path_clearance(path, [(0.7, 0.8), (0.8, 0.2)]), 0.2)
