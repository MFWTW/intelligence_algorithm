"""The warehouse plate gate: three colours, one band, fail open.

2026-11-25 (user request): every warehouse owns a floor plate - A green,
B yellow, C cyan - and "the bottom of the camera is that colour" is both an
identity check and a range check. These tests pin the three hue windows, the
band arithmetic (fill and centroid), and the fact that a gate that cannot
confirm still lets the placement through while saying so.
"""
import ast
import pathlib
import textwrap

import numpy as np

SCRIPT = (pathlib.Path(__file__).resolve().parents[1]
          / 'scripts' / 'fixed_joint_pick_place.py')


def _source() -> str:
    return SCRIPT.read_text(encoding='utf-8')


def _module() -> dict:
    """Exec the three plate helpers plus their tables, without rclpy."""
    source = _source()
    wanted_functions = {'zone_plate_hue_window', 'plate_band_metrics'}
    wanted_assignments = {'ZONE_PLATE_HUE_DEG', 'ZONE_PLATE_COLOUR'}
    namespace = {'math': __import__('math'), 'np': np}
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef) and node.name in wanted_functions:
            segment = ast.get_source_segment(source, node)
            exec(textwrap.dedent(segment), namespace)  # noqa: S102
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if (isinstance(target, ast.Name)
                        and target.id in wanted_assignments):
                    segment = ast.get_source_segment(source, node)
                    exec(textwrap.dedent(segment), namespace)  # noqa: S102
    missing = wanted_functions - set(namespace)
    assert not missing, f'helpers missing from the executor: {sorted(missing)}'
    return namespace


def _hsv_band(colour_bgr, rows=24, width=64) -> np.ndarray:
    """A flat HSV image whose bottom band is exactly `colour_bgr`."""
    import cv2
    bgr = np.zeros((rows, width, 3), dtype=np.uint8)
    bgr[:] = colour_bgr
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)


def _blank(width=64, rows=24) -> np.ndarray:
    import cv2
    bgr = np.zeros((rows, width, 3), dtype=np.uint8)
    bgr[:] = (40, 40, 40)                      # the grey Gazebo floor
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)


def test_the_three_plate_colours_are_what_the_world_file_says():
    module = _module()
    # (0 1 1) is 180 deg: the first value used here was 90 and warehouse C
    # measured fill=0.00 for a whole round because of it.
    assert module['ZONE_PLATE_HUE_DEG'] == {'A': 120.0, 'B': 60.0, 'C': 180.0}
    assert module['ZONE_PLATE_COLOUR'] == {'A': 'green', 'B': 'yellow', 'C': 'cyan'}


def test_neighbouring_plates_do_not_share_a_hue_window():
    module = _module()
    window_of = module['zone_plate_hue_window']
    windows = {house: window_of(house, 14.0) for house in 'ABC'}
    # A (green, 120 deg) and C (cyan, 90 deg) are only 30 deg apart, so this is
    # the pair that a sloppy tolerance would merge.
    for left, right in (('A', 'C'), ('C', 'B')):
        lo_l, hi_l = windows[left]
        lo_r, hi_r = windows[right]
        assert hi_l < lo_r or hi_r < lo_l, (left, windows[left], right, windows[right])


def test_an_unknown_warehouse_has_no_window():
    module = _module()
    assert module['zone_plate_hue_window']('D', 14.0) is None
    assert module['zone_plate_hue_window']('', 14.0) is None


def test_a_plate_filling_the_band_measures_as_full_and_centred():
    module = _module()
    metrics = module['plate_band_metrics']
    hsv = _hsv_band((0, 255, 0))               # pure green, BGR
    fill, centroid, rows, columns, dominant = metrics(
        hsv, 0.5, 1.0, module['zone_plate_hue_window']('A', 14.0), 90.0, 60.0)
    assert fill == 1.0
    assert abs(centroid) < 1e-9
    assert rows == 12 and columns == 64
    assert dominant == 60                       # green, in OpenCV 0-179


def test_a_grey_floor_measures_as_empty():
    module = _module()
    metrics = module['plate_band_metrics']
    fill, centroid, _, _, dominant = metrics(
        _blank(), 0.5, 1.0, module['zone_plate_hue_window']('A', 14.0), 90.0, 60.0)
    assert fill == 0.0
    assert centroid == 0.0
    assert dominant == -1                       # grey: nothing passes the gates


def test_the_wrong_plates_colour_measures_as_empty():
    """Cyan is not green: the gate must not accept a neighbouring dock."""
    module = _module()
    metrics = module['plate_band_metrics']
    fill, _, _, _, dominant = metrics(
        _hsv_band((255, 255, 0)), 0.5, 1.0,       # cyan, BGR
        module['zone_plate_hue_window']('A', 14.0), 90.0, 60.0)
    assert fill == 0.0
    assert dominant == 90                       # and the log says so


def test_an_off_centre_plate_reports_its_side():
    module = _module()
    metrics = module['plate_band_metrics']
    hsv = _blank()
    hsv[:, 40:, 0] = 60                        # green patch on the right only
    hsv[:, 40:, 1] = 255
    hsv[:, 40:, 2] = 255
    fill, centroid, _, _, _ = metrics(
        hsv, 0.5, 1.0, module['zone_plate_hue_window']('A', 14.0), 90.0, 60.0)
    assert fill > 0.3
    assert centroid > 0.3                      # right half -> positive


def test_cyan_is_found_by_the_c_window():
    module = _module()
    fill, _, _, _, dominant = module['plate_band_metrics'](
        _hsv_band((255, 255, 0)), 0.5, 1.0,
        module['zone_plate_hue_window']('C', 14.0), 90.0, 60.0)
    assert fill == 1.0
    assert dominant == 90


def test_the_gate_step_runs_before_the_arm_leaves_transport_posture():
    source = _source()
    gate = source.index("'PLACING', 'check the target plate'")
    pre_place = source.index("'PLACING', 'pre-place'")
    carry = source.index("'VERIFY_GRASP', 'verify carried cube'")
    assert carry < gate < pre_place


def test_the_gate_is_observational_by_default_but_can_refuse():
    source = _source()
    assert "declare_parameter('place_visual_gate_fail_open', True)" in source
    assert "declare_parameter('place_visual_gate', True)" in source
    method = source[source.index('def _verify_place_target_plate'):
                    source.index('def _image_callback')]
    # A gate that cannot confirm must return the parameter, not a hard False.
    assert 'place_visual_gate_fail_open' in method
    assert 'PLACE_GATE:PASS' in method and 'PLACE_GATE:MISS' in method
    # And it must always hand the camera back to the cube pipeline.
    assert 'self._vision_enabled = False' in method


# --- L2: steering the last metre with the plate centroid --------------------
#
# Re-added 2026-11-25 on the user's request ("把L2也加进来"). The law the base
# follows into a berth re-aims at the goal whenever the lateral residual leaves
# `lateral_band`; every re-aim turns the base, which is the visible hunt at the
# plate. The plate centroid measures that same error directly, so `visual` lets
# the law stop inferring it from dead reckoning.

def _law_module() -> dict:
    """Exec `dock_creep_twist` alone, with just the maths it needs."""
    source = _source()
    namespace = {'math': __import__('math')}
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef) and node.name == 'dock_creep_twist':
            exec(textwrap.dedent(ast.get_source_segment(source, node)),  # noqa: S102
                 namespace)
    assert 'dock_creep_twist' in namespace
    return namespace


def _twist(visual=None, dx=0.05, dy=0.01, yaw=0.0, base_yaw=0.0,
           distance=0.10, align_distance=0.25):
    law = _law_module()['dock_creep_twist']
    return law(dx, dy, yaw, base_yaw, distance, 0.15, 0.8, 0.06,
               align_distance, visual=visual)


def test_without_vision_a_sideways_residual_still_re_aims():
    # Baseline: the hunt this change is meant to remove.
    vx, wz = _twist(visual=None, dx=0.10, dy=0.20)
    assert vx > 0.0
    assert abs(wz) > 0.5, wz


def test_a_plate_to_the_right_steers_the_base_right():
    plain_vx, plain_wz = _twist(visual=None)
    vx, wz = _twist(visual=(0.9, 0.6))
    # centroid > 0 = plate right of centre -> positive wz turns the base left,
    # so the correction term must pull wz the other way.
    assert wz < plain_wz, (wz, plain_wz)
    assert vx > 0.0


def test_a_centred_plate_leaves_only_the_heading_and_lateral_terms():
    vx, wz = _twist(visual=(1.0, 0.0), dy=0.0, base_yaw=0.02)
    assert vx > 0.0
    assert abs(wz) < 0.5, wz


def test_the_visual_term_is_ignored_far_from_the_plate():
    _, plain = _twist(visual=None, distance=3.0)
    _, near = _twist(visual=(1.0, 1.0), distance=3.0)
    assert abs(plain - near) < 1e-9


def test_the_creep_samples_the_plate_only_near_the_berth():
    source = _source()
    body = source[source.index('def _creep_to_dock__timed'):
                  source.index('def _navigate_to_pickup')]
    assert 'place_servo' in body and 'servo_engage_m' in body
    assert 'visual=visual' in body
    # The camera must be handed back even when the creep fails early.
    wrapper = source[source.index('def _creep_to_dock('):
                     source.index('def _servo_house')]
    assert 'try:' in wrapper and 'finally:' in wrapper
    assert '_stop_place_servo' in wrapper
