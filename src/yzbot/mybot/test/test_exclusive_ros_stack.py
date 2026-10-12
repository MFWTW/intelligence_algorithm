"""Passive exclusive-stack guard matching; does not kill or launch ROS."""
import importlib.util
from pathlib import Path
import unittest

SCRIPT = Path(__file__).resolve().parents[4] / 'run_logs/tools/assert_exclusive_ros_stack.py'
spec = importlib.util.spec_from_file_location('exclusive_guard_test', SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ExclusiveGuardTests(unittest.TestCase):
    def test_detects_stale_instrumented_and_stock_smoothers(self):
        self.assertEqual(module.classify(['/home/x/velocity_smoother_trace', '--ros-args']),
                         'velocity_smoother_trace')
        self.assertEqual(module.classify(['/opt/ros/lib/nav2_velocity_smoother/velocity_smoother']),
                         'velocity_smoother')
        self.assertEqual(module.classify(['/usr/bin/python3', '/x/task_state_machine.py']),
                         'task_state_machine.py')

    def test_does_not_match_script_name_in_shell_source_text(self):
        self.assertIsNone(module.classify(['/usr/bin/python3', '-c',
                                          'print("velocity_smoother_trace")']))
        self.assertIsNone(module.classify(['/usr/bin/bash', '/x/safe_tool.sh']))
        self.assertIsNone(module.classify(['/usr/bin/git', 'diff', '--',
                                          '/x/competition_bringup.launch.py']))


if __name__ == '__main__':
    unittest.main()
