"""Guard scoped diagnostic cleanup from PID-only kill regressions."""
import importlib.util
import os
from pathlib import Path
import unittest

SCRIPT = Path(__file__).resolve().parents[4] / 'run_logs/tools/stop_owned_smoother_trace.py'
spec = importlib.util.spec_from_file_location('stop_owned_trace_test', SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class StopOwnedTraceTests(unittest.TestCase):
    def test_parses_only_trace_process_start(self):
        log = ('[INFO] [velocity_smoother_trace-23]: process started with pid [3049]\n'
               '[INFO] [velocity_smoother-22]: process started with pid [3050]')
        self.assertEqual(module.PID_LINE.findall(log), ['3049'])

    def test_refuses_to_signal_unowned_current_process(self):
        marker = b'MYBOT_SMOOTHER_TRACE_FILE=/tmp/not_this_round'
        self.assertFalse(module.owned(os.getpid(), marker))
        self.assertTrue(module.stop(os.getpid(), marker))


if __name__ == '__main__':
    unittest.main()
