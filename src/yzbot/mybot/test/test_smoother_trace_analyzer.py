"""Synthetic alignment test for read-only internal/observer comparison."""
import importlib.util
from pathlib import Path
import unittest

SCRIPT = Path(__file__).resolve().parents[4] / 'run_logs/tools/analyze_smoother_trace.py'
spec = importlib.util.spec_from_file_location('trace_analyzer_under_test', SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class TraceAnalysisTests(unittest.TestCase):
    def test_attributes_only_aligned_fresh_angular_gaps(self):
        recorder = [dict(mono_abs=100., work_state='前往放置区',
                         nav_age=.1, nav_wz=1.9, cmd_age=9., grab_index=2)]
        observer = [dict(mono_abs=100.1, nav=dict(age_s=.1, wz=1.9),
                         cmd=dict(age_s=9.))]
        def trace(t, n):
            return dict(mono_abs=t, input_valid=n, timer_enter=n*20,
                        publish_after=n*20, input_mono_ns=int(t*1e9),
                        publish_mono_ns=int(t*1e9),
                        timer_timeout_stopped=0, pid=123)
        result = module.analyze(recorder, observer, [trace(99.,0),trace(100.,1)])
        self.assertEqual(result['both_output_stale'],1)
        self.assertEqual(result['stale_with_internal_input'],1)
        self.assertEqual(result['stale_with_internal_publish'],1)
        self.assertEqual(result['no_trace_near_pair'],0)
        self.assertEqual(result['stale_reliable_reader_unrecorded'],1)
        observer[0]['nav_reliable'] = dict(age_s=.1, wz=1.9)
        fresh = module.analyze(recorder, observer, [trace(99.,0), trace(100.,1)])
        self.assertEqual(fresh['stale_reliable_reader_fresh'],1)
        observer[0]['nav_reliable'] = dict(age_s=5., wz=1.9)
        stale_reliable = module.analyze(recorder, observer, [trace(99.,0),trace(100.,1)])
        self.assertEqual(stale_reliable['stale_reliable_reader_stale'],1)
        self.assertEqual(stale_reliable['be_nav_fresh_reliable_nav_stale'],1)
        self.assertEqual(stale_reliable['reliable_reader_stale_internal_input_active'],1)
        stale = module.analyze(recorder, observer, [trace(90.,0), trace(91.,1)])
        self.assertEqual(stale['both_output_stale'],1)
        self.assertEqual(stale['no_trace_near_pair'],1)
        self.assertEqual(stale['stale_with_internal_publish'],0)


if __name__ == '__main__':
    unittest.main()
