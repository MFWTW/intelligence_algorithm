"""Synthetic monotonic joins distinguish UDP reader visibility from default DDS."""
import importlib.util
from pathlib import Path
import unittest

SCRIPT=Path(__file__).resolve().parents[4]/'run_logs/tools/compare_reliable_transports.py'
spec=importlib.util.spec_from_file_location('compare_reliable_test',SCRIPT)
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class CompareReliableTransportsTests(unittest.TestCase):
    def test_default_reliable_stall_with_udp_reader_fresh(self):
        chain=[dict(mono_abs=float(t),nav=dict(age_s=.1),
                    nav_reliable=dict(age_s=10.)) for t in (100,101)]
        default=[dict(mono_abs=float(t),age_s=10.) for t in (100,101)]
        udp=[dict(mono_abs=float(t),age_s=.1) for t in (100,101)]
        trace=[dict(mono_abs=float(t),input_valid=(t-99)*10)
               for t in (100,101)]
        result=module.summarize(chain,default,udp,trace)
        self.assertEqual(result['default_solo_stale_udp_fresh'],2)
        self.assertEqual(result['both_solo_stale'],0)
        self.assertEqual(result['episodes']['solo_default_reliable'][0][
            'smoother_input_valid_delta'],10)
        self.assertEqual(result['reader_unknown']['solo_udp_reliable'],0)
        deep=[dict(mono_abs=float(t),age_s=.1) for t in (100,101)]
        with_deep=module.summarize(chain,default,udp,trace,deep)
        self.assertEqual(with_deep['shallow_stale_deep_fresh'],2)
        self.assertEqual(with_deep['both_default_depths_stale'],0)
        no_udp=module.summarize(chain,default,[],trace)
        self.assertEqual(no_udp['default_solo_stale_udp_fresh'],0)
        self.assertEqual(no_udp['reader_unknown']['solo_udp_reliable'],2)


if __name__=='__main__':
    unittest.main()
