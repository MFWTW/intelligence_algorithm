"""Independent reliable-only observer serialization; never starts ROS."""
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[4] / 'run_logs/tools/reliable_nav_probe.py'
spec = importlib.util.spec_from_file_location('solo_reliable_under_test', SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class SoloReliableProbeTests(unittest.TestCase):
    def test_writes_message_age_and_count(self):
        node = SimpleNamespace(output=io.StringIO(), last=None, received=0,
                               graph_due=100., transport_label='udp_only', qos_depth=10)
        msg = SimpleNamespace(linear=SimpleNamespace(x=.2), angular=SimpleNamespace(z=-1.9))
        with patch.object(module.time, 'monotonic', return_value=10.):
            module.ReliableNavProbe.command(node, msg)
        with patch.object(module.time, 'monotonic', return_value=11.), \
             patch.object(module.time, 'time', return_value=100.):
            module.ReliableNavProbe.sample(node)
        row = json.loads(node.output.getvalue())
        self.assertEqual((row['messages'],row['age_s'],row['vx'],row['wz'],
                          row['transport_label'],row['qos_depth']),
                         (1,1.,.2,-1.9,'udp_only',10))


if __name__ == '__main__':
    unittest.main()
