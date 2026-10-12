"""Default-off, version-pinned, single-process diagnostic replacement for Nav2.

Uses stock Humble 1.1.20 navigation launch to preserve every node, remap,
parameter and lifecycle name. Only velocity_smoother's package/executable and
its process-local *read-only telemetry file* differ. This is NEVER the default.
No velocity command source, bypass or costmap setting is added.
"""
import hashlib
import importlib.util
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch.substitutions import LaunchConfiguration

STOCK_NAV_SHA256 = '11438a4bc65def4a3d5805919e82bc52ff6008ee6d1fd7c5dd90a29c2f50f56f'


def trace_node_factory(original_node, matches):
    def make_node(*args, **kwargs):
        if kwargs.get('package') == 'nav2_velocity_smoother':
            if kwargs.get('executable') != 'velocity_smoother' or 'additional_env' in kwargs:
                raise RuntimeError('Unrecognized smoother node; abort trace experiment')
            matches.append(kwargs['executable'])
            kwargs['package'] = 'mybot_velocity_smoother_trace'
            kwargs['executable'] = 'velocity_smoother_trace'
            kwargs['additional_env'] = {
                'MYBOT_SMOOTHER_TRACE_FILE': LaunchConfiguration('smoother_trace_file'),
                # Empty unless a P0 poll-fallback round is explicitly requested.
                'MYBOT_SMOOTHER_POLL_FALLBACK_SEC':
                    LaunchConfiguration('smoother_poll_fallback_sec'),
            }
        return original_node(*args, **kwargs)
    return make_node


def generate_launch_description():
    stock = Path(get_package_share_directory('nav2_bringup')) / 'launch' / 'navigation_launch.py'
    if hashlib.sha256(stock.read_bytes()).hexdigest() != STOCK_NAV_SHA256:
        raise RuntimeError(f'Stock Nav2 launch changed: refuse trace experiment ({stock})')
    binary = Path(get_package_share_directory('mybot_velocity_smoother_trace'))
    if not (binary / 'package.xml').is_file():
        raise RuntimeError('Instrumented smoother package not installed')
    spec = importlib.util.spec_from_file_location('_pinned_stock_navigation_trace', stock)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    matches = []
    module.Node = trace_node_factory(module.Node, matches)
    description = module.generate_launch_description()
    if matches != ['velocity_smoother']:
        raise RuntimeError(f'Expected one stock smoother process, found {matches}')
    return description
