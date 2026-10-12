"""Default-off, version-pinned single-variable P0 test: trace AND UDP together.

Exactly one Nav2 node changes relative to stock Humble 1.1.20: the
velocity_smoother process. It runs the instrumented binary (so its own
`input_valid` / `publish_after` counters judge the run instead of any external
observer) *and* that same process gets the UDP-only Fast DDS profile.

Everything else is deliberately identical to stock: the subscription stays
RELIABLE with depth 1, the zero-Twist stop / timeout deceleration semantics are
untouched, no publisher, remap, parameter or safety gate is added, and no second
smoother exists. This is NEVER the default and must not be used in a judged run
without full 5-of-5 evidence.
"""
import hashlib
import importlib.util
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch.substitutions import LaunchConfiguration

STOCK_NAV_SHA256 = '11438a4bc65def4a3d5805919e82bc52ff6008ee6d1fd7c5dd90a29c2f50f56f'


def trace_udp_node_factory(original_node, udp_xml, matches):
    """Wrap only the stock smoother's Node; guard cardinality at call site."""
    def make_node(*args, **kwargs):
        if kwargs.get('package') == 'nav2_velocity_smoother':
            if kwargs.get('executable') != 'velocity_smoother' or 'additional_env' in kwargs:
                raise RuntimeError('Unrecognized smoother node; abort trace+UDP experiment')
            matches.append(kwargs['executable'])
            kwargs['package'] = 'mybot_velocity_smoother_trace'
            kwargs['executable'] = 'velocity_smoother_trace'
            kwargs['additional_env'] = {
                'MYBOT_SMOOTHER_TRACE_FILE': LaunchConfiguration('smoother_trace_file'),
                # Empty unless a P0 poll-fallback round is explicitly requested.
                'MYBOT_SMOOTHER_POLL_FALLBACK_SEC':
                    LaunchConfiguration('smoother_poll_fallback_sec'),
                'FASTRTPS_DEFAULT_PROFILES_FILE': str(udp_xml),
            }
        return original_node(*args, **kwargs)
    return make_node


def generate_launch_description():
    stock = Path(get_package_share_directory('nav2_bringup')) / 'launch' / 'navigation_launch.py'
    if hashlib.sha256(stock.read_bytes()).hexdigest() != STOCK_NAV_SHA256:
        raise RuntimeError(f'Stock Nav2 launch changed: refuse trace+UDP experiment ({stock})')
    if not (Path(get_package_share_directory('mybot_velocity_smoother_trace'))
            / 'package.xml').is_file():
        raise RuntimeError('Instrumented smoother package not installed')
    udp_xml = Path(get_package_share_directory('mybot')) / 'config' / 'fastdds_wsl_udp.xml'
    if not udp_xml.is_file() or 'UDPv4' not in udp_xml.read_text():
        raise RuntimeError(f'UDP-only profile unavailable: {udp_xml}')
    spec = importlib.util.spec_from_file_location('_pinned_stock_navigation_trace_udp', stock)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    matches = []
    module.Node = trace_udp_node_factory(module.Node, udp_xml, matches)
    description = module.generate_launch_description()
    if matches != ['velocity_smoother']:
        raise RuntimeError(f'Expected exactly one stock smoother process; found {matches}')
    return description
