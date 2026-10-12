"""Opt-in, version-pinned Nav2 launch with UDP on velocity_smoother ONLY.

Never use this experiment in a judged run without full safety/5-of-5 evidence.
It executes the installed Humble navigation_launch.py unmodified except for one
Node constructor's per-process environment. The upstream launch SHA guard fails
closed when the installation changes. No second smoother or cmd_vel publisher
is created; the existing velocity limits, lifecycle manager and remaps remain.
"""
import hashlib
import importlib.util
from pathlib import Path

from ament_index_python.packages import get_package_share_directory

STOCK_NAV_SHA256 = '11438a4bc65def4a3d5805919e82bc52ff6008ee6d1fd7c5dd90a29c2f50f56f'


def targeted_node_factory(original_node, udp_xml, matches):
    """Wrap only the stock smoother's Node; guard cardinality at call site."""
    def make_node(*args, **kwargs):
        if kwargs.get('package') == 'nav2_velocity_smoother':
            if kwargs.get('executable') != 'velocity_smoother' or 'additional_env' in kwargs:
                raise RuntimeError('Unrecognized smoother launch node; abort UDP experiment')
            matches.append(kwargs['executable'])
            kwargs['additional_env'] = {'FASTRTPS_DEFAULT_PROFILES_FILE': str(udp_xml)}
        return original_node(*args, **kwargs)
    return make_node


def generate_launch_description():
    stock = Path(get_package_share_directory('nav2_bringup')) / 'launch' / 'navigation_launch.py'
    if hashlib.sha256(stock.read_bytes()).hexdigest() != STOCK_NAV_SHA256:
        raise RuntimeError(f'Stock Nav2 launch changed: refuse targeted UDP experiment ({stock})')
    udp_xml = Path(get_package_share_directory('mybot')) / 'config' / 'fastdds_wsl_udp.xml'
    if not udp_xml.is_file() or 'UDPv4' not in udp_xml.read_text():
        raise RuntimeError(f'UDP-only profile unavailable: {udp_xml}')
    spec = importlib.util.spec_from_file_location('_pinned_stock_navigation', stock)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    matches = []
    module.Node = targeted_node_factory(module.Node, udp_xml, matches)
    description = module.generate_launch_description()
    if matches != ['velocity_smoother']:
        raise RuntimeError(f'Expected exactly one stock smoother process; found {matches}')
    return description
