"""Opt-in Nav2 launch: put ALL motion commands through the yield gate.

The pinned stock launch still owns the same lifecycle, planner, controller and
smoother. Only the smoother's input and behavior_server's direct recovery
output are remapped; all Nav2 motion reaches /cmd_vel_nav_guarded via the gate.
"""
import hashlib
import importlib.util
from pathlib import Path

from ament_index_python.packages import get_package_share_directory

STOCK_NAV_SHA256 = '11438a4bc65def4a3d5805919e82bc52ff6008ee6d1fd7c5dd90a29c2f50f56f'


def guarded_nodes(original, found):
    def make_node(*args, **kwargs):
        package = kwargs.get('package')
        if package == 'nav2_velocity_smoother':
            if kwargs.get('executable') != 'velocity_smoother':
                raise RuntimeError('Unexpected smoother: abort guarded navigation')
            remaps = list(kwargs['remappings'])
            expected = ('cmd_vel', 'cmd_vel_nav')
            if remaps.count(expected) != 1:
                raise RuntimeError('Unexpected smoother input: abort guarded navigation')
            remaps[remaps.index(expected)] = ('cmd_vel', 'cmd_vel_nav_guarded')
            output = ('cmd_vel_smoothed', 'cmd_vel')
            if remaps.count(output) != 1:
                raise RuntimeError('Unexpected smoother output: abort guarded navigation')
            remaps[remaps.index(output)] = (
                'cmd_vel_smoothed', 'cmd_vel_smoothed_raw')
            kwargs['remappings'] = remaps
            found.append('smoother')
        elif package == 'nav2_behaviors':
            if kwargs.get('executable') != 'behavior_server':
                raise RuntimeError('Unexpected recovery server: abort guarded navigation')
            # Humble's behavior_server publishes Spin/BackUp directly to
            # cmd_vel, bypassing the smoother. Route it into the guarded input
            # as well; it must never command the base behind the gate's back.
            kwargs['remappings'] = list(kwargs['remappings']) + [
                ('cmd_vel', 'cmd_vel_nav')]
            found.append('behavior')
        return original(*args, **kwargs)
    return make_node


def generate_launch_description():
    stock = Path(get_package_share_directory('nav2_bringup')) / 'launch' / 'navigation_launch.py'
    if hashlib.sha256(stock.read_bytes()).hexdigest() != STOCK_NAV_SHA256:
        raise RuntimeError(f'Nav2 launch changed; refuse yield gate ({stock})')
    spec = importlib.util.spec_from_file_location('_yield_guarded_stock_nav', stock)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    found = []
    module.Node = guarded_nodes(module.Node, found)
    description = module.generate_launch_description()
    if sorted(found) != ['behavior', 'smoother']:
        raise RuntimeError(f'Expected one behavior server and smoother: {found}')
    return description
