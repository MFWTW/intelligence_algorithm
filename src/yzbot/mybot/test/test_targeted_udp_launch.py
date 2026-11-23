"""Isolated launch-graph regression: never spins ROS nodes or publishes control."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

from launch import LaunchContext
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.utilities import perform_substitutions
from launch_ros.actions import Node

BOT = Path(__file__).resolve().parents[2] / 'bot_navigation'
SCRIPT = BOT / 'launch' / 'navigation_udp_experiment.launch.py'
spec = importlib.util.spec_from_file_location('nav_udp_under_test', SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
trace_spec = importlib.util.spec_from_file_location(
    'nav_trace_under_test', BOT / 'launch' / 'navigation_trace_experiment.launch.py')
trace_module = importlib.util.module_from_spec(trace_spec)
trace_spec.loader.exec_module(trace_module)
nav_spec = importlib.util.spec_from_file_location(
    'nav_bringup_under_test', BOT / 'launch' / 'nav_bringup_gazebo2.launch.py')
nav_module = importlib.util.module_from_spec(nav_spec)
nav_spec.loader.exec_module(nav_module)
trace_udp_spec = importlib.util.spec_from_file_location(
    'nav_trace_udp_under_test', BOT / 'launch' / 'navigation_trace_udp_experiment.launch.py')
trace_udp_module = importlib.util.module_from_spec(trace_udp_spec)
trace_udp_spec.loader.exec_module(trace_udp_module)


class TargetedUdpLaunchTests(unittest.TestCase):
    def test_only_existing_smoother_has_process_local_udp(self):
        desc = module.generate_launch_description()
        nodes = [action for entity in desc.entities
                 for action in getattr(entity, 'get_sub_entities', lambda: [])()
                 if isinstance(action, Node)]
        self.assertEqual(len(nodes), 8)
        names = [n.__dict__['_Node__package'] for n in nodes]
        self.assertEqual(names.count('nav2_velocity_smoother'), 1)
        for n in nodes:
            executable = n.__dict__['_ExecuteLocal__process_description']
            env = executable.__dict__['_Executable__additional_env']
            if n.__dict__['_Node__package'] == 'nav2_velocity_smoother':
                self.assertIsNotNone(env)
                self.assertEqual([key[0].text for key, _ in env],
                                 ['FASTRTPS_DEFAULT_PROFILES_FILE'])
                self.assertTrue(env[0][1][0].text.endswith('fastdds_wsl_udp.xml'))
            else:
                self.assertIsNone(env)

    def test_stock_nav_version_change_fails_closed(self):
        with patch.object(module, 'STOCK_NAV_SHA256', '0' * 64):
            with self.assertRaisesRegex(RuntimeError, 'Stock Nav2 launch changed'):
                module.generate_launch_description()

    def nav_context(self, description, **overrides):
        context = LaunchContext()
        context.launch_configurations.update(overrides)
        for action in description.entities:
            if isinstance(action, DeclareLaunchArgument):
                action.execute(context)
        return context

    def active_nav_includes(self, description, context):
        guard = next(e for e in description.entities if isinstance(e, OpaqueFunction))
        for action in guard.execute(context):
            action.execute(context)
        return [e for e in description.entities if isinstance(e, IncludeLaunchDescription)
                and e.condition.evaluate(context)]

    def test_default_stays_on_stock_bringup(self):
        desc = nav_module.generate_launch_description()
        ctx = self.nav_context(desc, _nav_experiment_mode='udp')
        active = self.active_nav_includes(desc, ctx)
        self.assertEqual(len(active), 1)
        source = active[0].launch_description_source
        source.get_launch_description(ctx)
        self.assertEqual(Path(source.location).name, 'bringup_launch.py')

    def test_trace_replaces_only_one_stock_smoother_without_udp(self):
        desc = trace_module.generate_launch_description()
        nodes = [action for entity in desc.entities
                 for action in getattr(entity, 'get_sub_entities', lambda: [])()
                 if isinstance(action, Node)]
        self.assertEqual(len(nodes), 8)
        names = [n.__dict__['_Node__package'] for n in nodes]
        self.assertEqual(names.count('nav2_velocity_smoother'), 0)
        self.assertEqual(names.count('mybot_velocity_smoother_trace'), 1)
        for node in nodes:
            env = node.__dict__['_ExecuteLocal__process_description'].__dict__[
                '_Executable__additional_env']
            if node.__dict__['_Node__package'] == 'mybot_velocity_smoother_trace':
                self.assertEqual([key[0].text for key, _ in env],
                                 ['MYBOT_SMOOTHER_TRACE_FILE',
                                  'MYBOT_SMOOTHER_POLL_FALLBACK_SEC'])
                self.assertEqual(node.__dict__['_Node__node_executable'],
                                 'velocity_smoother_trace')
                self.assertEqual(node.__dict__['_Node__node_name'], 'velocity_smoother')
            else:
                self.assertIsNone(env)

    def test_trace_udp_swaps_one_smoother_and_sets_both_env_vars(self):
        desc = trace_udp_module.generate_launch_description()
        nodes = [action for entity in desc.entities
                 for action in getattr(entity, 'get_sub_entities', lambda: [])()
                 if isinstance(action, Node)]
        self.assertEqual(len(nodes), 8)
        names = [n.__dict__['_Node__package'] for n in nodes]
        self.assertEqual(names.count('nav2_velocity_smoother'), 0)
        self.assertEqual(names.count('mybot_velocity_smoother_trace'), 1)
        for node in nodes:
            env = node.__dict__['_ExecuteLocal__process_description'].__dict__[
                '_Executable__additional_env']
            if node.__dict__['_Node__package'] == 'mybot_velocity_smoother_trace':
                # Exactly one process changes; it gets BOTH the telemetry path
                # and the UDP profile, so transport is the only control-chain
                # variable and the smoother's own counters still judge the run.
                self.assertEqual([key[0].text for key, _ in env],
                                 ['MYBOT_SMOOTHER_TRACE_FILE',
                                  'MYBOT_SMOOTHER_POLL_FALLBACK_SEC',
                                  'FASTRTPS_DEFAULT_PROFILES_FILE'])
                ctx = LaunchContext()
                ctx.launch_configurations['smoother_trace_file'] = '/tmp/t.jsonl'
                ctx.launch_configurations['smoother_poll_fallback_sec'] = ''
                self.assertEqual(perform_substitutions(ctx, env[0][1]), '/tmp/t.jsonl')
                # Empty means the P0 poll fallback stays off unless asked for.
                self.assertEqual(perform_substitutions(ctx, env[1][1]), '')
                self.assertTrue(perform_substitutions(ctx, env[2][1])
                                .endswith('fastdds_wsl_udp.xml'))
                self.assertEqual(node.__dict__['_Node__node_executable'],
                                 'velocity_smoother_trace')
                self.assertEqual(node.__dict__['_Node__node_name'], 'velocity_smoother')
            else:
                self.assertIsNone(env)

    def test_legacy_flags_route_to_localization_and_one_navigation_variant(self):
        cases = [
            ('smoother_udp_experiment', 'true', 'navigation_udp_experiment.launch.py'),
            ('smoother_trace_experiment', 'true', 'navigation_trace_experiment.launch.py'),
            ('smoother_trace_udp_experiment', '1', 'navigation_trace_udp_experiment.launch.py'),
            ('yield_pause_mode', 'true', 'navigation_yield_pause.launch.py'),
        ]
        for flag, value, expected in cases:
            with self.subTest(flag=flag):
                desc = nav_module.generate_launch_description()
                ctx = self.nav_context(desc, **{flag: value}, map='/tmp/test-map.yaml',
                                       smoother_trace_file='/tmp/trace.jsonl',
                                       smoother_poll_fallback_sec='0.3')
                active = self.active_nav_includes(desc, ctx)
                self.assertEqual(len(active), 1)
                *arguments, child = active[0].execute(ctx)
                self.assertEqual(Path(active[0].launch_description_source.location).name,
                                 'navigation_experiments.launch.py')
                for action in arguments:
                    action.execute(ctx)
                localization, navigation = child.entities[0].execute(ctx)
                for include, filename in ((localization, 'localization_launch.py'),
                                          (navigation, expected)):
                    source = include.launch_description_source
                    source.get_launch_description(ctx)
                    self.assertEqual(Path(source.location).name, filename)
                map_arg = dict(localization.launch_arguments)['map']
                self.assertEqual(map_arg.perform(ctx), '/tmp/test-map.yaml')
                nav_args = dict(navigation.launch_arguments)
                if 'trace' in flag:
                    self.assertEqual(nav_args['smoother_trace_file'].perform(ctx),
                                     '/tmp/trace.jsonl')
                    self.assertEqual(nav_args['smoother_poll_fallback_sec'].perform(ctx), '0.3')
                else:
                    self.assertNotIn('smoother_trace_file', nav_args)

    def test_invalid_experiment_flags_fail_closed(self):
        desc = nav_module.generate_launch_description()
        cases = [
            ('mutually exclusive', {'smoother_udp_experiment': 'true',
                                    'smoother_trace_experiment': 'true'}),
            ('mutually exclusive', {'smoother_trace_udp_experiment': '1',
                                    'smoother_udp_experiment': 'true'}),
            ('mutually exclusive', {'smoother_trace_experiment': 'true',
                                    'smoother_trace_udp_experiment': 'true'}),
            ('mutually exclusive', {'yield_pause_mode': 'true',
                                    'smoother_udp_experiment': '1'}),
            ('non-empty smoother_trace_file', {'smoother_trace_udp_experiment': '1',
                                              'smoother_trace_file': '  '}),
            ('Unrecognized smoother experiment', {'smoother_trace_udp_experiment': 'maybe'}),
            ('non-composed', {'smoother_trace_udp_experiment': '1', 'use_composition': 'true'}),
            ('map-based', {'smoother_udp_experiment': 'true', 'slam': 'true'}),
        ]
        for error, overrides in cases:
            with self.subTest(overrides=overrides), self.assertRaisesRegex(RuntimeError, error):
                self.active_nav_includes(desc, self.nav_context(desc, **overrides))

    def test_missing_or_extra_smoother_environment_fails_closed(self):
        calls = []
        def stub(*args, **kwargs):
            calls.append(kwargs)
        wrapped = module.targeted_node_factory(stub, '/isolated/udp.xml', [])
        wrapped(package='nav2_controller', executable='controller_server')
        self.assertNotIn('additional_env', calls[-1])
        wrapped(package='nav2_velocity_smoother', executable='velocity_smoother')
        self.assertEqual(calls[-1]['additional_env'],
                         {'FASTRTPS_DEFAULT_PROFILES_FILE': '/isolated/udp.xml'})
        with self.assertRaisesRegex(RuntimeError, 'Unrecognized smoother'):
            wrapped(package='nav2_velocity_smoother', executable='velocity_smoother',
                    additional_env={'OTHER': 'value'})

    def test_trace_udp_factory_rejects_stock_renames_and_presets(self):
        calls = []
        wrapped = trace_udp_module.trace_udp_node_factory(
            lambda *a, **k: calls.append(k), '/isolated/udp.xml', [])
        wrapped(package='nav2_controller', executable='controller_server')
        self.assertNotIn('additional_env', calls[-1])
        wrapped(package='nav2_velocity_smoother', executable='velocity_smoother')
        env = calls[-1]['additional_env']
        self.assertEqual(sorted(env), ['FASTRTPS_DEFAULT_PROFILES_FILE',
                                       'MYBOT_SMOOTHER_POLL_FALLBACK_SEC',
                                       'MYBOT_SMOOTHER_TRACE_FILE'])
        self.assertEqual(env['FASTRTPS_DEFAULT_PROFILES_FILE'], '/isolated/udp.xml')
        self.assertEqual(calls[-1]['package'], 'mybot_velocity_smoother_trace')
        self.assertEqual(calls[-1]['executable'], 'velocity_smoother_trace')
        for kwargs in ({'executable': 'smoother_v2'}, {'additional_env': {'A': 'b'}}):
            with self.assertRaisesRegex(RuntimeError, 'Unrecognized smoother'):
                wrapped(package='nav2_velocity_smoother',
                        executable=kwargs.get('executable', 'velocity_smoother'),
                        **({'additional_env': kwargs['additional_env']}
                           if 'additional_env' in kwargs else {}))


if __name__ == '__main__':
    unittest.main()
