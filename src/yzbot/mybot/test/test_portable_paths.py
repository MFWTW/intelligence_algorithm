"""The repo must not depend on one machine's home directory.

Measured 2026-10-13 on a "if I switch computers" review: three shipped files
carried paths from the machine that wrote them, and each failed differently on a
clone elsewhere:

  * ``config/profiles/nav_rpp15.yaml`` and
    ``bot_navigation/param/originbot_nav2_2.yaml`` named the tuned behavior tree
    as ``/home/<user>/dev_ws/install/...`` - Nav2 then warns "Could not load
    behavior tree file" and silently runs its STOCK recovery tree, so the robot
    still moves but the tuned 4-second recovery is gone (a behaviour regression
    that looks like nothing is wrong);
  * ``config/task_parser.yaml`` + ``launch/task_parser.launch.py`` assumed the
    repo was cloned to ``~/dev_ws``, so the problem generator was not found and
    the round ended in TASK_PARSE:FAILED before the robot moved;
  * the operator docs pointed at one Windows user's Downloads folder.

The fixes are all of the "resolve it where you actually are" kind, never a
literal ``../``: Nav2 opens a relative BT path against the node's own CWD, so a
relative path there would be a new bug rather than a fix. These tests pin the
resolution, not the mechanism.
"""
import importlib.util
import os
import re
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import yaml

ROOT = Path(__file__).resolve().parents[1]        # src/yzbot/mybot
YZBOT = ROOT.parent                               # src/yzbot
NAV_LAUNCH = YZBOT / 'bot_navigation' / 'launch' / 'nav_bringup_gazebo2.launch.py'
GENERATOR_NAME = 'TMSCQtest_x86_x64.bin'

# A path that can only exist on ONE machine: a real user's home, root's home, a
# mounted Windows drive, or a \\wsl.localhost share. "~" and $HOME are fine -
# they expand per user.
MACHINE_PATH = re.compile(
    r'/home/[A-Za-z0-9_.-]+|/root/|/mnt/[a-z]/|\\\\wsl|[A-Za-z]:\\')

# Shipped runtime files only. Test files legitimately contain invented paths
# (e.g. '/home/x/velocity_smoother_trace' as a classifier fixture) and .vscode/
# is gitignored, so neither is part of what travels to another machine.
def _shipped_runtime_files():
    patterns = ['launch/*.py', 'scripts/*.py', 'scripts/*.sh',
                'config/*.yaml', 'config/**/*.yaml', 'param/*.yaml']
    files = []
    for package in sorted(YZBOT.iterdir()):
        if not package.is_dir():
            continue
        for pattern in patterns:
            files.extend(sorted(package.glob(pattern)))
    return files


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class NoMachinePathsInShippedFilesTests(unittest.TestCase):
    def test_no_shipped_runtime_file_names_a_home_directory(self):
        offenders = {}
        for path in _shipped_runtime_files():
            text = path.read_text(encoding='utf-8', errors='replace')
            hits = [f'{path.relative_to(YZBOT)}:{number}: {line.strip()}'
                    for number, line in enumerate(text.splitlines(), 1)
                    if MACHINE_PATH.search(line)]
            if hits:
                offenders[path.name] = hits
        self.assertEqual(
            offenders, {},
            'these paths exist on one machine only; resolve them from the '
            'package share directory / workspace root instead')

    def test_the_scanner_would_catch_a_regression(self):
        # A test that cannot fail is worse than no test.
        self.assertTrue(MACHINE_PATH.search(
            'default_nav_to_pose_bt_xml: /home/someone/dev_ws/install/x.xml'))
        self.assertTrue(MACHINE_PATH.search(r'C:\Users\someone\Downloads\x.json'))
        self.assertTrue(MACHINE_PATH.search(r'\\wsl.localhost\Ubuntu\x.json'))
        for portable in ('~/dev_ws/x.bin', '$HOME/dev_ws/x.bin',
                         'TMSCQtest_x86_x64.bin', '/opt/ros/humble/setup.bash'):
            self.assertIsNone(MACHINE_PATH.search(portable), portable)


class NavBehaviorTreeResolutionTests(unittest.TestCase):
    """Nav2's tuned recovery tree must be found on whatever machine runs it."""

    @classmethod
    def setUpClass(cls):
        cls.nav = _load(NAV_LAUNCH, 'nav_bringup_portable_under_test')

    def includes(self, description):
        def walk(entities):
            for entity in entities:
                yield entity
                for sub in getattr(entity, 'get_sub_entities', lambda: [])():
                    yield sub
        from launch.actions import IncludeLaunchDescription
        return [a for a in walk(description.entities)
                if isinstance(a, IncludeLaunchDescription)]

    def test_the_path_resolves_to_this_machines_installation(self):
        resolved = self.nav.resolve_nav_bt_xml()
        self.assertTrue(resolved, 'mybot is installed, so a path must resolve')
        self.assertTrue(Path(resolved).is_file(),
                        f'the resolved BT xml must exist: {resolved}')
        self.assertEqual(Path(resolved).name, 'nav_to_pose_short_recovery.xml')

    def test_an_explicit_bt_tree_override_is_honoured_and_fails_closed(self):
        """The A/B stable-path tree is selected by env var, never by editing
        the production YAML, and a typo must abort instead of silently falling
        back to the tuned tree (which would make the A/B meaningless)."""
        import os
        stable = ROOT / 'config' / 'nav_to_pose_stable_path.xml'
        self.assertTrue(stable.is_file())
        previous = os.environ.get('MYBOT_NAV_TO_POSE_BT_XML')
        try:
            os.environ['MYBOT_NAV_TO_POSE_BT_XML'] = str(stable)
            self.assertEqual(self.nav.resolve_nav_bt_xml(), str(stable))
            os.environ['MYBOT_NAV_TO_POSE_BT_XML'] = str(ROOT / 'config' / 'nope.xml')
            with self.assertRaisesRegex(RuntimeError, 'missing file'):
                self.nav.resolve_nav_bt_xml()
        finally:
            if previous is None:
                os.environ.pop('MYBOT_NAV_TO_POSE_BT_XML', None)
            else:
                os.environ['MYBOT_NAV_TO_POSE_BT_XML'] = previous
        self.assertEqual(Path(self.nav.resolve_nav_bt_xml()).name,
                         'nav_to_pose_short_recovery.xml')

    def test_the_stable_path_tree_keeps_a_valid_path_instead_of_replanning(self):
        """The A/B tree must replan only when there is no valid path yet."""
        from xml.etree import ElementTree
        for name in ('nav_to_pose_stable_path.xml',
                     'nav_through_poses_stable_path.xml'):
            with self.subTest(name=name):
                root = ElementTree.parse(ROOT / 'config' / name).getroot()
                self.assertIsNotNone(root.find('.//IsPathValid'))
                self.assertIsNotNone(root.find('.//GlobalUpdatedGoal'))
                # The recovery ladder and the tuned short-recovery shape stay:
                # a stable route must not cost the ability to recover.
                self.assertIsNotNone(root.find('.//FollowPath'))
                self.assertIsNotNone(root.find('.//BackUp'))

    def test_every_nav2_include_gets_the_rewritten_params_file(self):
        # Six includes: stock bringup, experimental localization, three smoother
        # experiments, and guarded yield-pause Nav2. Every mode must see the same
        # BT path rewrite instead of silently switching to Nav2's stock tree.
        from launch import LaunchContext
        from nav2_common.launch import RewrittenYaml
        description = self.nav.generate_launch_description()
        includes = self.includes(description)
        self.assertEqual(len(includes), 6)

        context = LaunchContext()
        shipped = Path(self.nav.get_package_share_directory('bot_navigation')) \
            / 'param' / 'originbot_nav2_2.yaml'
        context.launch_configurations['params_file'] = str(shipped)
        for include in includes:
            arguments = dict(include.launch_arguments)
            self.assertIn('params_file', arguments)
            self.assertIsInstance(arguments['params_file'], RewrittenYaml)
            rewritten = context.perform_substitution(arguments['params_file'])
            data = yaml.safe_load(Path(rewritten).read_text(encoding='utf-8'))
            got = data['bt_navigator']['ros__parameters'][
                'default_nav_to_pose_bt_xml']
            self.assertTrue(Path(got).is_file(), got)

    def test_the_shipped_profiles_leave_the_path_empty(self):
        # Empty is the contract with the launch file: an absolute path here is
        # what used to break on other machines, and RewrittenYaml only rewrites
        # keys that exist, so the key must stay.
        for path in (ROOT / 'config' / 'profiles' / 'nav_rpp15.yaml',
                     YZBOT / 'bot_navigation' / 'param' / 'originbot_nav2_2.yaml'):
            data = yaml.safe_load(path.read_text(encoding='utf-8'))
            value = data['bt_navigator']['ros__parameters'][
                'default_nav_to_pose_bt_xml']
            self.assertEqual(value, '', f'{path.name} must ship an empty path')

    def test_a_machine_without_mybot_falls_back_instead_of_crashing(self):
        # bot_navigation must stay usable on its own: no rewrite, and Nav2 then
        # uses its stock recovery tree - the same graceful fallback as before.
        from ament_index_python.packages import PackageNotFoundError
        original = self.nav.get_package_share_directory

        def missing(name):
            if name == 'mybot':
                raise PackageNotFoundError(name)
            return original(name)

        self.nav.get_package_share_directory = missing
        try:
            self.assertEqual(self.nav.resolve_nav_bt_xml(), '')
            self.assertEqual(self.nav.nav_params_with_local_bt_xml('/tmp/p.yaml'),
                             '/tmp/p.yaml')
        finally:
            self.nav.get_package_share_directory = original

    def test_a_relative_or_home_path_is_never_used_for_the_bt_xml(self):
        # The two forms that look portable and are not: Nav2 opens a relative
        # path against the node's CWD, and it does not expand $HOME inside a
        # parameter file.
        text = NAV_LAUNCH.read_text(encoding='utf-8')
        self.assertIn('RewrittenYaml', text)
        for path in (ROOT / 'config' / 'profiles' / 'nav_rpp15.yaml',
                     YZBOT / 'bot_navigation' / 'param' / 'originbot_nav2_2.yaml'):
            value = next(
                line.split(':', 1)[1].strip()
                for line in path.read_text(encoding='utf-8').splitlines()
                if line.strip().startswith('default_nav_to_pose_bt_xml'))
            self.assertNotIn('$HOME', value)
            self.assertNotIn('/home/', value)


class GeneratorPathTests(unittest.TestCase):
    """The problem generator ships in the workspace root; find it there."""

    @classmethod
    def setUpClass(cls):
        cls.parser = _load(ROOT / 'scripts' / 'task_parser.py',
                           'task_parser_portable_under_test')

    def node(self, generator_path, **over):
        values = {'generator_path': generator_path, 'problem_override': '',
                  'generator_timeout_sec': 5.0, 'competition_mode': False,
                  'generator_args': ''}
        values.update(over)
        node = self.parser.TaskParser.__new__(self.parser.TaskParser)
        node.get_parameter = lambda name: SimpleNamespace(value=values[name])
        return node

    def test_the_shipped_config_does_not_name_a_directory(self):
        params = yaml.safe_load(
            (ROOT / 'config' / 'task_parser.yaml').read_text(encoding='utf-8')
        )['task_parser']['ros__parameters']
        self.assertEqual(params['generator_path'], GENERATOR_NAME)

    def test_the_real_workspace_copy_is_found(self):
        resolved = self.parser.TaskParser._resolved_generator_path(
            self.node(GENERATOR_NAME))
        self.assertEqual(Path(resolved).name, GENERATOR_NAME)
        self.assertTrue(os.path.isfile(resolved), resolved)

    def test_a_relative_generator_in_cwd_becomes_an_executable_path(self):
        # Existence is not enough: subprocess.run(['name']) searches PATH and
        # will fail to execute a file that only exists in cwd.
        with tempfile.TemporaryDirectory() as tmp:
            fake = Path(tmp) / 'relative_generator.bin'
            fake.write_text('#!/bin/sh\necho ready\n', encoding='utf-8')
            fake.chmod(0o755)
            with patch('os.getcwd', return_value=tmp):
                with patch('os.path.isfile', side_effect=lambda name:
                           name == 'relative_generator.bin'):
                    resolved = self.parser.TaskParser._resolved_generator_path(
                        self.node('relative_generator.bin'))
        self.assertTrue(os.path.isabs(resolved))
        self.assertEqual(Path(resolved).name, 'relative_generator.bin')

    def test_mybot_ws_selects_the_workspace_when_it_is_set(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake = Path(tmp) / 'other_generator.bin'
            fake.write_text('#!/bin/sh\n')
            with patch.dict(os.environ, {'MYBOT_WS': tmp}):
                resolved = self.parser.TaskParser._resolved_generator_path(
                    self.node('other_generator.bin'))
        self.assertEqual(resolved, str(fake))

    def test_an_explicit_absolute_path_still_wins(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake = Path(tmp) / 'explicit.bin'
            fake.write_text('#!/bin/sh\n')
            resolved = self.parser.TaskParser._resolved_generator_path(
                self.node(str(fake)))
        self.assertEqual(resolved, str(fake))

    def test_home_and_environment_forms_still_expand(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake = Path(tmp) / 'env.bin'
            fake.write_text('#!/bin/sh\n')
            with patch.dict(os.environ, {'MYBOT_FAKE_DIR': tmp}):
                resolved = self.parser.TaskParser._resolved_generator_path(
                    self.node('$MYBOT_FAKE_DIR/env.bin'))
        self.assertEqual(resolved, str(fake))

    def test_a_missing_generator_reports_every_place_it_looked(self):
        errors = []
        node = self.node('definitely_absent_generator_xyz.bin')
        node.get_logger = lambda: SimpleNamespace(
            info=lambda *a, **k: None, warning=lambda *a, **k: None,
            error=lambda message, *a, **k: errors.append(str(message)))
        self.assertIsNone(self.parser.TaskParser._generate_problem(node))
        self.assertTrue(errors, 'a missing generator must be reported')
        self.assertIn('MYBOT_WS', errors[0], 'the fix must be in the message')
        self.assertIn('also tried', errors[0])

    def test_the_launch_default_points_at_an_existing_file(self):
        launch = _load(ROOT / 'launch' / 'task_parser.launch.py',
                       'task_parser_launch_portable_under_test')
        default = launch.default_generator_path()
        self.assertTrue(os.path.isfile(default), default)
        self.assertEqual(Path(default).name, GENERATOR_NAME)


class StartCompetitionWorkspaceTests(unittest.TestCase):
    """The launcher finds its own workspace instead of assuming ~/dev_ws."""

    SCRIPT = ROOT / 'scripts' / 'start_competition.sh'

    def test_the_home_default_is_gone(self):
        text = self.SCRIPT.read_text(encoding='utf-8')
        self.assertNotIn('MYBOT_WS:-$HOME/dev_ws', text)
        self.assertIn('script_dir=', text)

    def test_it_walks_up_from_its_own_location(self):
        # Installed: <ws>/install/mybot/lib/mybot/ ; source tree:
        # <ws>/src/yzbot/mybot/scripts/ - four levels up is <ws> in both.
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp) / 'some_ws'
            installed = ws / 'install' / 'mybot' / 'lib' / 'mybot'
            installed.mkdir(parents=True)
            (ws / 'install').mkdir(exist_ok=True)
            (ws / 'install' / 'setup.bash').write_text('')
            script = installed / 'start_competition.sh'
            script.write_text(self.SCRIPT.read_text(encoding='utf-8'))
            environment = dict(os.environ)
            environment.pop('MYBOT_WS', None)
            environment['HOME'] = str(Path(tmp) / 'home_without_dev_ws')
            result = subprocess.run(
                ['bash', '-x', str(script), '--help'],
                capture_output=True, text=True, env=environment, timeout=60)
            self.assertEqual(result.returncode, 0, result.stderr[-800:])
            resolved = [line.split('=', 1)[1] for line in result.stderr.splitlines()
                        if line.startswith('+ WS=')]
            self.assertTrue(resolved, result.stderr[-800:])
            self.assertEqual(resolved[-1], str(ws))


if __name__ == '__main__':
    unittest.main()
