"""Only recognised, validated arithmetic may replace an unavailable LLM."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
# task_parser imports its sibling task_state_machine for the start-button QoS
# contract. Under `ros2 run` the script's own directory is on sys.path; under
# importlib's bare loader it is not, so put it there rather than dressing the
# import up in the parser to suit the harness (the same reasoning the executor's
# obstacle_yield import uses in test_navigation_control.py).
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
SCRIPT = SCRIPTS / 'task_parser.py'
spec = importlib.util.spec_from_file_location('task_parser_under_test', SCRIPT)
parser = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parser)

PROBLEM = ('题目: 每个衣柜有5件衣物;有T恤、裤子两种衣柜;'
           '小美需要2件T恤,1条裤子;小帅、小丽都需要7件T恤;'
           '小娜需要2件T恤,1条裤子;小杰需要1条裤子。'
           '设T恤衣柜数量为x,裤子衣柜数量为y,计算x、y分别为多少。')


class GeneratorArgvTests(unittest.TestCase):
    """Competition mode is switched by appending 11470507 to the generator argv."""

    def generate(self, tmp, **over):
        # A fake generator that echoes the arguments it was given.
        fake = Path(tmp) / 'gen.sh'
        fake.write_text('#!/bin/sh\necho "argv:$*"\n')
        fake.chmod(0o755)
        values = {'problem_override': '', 'generator_path': str(fake),
                  'generator_timeout_sec': 5.0, 'competition_mode': False,
                  'generator_args': ''}
        values.update(over)
        node = parser.TaskParser.__new__(parser.TaskParser)
        node.get_parameter = lambda name: SimpleNamespace(value=values[name])
        log = []
        node.get_logger = lambda: SimpleNamespace(
            warning=lambda m, *a, **k: log.append(str(m)),
            error=lambda m, *a, **k: log.append(str(m)),
            info=lambda m, *a, **k: log.append(str(m)))
        node._resolved_generator_path = lambda: str(fake)
        return parser.TaskParser._generate_problem(node), log

    def test_practice_mode_passes_no_arguments(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            problem, log = self.generate(tmp)
        self.assertEqual(problem, 'argv:')
        self.assertTrue(any('practice mode' in m for m in log))

    def test_competition_mode_appends_the_code(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            problem, log = self.generate(tmp, competition_mode=True)
        self.assertEqual(problem, f'argv:{parser.COMPETITION_CODE}')
        self.assertTrue(any('competition mode' in m for m in log))

    def test_explicit_arguments_pass_through_and_are_not_duplicated(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            problem, _ = self.generate(tmp, generator_args='--seed 7')
            self.assertEqual(problem, 'argv:--seed 7')
            problem, _ = self.generate(
                tmp, competition_mode=True,
                generator_args=parser.COMPETITION_CODE)
            self.assertEqual(problem, f'argv:{parser.COMPETITION_CODE}')


class OracleTwistTests(unittest.TestCase):
    """The generator's twist clauses, computed deterministically.

    Measured 2026-10-12 (seed 995411): a 书箱/书籍 problem where 小雅's clause says
    "他是双胞胎所以需要双倍的书" and 小静's says "和小文需求一样,但少一本故事书".
    deepseek-chat extracted 4 instead of 8 and its repair answered x+y=6 for a
    problem guaranteed to be 5, so TASK_PARSE:FAILED:VALIDATION lost the round
    before the robot moved.

    Three separate defects made the oracle useless for this family:
      * the measure word 本 was missing from [件只条个], so "4本故事书" parsed as
        the item "本故事书" and never matched a category;
      * the head regex greedily took "小文想" as the name (the verb list then
        matched the leftover 要), so a back-reference to 小文 could not resolve;
      * 双倍 / 一样 / 少一本 were not understood at all.
    """

    PROBLEM = (
        '题目: 每个书箱有5本书籍;有故事书、科普书两种书箱。小文想要4本故事书,1本科普书;'
        '小雅想要4本故事书,他是双胞胎所以需要双倍的书;小静和小文需求一样,但少一本故事书;'
        '小豪想要1本科普书;小智想要1本故事书,1本科普书。'
        '设故事书箱数为x,科普书箱数为y,计算x、y。'
    )

    def totals(self, problem):
        payload = parser.TaskParser._offline_oracle_payload(problem)
        if payload is None:
            return None
        item = payload['items'][0]
        return item['first'], item['second']

    def test_the_twist_problem_is_solved_deterministically(self):
        import math
        totals = self.totals(self.PROBLEM)
        self.assertIsNotNone(totals, 'the oracle must recognise this template')
        first, second = totals
        self.assertEqual((first, second), (16, 4))
        x, y = math.ceil(first / 5), math.ceil(second / 5)
        self.assertEqual((x, y), (4, 1))
        self.assertEqual(x + y, 5, 'the generator guarantees x + y == 5')

    def test_the_measure_word_ben_is_understood(self):
        self.assertEqual(self.totals(
            '题目: 每个书箱有5本书籍;有故事书、科普书两种书箱。'
            '小文想要4本故事书,1本科普书。设x,y。'), (4, 1))

    def test_doubling_is_applied(self):
        self.assertEqual(self.totals(
            '题目: 每个书箱有5本书籍;有故事书、科普书两种书箱。'
            '小文想要4本故事书,1本科普书;小雅想要4本故事书,他是双胞胎所以需要双倍的书。'
            '设x,y。'), (12, 1))

    def test_a_back_reference_inherits_and_subtracts(self):
        self.assertEqual(self.totals(
            '题目: 每个书箱有5本书籍;有故事书、科普书两种书箱。'
            '小文想要4本故事书,1本科普书;小静和小文需求一样,但少一本故事书。'
            '设x,y。'), (7, 2))

    def test_a_switch_clause_changes_the_kind(self):
        """'也和小芳一样但是换成橙子' inherits the amount AND changes the kind.

        Measured 2026-10-12 (seed 995433): the oracle ignored 换成, counted 小美's
        item as an apple, and reported 5 apples where the题面 has 4. That false
        cross-check then vetoed the model's CORRECT answer (4, 16) and the round
        ended at TASK_PARSE:FAILED:VALIDATION before the robot moved.
        """
        problem = ('题目: 每个果篮有5个水果;有苹果、橙子两种果篮。小芳想要1个苹果;'
                   '小强想要5个橙子,1个苹果;小华说和小芳一样;小丽要5个橙子;'
                   '小美也和小芳一样但是换成橙子;小刚想要1个苹果, 5个橙子。'
                   '设苹果果篮数量为x,橙子果篮数量为y,计算x、y。')
        self.assertEqual(self.totals(problem), (4, 16))
        import math
        first, second = self.totals(problem)
        self.assertEqual(math.ceil(first / 5) + math.ceil(second / 5), 5)

    def test_a_says_same_as_clause_does_not_swallow_the_speaker(self):
        """'小华说和小芳一样' - the 说 belongs to the speaker."""
        self.assertEqual(self.totals(
            '题目: 每个果篮有5个水果;有苹果、橙子两种果篮。'
            '小芳想要1个苹果;小华说和小芳一样。设x,y。'), (2, 0))

    def test_the_earlier_templates_still_work(self):
        """No regression: the shapes that already parsed must keep parsing."""
        self.assertEqual(self.totals(
            '题目: 每个衣柜有5件衣物;有T恤、裤子两种衣柜;小美需要2件T恤,1条裤子;'
            '小帅、小丽都需要7件T恤;小娜需要2件T恤,1条裤子;小杰需要1条裤子。设x,y。'),
            (18, 3))
        self.assertEqual(self.totals(
            '题目: 每个宠物店有5只宠物;有猫、狗两种宠物店;小爱领养2只猫,5只狗;'
            '小宠、小动都领养2只猫;小乐领养3只猫,5只狗;小欢领养4只狗。设x,y。'),
            (9, 14))

    def test_unknown_wording_still_fails_closed(self):
        self.assertIsNone(self.totals(
            '题目: 每个框有5个东西;有甲、乙两种框;某人需要若干甲。设x,y。'))

    def test_the_oracle_is_used_when_both_models_fail(self):
        src = (Path(__file__).resolve().parents[1] / 'scripts' /
               'task_parser.py').read_text(encoding='utf-8')
        self.assertIn('using the ', src)
        self.assertIn('_offline_oracle_payload(problem)', src)
        # It must be reached from the validation-failure path, not only when the
        # LLM is unreachable.
        self.assertIn('Both models failed validation', src)


class ValidationFallbackTests(unittest.TestCase):
    """A wrong answer must not end the round if a stronger model can fix it.

    Measured 2026-10-12 (seed 995411): the problem generator guarantees x+y=5,
    deepseek-chat answered x+y=4, and its repair answered x+y=6 - the round never
    started (TASK_PARSE:FAILED:VALIDATION). The configured fallback model was only
    ever used for network retries, never for validation failures.
    """

    def source(self):
        import pathlib
        return pathlib.Path(
            pathlib.Path(__file__).resolve().parents[1] / 'scripts' /
            'task_parser.py').read_text(encoding='utf-8')

    def test_the_repair_can_be_told_which_model_to_use(self):
        src = self.source()
        self.assertIn('def _repair(self, problem: str, reply: str, errors: list[str],',
                      src)
        self.assertIn("model: str = ''", src)

    def test_ask_model_accepts_a_forced_model(self):
        src = self.source()
        self.assertIn("def _ask_model(self, problem: str, force_model: str = '')",
                      src)
        self.assertIn('if force_model:', src)

    def test_a_second_repair_uses_the_fallback_model(self):
        src = self.source()
        self.assertIn('stronger = str(self.get_parameter(', src)
        self.assertIn('model=stronger', src)
        self.assertIn("self._status('REPAIRED_STRONGER'", src)

    def test_the_fallback_model_is_configured(self):
        import yaml
        cfg = (Path(__file__).resolve().parents[1] / 'config' /
               'task_parser.yaml')
        params = yaml.safe_load(cfg.read_text(encoding='utf-8'))[
            'task_parser']['ros__parameters']
        self.assertTrue(str(params.get('fallback_model', '')).strip(),
                        'fallback_model must be set for this path to work')


class GeneratorOutputFormatTests(unittest.TestCase):
    """Competition mode prints a seed line before the problem.

    Practice mode emits one line ("题目: ..."); competition mode (the 11470507
    argument) emits four: a "使用种子: N" line, a rule of dashes, the 题目 line,
    and another rule. Taking the first non-empty line - which is what the parser
    did - would hand the model "使用种子: 995389" as the problem.
    """

    PRACTICE = '题目: 每个衣柜有5件衣物;有T恤、裤子两种衣柜;小美需要2件T恤。'

    COMPETITION = (
        '使用种子: 995389\n'
        '------------------------------------------------------------\n'
        '题目: 每个书箱有5本书籍;有故事书、科普书两种书箱;小文借阅3本故事书。\n'
        '------------------------------------------------------------\n'
    )

    def test_practice_output_has_no_seed(self):
        problem, seed = parser.parse_generator_output(self.PRACTICE)
        self.assertIsNone(seed)
        self.assertEqual(problem, self.PRACTICE)

    def test_competition_output_yields_the_problem_not_the_seed(self):
        problem, seed = parser.parse_generator_output(self.COMPETITION)
        self.assertEqual(seed, '995389')
        self.assertTrue(problem.startswith('题目:'))
        self.assertNotIn('使用种子', problem)

    def test_a_wrapped_problem_is_joined(self):
        problem, seed = parser.parse_generator_output(
            '使用种子: 7\n----\n题目: 第一段\n第二段\n----\n')
        self.assertEqual(seed, '7')
        self.assertEqual(problem, '题目: 第一段 第二段')

    def test_empty_output_is_reported_as_no_problem(self):
        for text in ('', '\n\n', '------\n------\n'):
            problem, seed = parser.parse_generator_output(text)
            self.assertIsNone(problem)
            self.assertIsNone(seed)


class OfflineFallbackTests(unittest.TestCase):
    def test_recognised_problem_solves_and_passes_original_gates(self):
        payload = parser.TaskParser._offline_oracle_payload(PROBLEM)
        self.assertEqual(payload['capacity'], 5)
        self.assertEqual(payload['items'][0]['first'], 18)
        self.assertEqual(payload['items'][0]['second'], 3)
        params = {'enforce_count_invariant': True, 'min_count_per_color': 1,
                  'expected_total_counts': 5, 'max_cubes_per_color': 5}
        fake = SimpleNamespace(
            get_parameter=lambda name: SimpleNamespace(value=params[name]),
            _oracle_totals=parser.TaskParser._oracle_totals,
        )
        self.assertEqual(parser.TaskParser._validate(fake, payload), [])
        solved = parser.TaskParser._solve(fake, payload)
        self.assertEqual((solved['x'], solved['y']), (4, 1))
        self.assertEqual(parser.TaskParser._cross_check(fake, solved, PROBLEM), [])

    def test_run_once_publishes_offline_task_when_network_fails(self):
        node = parser.TaskParser.__new__(parser.TaskParser)
        for name in ('_problem_pub', '_rules_pub', '_reply_pub', '_task_pub'):
            setattr(node, name, object())
        params = {'mapping_rules': '', 'color_x': 'red', 'color_y': 'blue',
                  'zone_rule_text': '', 'zone_by_count': ['C', 'C', 'A', 'A'],
                  # The three competition mapping rules are selected by
                  # `mapping_rule`; 2 is the current default.
                  'mapping_rule': 2, 'red_min_count': 3,
                  'zone_x': 'B', 'zone_y': 'A', 'zone_blue': 'B', 'zone_red': 'C',
                  'enforce_count_invariant': True, 'min_count_per_color': 1,
                  'expected_total_counts': 5, 'max_cubes_per_color': 5}
        publish = Mock()
        status = Mock()
        with patch.object(parser.TaskParser, '_generate_problem', return_value=PROBLEM), \
             patch.object(parser.TaskParser, '_ask_model', return_value=None), \
             patch.object(parser.TaskParser, 'get_parameter',
                          side_effect=lambda name: SimpleNamespace(value=params[name])), \
             patch.object(parser.TaskParser, 'get_logger', return_value=Mock()), \
             patch.object(parser.TaskParser, '_publish', publish), \
             patch.object(parser.TaskParser, '_status', status):
            self.assertTrue(node.run_once())
        self.assertEqual(status.call_args.args[0], 'SUCCESS')
        import json
        tasks = [json.loads(c.args[1]) for c in publish.call_args_list
                 if c.args[0] is node._task_pub]
        # x=4, y=1 under the current default rule 2: a count >= 3 is red, and
        # the letter decides the zone (x -> B, y -> A). Under rule 1 the same
        # counts were red4 -> A and blue1 -> C, so this assertion is what pins
        # the default; the switch itself is covered by MappingRuleTests.
        self.assertEqual(tasks[0]['tasks'], [
            {'color': 'red', 'count': 4, 'zone': 'B'},
            {'color': 'blue', 'count': 1, 'zone': 'A'},
        ])

    def test_switching_the_rule_changes_the_assignment(self):
        node = parser.TaskParser.__new__(parser.TaskParser)
        for rule, expected in (
            (1, [{'color': 'red', 'count': 4, 'zone': 'A'},
                 {'color': 'blue', 'count': 1, 'zone': 'C'}]),
            (2, [{'color': 'red', 'count': 4, 'zone': 'B'},
                 {'color': 'blue', 'count': 1, 'zone': 'A'}]),
            (3, [{'color': 'blue', 'count': 4, 'zone': 'B'},
                 {'color': 'red', 'count': 1, 'zone': 'C'}]),
        ):
            values = {'mapping_rule': rule, 'red_min_count': 3, 'zone_x': 'B',
                      'zone_y': 'A', 'zone_blue': 'B', 'zone_red': 'C',
                      'color_x': 'red', 'color_y': 'blue',
                      'zone_by_count': ['C', 'C', 'A', 'A']}
            node.get_parameter = lambda name, v=values: SimpleNamespace(value=v[name])
            node.get_logger = lambda: SimpleNamespace(warning=lambda *a, **k: None)
            self.assertEqual(parser.TaskParser._build_tasks(node, {'x': 4, 'y': 1}),
                             expected, f'rule {rule}')

    def test_unknown_or_partial_question_fails_closed(self):
        for problem in ('未知题型，不认识，不可猜答案',
                        '有T恤、裤子两种衣柜;小美需要2件T恤。',
                        '每个衣柜有5件衣物;有T恤、裤子两种衣柜;'
                        '小美需要2件T恤;另外还要随机物品。'):
            self.assertIsNone(parser.TaskParser._offline_oracle_payload(problem))


if __name__ == '__main__':
    unittest.main()


class StartGateTests(unittest.TestCase):
    """The parse runs only after the operator starts the run.

    The rule sheet hands the problem over at the start signal, so a parse that
    ran at launch would answer a problem nobody has been given yet. Measured cost
    of the gate on 2026-10-14: generator + LLM round trip 6.2 s (4.6 s + 1.5 s).
    """

    def test_the_deployment_profile_gates_the_parse_on_the_start_button(self):
        import yaml
        params = yaml.safe_load(
            (Path(__file__).resolve().parents[1] / 'config'
             / 'task_parser.yaml').read_text(encoding='utf-8'))[
                'task_parser']['ros__parameters']
        self.assertTrue(params['wait_for_start'])
        # A launch that never presses the button must still be able to parse.
        source = SCRIPT.read_text(encoding='utf-8')
        self.assertIn("declare_parameter('wait_for_start', False)", source)
        self.assertIn("declare_parameter('start_topic', '/competition/start')",
                      source)

    def test_the_start_contract_is_shared_with_the_supervisor(self):
        """A VOLATILE press must reach the parser too, as it does the machine."""
        source = SCRIPT.read_text(encoding='utf-8')
        self.assertIn('from task_state_machine import start_button_fallback_qos',
                      source)
        self.assertIn('start_button_fallback_qos()', source)

    def test_the_gate_returns_immediately_when_it_is_off(self):
        node = parser.TaskParser.__new__(parser.TaskParser)
        node.get_parameter = lambda name: SimpleNamespace(value=False)
        with patch.object(parser.rclpy, 'spin_once') as spin:
            self.assertTrue(node.wait_for_start())
        spin.assert_not_called()

    def test_the_gate_blocks_until_the_press_arrives(self):
        node = parser.TaskParser.__new__(parser.TaskParser)
        values = {'wait_for_start': True, 'start_topic': '/competition/start'}
        node.get_parameter = lambda name: SimpleNamespace(value=values[name])
        node._start_requested = False
        statuses = []
        node._status = statuses.append
        node.get_logger = lambda: SimpleNamespace(warning=lambda *a, **k: None)

        def deliver(*_args, **_kwargs):
            # The press arrives on the third poll, like a human pressing a
            # second or two after the stack says it is waiting.
            if len(statuses) and deliver.calls == 2:
                node._on_start_button(SimpleNamespace(data=True))
            deliver.calls += 1

        deliver.calls = 0
        with patch.object(parser.rclpy, 'ok', return_value=True), \
             patch.object(parser.rclpy, 'spin_once', side_effect=deliver):
            self.assertTrue(node.wait_for_start())
        self.assertEqual(deliver.calls, 3, 'must poll until the press')
        self.assertEqual(statuses, ['WAITING_FOR_START'])
