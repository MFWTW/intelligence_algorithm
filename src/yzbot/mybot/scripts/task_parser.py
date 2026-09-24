#!/usr/bin/env python3

"""Parse the competition word problem into a structured robot task.

Pipeline: run the problem generator, ask the DeepSeek chat API to convert the
Chinese word problem into JSON under the configured mapping rules, validate the
result, and publish every intermediate stage. The API key is read from an
environment variable only and is never written to a parameter file.
"""

import json
import math
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String


COLORS = {'red', 'blue'}
ZONES = {'A', 'B', 'C'}


class TaskParser(Node):
    """Generate a problem, parse it with the cloud LLM, validate the JSON."""

    def __init__(self) -> None:
        super().__init__('task_parser')

        self.declare_parameter(
            'generator_path', '/home/yaowei/dev_ws/TMSCQtest_x86_x64.bin'
        )
        self.declare_parameter('generator_timeout_sec', 10.0)
        self.declare_parameter('problem_override', '')
        # The structured task is published transient-local, but a publisher that
        # exits immediately takes its durability cache with it. Keeping the node
        # alive briefly guarantees a late-starting supervisor still latches the
        # task. 0 restores the historical "parse once and exit" behaviour.
        self.declare_parameter('linger_sec', 0.0)

        self.declare_parameter('api_key_env', 'DEEPSEEK_API_KEY')
        self.declare_parameter(
            'api_base_url', 'https://api.deepseek.com/chat/completions'
        )
        self.declare_parameter('model', 'deepseek-chat')
        self.declare_parameter('fallback_model', '')
        self.declare_parameter('temperature', 0.0)
        self.declare_parameter('max_tokens', 400)
        self.declare_parameter('request_timeout_sec', 40.0)
        self.declare_parameter('max_retries', 4)
        self.declare_parameter('retry_backoff_sec', 1.5)

        self.declare_parameter('mapping_rules', '')
        self.declare_parameter('system_prompt', '')
        self.declare_parameter(
            'output_schema',
            '{"capacity":<int>,"items":'
            '[{"who":"<人名>","first":<int>,"second":<int>}]}',
        )
        # Colour and zone come from the competition rule sheet, not from the
        # model, so they are plain deterministic configuration.
        self.declare_parameter('color_x', 'red')
        self.declare_parameter('color_y', 'blue')
        self.declare_parameter('zone_by_count', ['C', 'C', 'A', 'A'])
        self.declare_parameter('zone_rule_text', '')
        self.declare_parameter('max_cubes_per_color', 5)
        # The generator guarantees x,y >= 1 and x + y = 5, which is a strong
        # check that works even for problem templates the oracle cannot parse.
        self.declare_parameter('enforce_count_invariant', True)
        self.declare_parameter('min_count_per_color', 1)
        self.declare_parameter('expected_total_counts', 5)

        self._last_solved = None
        result_qos = QoSProfile(depth=1)
        result_qos.reliability = ReliabilityPolicy.RELIABLE
        result_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self._problem_pub = self.create_publisher(
            String, '/competition/task_problem', result_qos
        )
        self._rules_pub = self.create_publisher(
            String, '/competition/task_rules', result_qos
        )
        self._reply_pub = self.create_publisher(
            String, '/competition/task_raw_reply', result_qos
        )
        self._task_pub = self.create_publisher(
            String, '/competition/task', result_qos
        )
        self._status_pub = self.create_publisher(
            String, '/competition/task_status', result_qos
        )

    # ---------------------------------------------------------------- utils
    def _publish(self, publisher, text: str) -> None:
        message = String()
        message.data = text
        publisher.publish(message)

    def _status(self, text: str) -> None:
        self.get_logger().info(f'Task status: {text}')
        self._publish(self._status_pub, text)

    # ------------------------------------------------------------ generate
    def _generate_problem(self) -> str | None:
        override = str(self.get_parameter('problem_override').value).strip()
        if override:
            self.get_logger().info('Using problem_override instead of the generator.')
            return override

        path = str(self.get_parameter('generator_path').value)
        timeout = float(self.get_parameter('generator_timeout_sec').value)
        if not os.path.isfile(path):
            self.get_logger().error(f'Problem generator not found: {path}')
            return None
        try:
            result = subprocess.run(
                [path], capture_output=True, text=True, timeout=timeout, check=False
            )
        except (OSError, subprocess.SubprocessError) as exc:
            self.get_logger().error(f'Problem generator failed to run: {exc}')
            return None
        if result.returncode != 0:
            self.get_logger().error(
                f'Problem generator exited with code {result.returncode}: '
                f'{result.stderr.strip()[:200]}'
            )
            return None
        for line in result.stdout.splitlines():
            line = line.strip()
            if line:
                return line
        self.get_logger().error('Problem generator produced no output.')
        return None

    # ------------------------------------------------------------- llm
    def _build_prompt(self, problem: str) -> str:
        rules = str(self.get_parameter('mapping_rules').value).strip()
        schema = str(self.get_parameter('output_schema').value).strip()
        return (
            f'映射规则：\n{rules}\n\n'
            f'原始题目：\n{problem}\n\n'
            f'请只输出如下结构的 JSON：{schema}'
        )

    def _request(self, model: str, problem: str) -> str:
        key_env = str(self.get_parameter('api_key_env').value)
        api_key = os.environ.get(key_env, '').strip()
        if not api_key:
            raise RuntimeError(
                f'API key environment variable {key_env} is empty; export it '
                'before starting this node.'
            )

        system_prompt = str(self.get_parameter('system_prompt').value).strip()
        payload = {
            'model': model,
            'temperature': float(self.get_parameter('temperature').value),
            'max_tokens': int(self.get_parameter('max_tokens').value),
            'response_format': {'type': 'json_object'},
            'messages': [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': self._build_prompt(problem)},
            ],
        }
        request = urllib.request.Request(
            str(self.get_parameter('api_base_url').value),
            data=json.dumps(payload).encode('utf-8'),
            headers={
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {api_key}',
            },
            method='POST',
        )
        timeout = float(self.get_parameter('request_timeout_sec').value)
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode('utf-8'))
        return body['choices'][0]['message']['content']

    def _ask_model(self, problem: str) -> str | None:
        primary = str(self.get_parameter('model').value)
        fallback = str(self.get_parameter('fallback_model').value).strip()
        attempts = max(1, int(self.get_parameter('max_retries').value))
        backoff = float(self.get_parameter('retry_backoff_sec').value)
        # The public endpoint occasionally drops TLS connections, so every
        # attempt is retried with backoff and the fallback model is tried last.
        for attempt in range(1, attempts + 1):
            model = primary
            if fallback and attempt > max(1, attempts - 2):
                model = fallback
            started = time.monotonic()
            try:
                reply = self._request(model, problem)
                self.get_logger().info(
                    f'{model} replied in {time.monotonic() - started:.1f}s '
                    f'(attempt {attempt}/{attempts}).'
                )
                return reply
            except urllib.error.HTTPError as exc:
                detail = ''
                try:
                    detail = exc.read().decode('utf-8', 'replace')[:200]
                except Exception:
                    pass
                self.get_logger().warning(
                    f'{model} HTTP {exc.code} on attempt {attempt}/{attempts}: {detail}'
                )
                if exc.code in (401, 403):
                    self.get_logger().error(
                        'API key rejected; fix the key instead of retrying.'
                    )
                    return None
            except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
                self.get_logger().warning(
                    f'{model} attempt {attempt}/{attempts} failed: '
                    f'{type(exc).__name__}: {exc}'
                )
            if attempt < attempts:
                time.sleep(backoff * attempt)
        self.get_logger().error('All LLM attempts failed.')
        return None


    @staticmethod
    def _oracle_totals(problem: str) -> dict | None:
        """Deterministically compute the totals from the generator templates.

        Returns None whenever the text does not match the known shapes, so an
        unfamiliar problem never produces a false rejection. When it does
        parse, the LLM's totals and x/y are checked against real arithmetic,
        which catches miscounting (e.g. reading 9 cats as 12).
        """
        capacity = re.search(r'有\s*(\d+)\s*[件只条个]', problem)
        categories = re.search(
            r'有\s*([^\s、,;，；]+?)\s*、\s*([^\s、,;，；]+?)\s*两种', problem
        )
        if not capacity or not categories:
            return None
        cap = int(capacity.group(1))
        first, second = categories.group(1), categories.group(2)
        if first == second:
            return None
        totals = {first: 0, second: 0}
        # Skip the rest of the declaration clause ("...两种衣柜;") so its tail
        # is not mistaken for a需求 clause.
        tail = re.search(r'[;；。]', problem[categories.end():])
        if tail is None:
            return None
        body = problem[categories.end() + tail.end():]
        # Split only on clause separators: a single person's two items are
        # joined by a comma and must stay in the same clause.
        clauses = re.split(r'[;；。]', body)
        parsed_clauses = 0
        for clause in clauses:
            clause = clause.strip()
            if not clause or clause.startswith('设'):
                continue
            head = re.match(
                r'^([\u4e00-\u9fa5]{2,4}(?:、[\u4e00-\u9fa5]{2,4})*)'
                r'(?:都)?(?:需要|领养|要|买|拿)',
                clause,
            )
            if head is None:
                return None
            people = len(head.group(1).split('、'))
            items = re.findall(
                r'(\d+)\s*[件只条个]?\s*([A-Za-z\u4e00-\u9fa5]+)',
                clause[head.end():],
            )
            if not items:
                return None
            for number, word in items:
                for kind in (first, second):
                    if word.startswith(kind):
                        totals[kind] += int(number) * people
                        break
                else:
                    return None
            parsed_clauses += 1
        if parsed_clauses == 0 or (totals[first] == 0 and totals[second] == 0):
            return None
        return {
            'capacity': cap,
            'first': totals[first],
            'second': totals[second],
            'first_name': first,
            'second_name': second,
        }

    # --------------------------------------------------------- validation
    @staticmethod
    def _extract_json(reply: str) -> dict:
        text = reply.strip()
        if text.startswith('```'):
            text = re.sub(r'^```[a-zA-Z]*\s*', '', text)
            text = re.sub(r'```\s*$', '', text).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if not match:
                raise
            return json.loads(match.group(0))

    def _validate(self, payload: dict) -> list[str]:
        """Check the extraction shape only; all arithmetic happens in _solve."""
        errors = []
        capacity = payload.get('capacity')
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity <= 0:
            errors.append(f'capacity must be a positive integer, got {capacity!r}')
        items = payload.get('items')
        if not isinstance(items, list) or not items:
            errors.append('items must be a non-empty list')
            return errors
        seen_value = False
        for index, item in enumerate(items):
            if not isinstance(item, dict):
                errors.append(f'items[{index}] is not an object')
                continue
            for key in ('first', 'second'):
                value = item.get(key, 0)
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    errors.append(
                        f'items[{index}].{key} must be a non-negative integer, '
                        f'got {value!r}'
                    )
                elif value:
                    seen_value = True
        if not seen_value:
            errors.append('every item is zero; the extraction looks empty')
        return errors

    def _solve(self, payload: dict) -> dict:
        """Sum the extracted line items and apply the ceiling division.

        The model only transcribes what each person needs; summing and
        rounding happen here so they cannot be miscomputed.
        """
        capacity = int(payload['capacity'])
        totals = {'first': 0, 'second': 0}
        for item in payload['items']:
            for key in ('first', 'second'):
                totals[key] += int(item.get(key, 0))
        return {
            'capacity': capacity,
            'totals': totals,
            'x': math.ceil(totals['first'] / capacity) if totals['first'] else 0,
            'y': math.ceil(totals['second'] / capacity) if totals['second'] else 0,
        }

    def _cross_check(self, solved: dict, problem: str) -> list[str]:
        """Compare the computed result against the题面 oracle and invariants."""
        errors = []
        oracle = self._oracle_totals(problem)
        if oracle is not None:
            if solved['capacity'] != oracle['capacity']:
                errors.append(
                    f'capacity={solved["capacity"]} but the题面 says '
                    f'{oracle["capacity"]}'
                )
            for key in ('first', 'second'):
                if solved['totals'][key] != oracle[key]:
                    name = oracle[f'{key}_name']
                    errors.append(
                        f'{name} total {solved["totals"][key]} but counting the题面 '
                        f'gives {oracle[key]}'
                    )
            expected_x = math.ceil(oracle['first'] / oracle['capacity']) \
                if oracle['first'] else 0
            expected_y = math.ceil(oracle['second'] / oracle['capacity']) \
                if oracle['second'] else 0
            if solved['x'] != expected_x:
                errors.append(f'x={solved["x"]} but expected {expected_x}')
            if solved['y'] != expected_y:
                errors.append(f'y={solved["y"]} but expected {expected_y}')

        if bool(self.get_parameter('enforce_count_invariant').value):
            minimum = int(self.get_parameter('min_count_per_color').value)
            total = int(self.get_parameter('expected_total_counts').value)
            for field in ('x', 'y'):
                if solved[field] < minimum:
                    errors.append(
                        f'{field}={solved[field]} is below the minimum {minimum}'
                    )
            if solved['x'] + solved['y'] != total:
                errors.append(
                    f'x + y = {solved["x"] + solved["y"]} but the generator '
                    f'guarantees {total}'
                )

        limit = int(self.get_parameter('max_cubes_per_color').value)
        for field in ('x', 'y'):
            if solved[field] > limit:
                errors.append(
                    f'{field}={solved[field]} exceeds the {limit} cubes per colour'
                )
        return errors

    def _check(self, payload: dict, problem: str) -> tuple[list[str], dict | None]:
        errors = self._validate(payload)
        if errors:
            return errors, None
        solved = self._solve(payload)
        return self._cross_check(solved, problem), solved

    def _repair(self, problem: str, reply: str, errors: list[str]) -> dict | None:
        """Ask the model once to fix a reply that failed validation."""
        feedback = (
            f'你上一次的回复未通过校验，错误如下：\n- '
            + '\n- '.join(errors)
            + f'\n\n上一次回复：\n{reply}'
            + '\n\n原始题目：\n' + problem
            + '\n\n请重新逐句抽取每个人需要的数量（“甲、乙都需要N个X”要拆成'
            + '甲、乙两行），不要自己求和，只输出修正后的 JSON。'
        )
        corrected = self._ask_model(feedback)
        if corrected is None:
            return None
        try:
            payload = self._extract_json(corrected)
        except json.JSONDecodeError as exc:
            self.get_logger().error(f'Repaired reply is still not JSON: {exc}')
            return None
        self._publish(self._reply_pub, corrected)
        remaining, solved = self._check(payload, problem)
        self._last_solved = solved
        if remaining:
            for error in remaining:
                self.get_logger().error(f'After repair: {error}')
            return None
        return payload

    # ------------------------------------------------------- task assembly
    def _assign_zone(self, count: int) -> str:
        """Look the count up in the rule table published before the match.

        Counts are guaranteed to be 1..4 (x,y >= 1 and x + y = 5), so a table
        indexed by count expresses any rule, including ones that use zone B.
        """
        table = [str(zone) for zone in self.get_parameter('zone_by_count').value]
        if not table:
            raise ValueError('zone_by_count must not be empty')
        unknown = [zone for zone in table if zone not in ZONES]
        if unknown:
            raise ValueError(f'zone_by_count {unknown} not in {sorted(ZONES)}')
        if count < 1:
            raise ValueError(f'count must be >= 1, got {count}')
        if count > len(table):
            self.get_logger().warning(
                f'count {count} exceeds the zone table {table}; using the last entry.'
            )
            return table[-1]
        return table[count - 1]

    def _build_tasks(self, payload: dict) -> list[dict]:
        """Map the solved counts onto colours and zones deterministically."""
        color_x = str(self.get_parameter('color_x').value)
        color_y = str(self.get_parameter('color_y').value)
        if color_x not in COLORS or color_y not in COLORS:
            raise ValueError(
                f'color_x/color_y must be in {sorted(COLORS)}; '
                f'got {color_x!r}/{color_y!r}'
            )
        tasks = []
        for field, color in (('x', color_x), ('y', color_y)):
            count = int(payload[field])
            if count > 0:
                tasks.append(
                    {'color': color, 'count': count, 'zone': self._assign_zone(count)}
                )
        return tasks

    # ---------------------------------------------------------------- run
    def run_once(self) -> bool:
        self._status('GENERATING')
        problem = self._generate_problem()
        if problem is None:
            self._status('FAILED:GENERATOR')
            return False
        self.get_logger().info(f'题目: {problem}')
        self._publish(self._problem_pub, problem)
        self._publish(
            self._rules_pub,
            str(self.get_parameter('mapping_rules').value)
            + '\n颜色映射: x='
            + str(self.get_parameter('color_x').value)
            + ', y='
            + str(self.get_parameter('color_y').value)
            + '\n区域规则: '
            + str(self.get_parameter('zone_rule_text').value),
        )

        self._status('PARSING')
        reply = self._ask_model(problem)
        if reply is None:
            self._status('FAILED:LLM')
            return False
        self._publish(self._reply_pub, reply)

        try:
            payload = self._extract_json(reply)
        except json.JSONDecodeError as exc:
            self.get_logger().error(f'Reply is not valid JSON: {exc}')
            self._status('FAILED:JSON')
            return False

        errors, solved = self._check(payload, problem)
        if errors:
            for error in errors:
                self.get_logger().warning(f'Validation: {error}')
            self._status('REPAIRING')
            repaired = self._repair(problem, reply, errors)
            if repaired is None:
                self._status('FAILED:VALIDATION')
                return False
            payload = repaired
            solved = self._last_solved
            self._status('REPAIRED')
        if solved is None:
            self._status('FAILED:VALIDATION')
            return False

        solved['tasks'] = self._build_tasks(solved)
        task_json = json.dumps(solved, ensure_ascii=False, separators=(',', ':'))
        self.get_logger().info(f'结构化任务: {task_json}')
        self._publish(self._task_pub, task_json)
        self._status('SUCCESS')
        return True


def main(args=None) -> None:
    rclpy.init(args=args)
    node = TaskParser()
    exit_code = 1
    try:
        exit_code = 0 if node.run_once() else 1
        linger = float(node.get_parameter('linger_sec').value)
        if exit_code == 0 and linger > 0.0:
            # Keep the transient-local sample alive so a supervisor that starts
            # after this node finished can still receive /competition/task.
            node.get_logger().info(
                f'Holding /competition/task for {linger:.0f}s so late '
                'subscribers latch it.'
            )
            deadline = time.monotonic() + linger
            while rclpy.ok() and time.monotonic() < deadline:
                rclpy.spin_once(node, timeout_sec=0.2)
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        node.get_logger().error(f'Task parser crashed: {exc}')
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    sys.exit(exit_code)


if __name__ == '__main__':
    main()
