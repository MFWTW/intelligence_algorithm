#!/usr/bin/env python3

"""Parse the competition word problem into a structured robot task.

Pipeline: run the problem generator, ask the DeepSeek chat API to convert the
Chinese word problem into JSON under the configured mapping rules, validate the
result, and publish every intermediate stage. The API key is read from the
environment or from a local secret file (see ``resolve_api_key``) and is never
written into a parameter file or a log line.
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
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String


COLORS = {'red', 'blue'}
ZONES = {'A', 'B', 'C'}

# Where the operator stores the key once so plain ``ros2 launch`` picks it up
# without exporting anything in the shell. ``set_deepseek_key.sh`` writes the
# first one. Files may hold either a bare key or ``DEEPSEEK_API_KEY=sk-...``.
DEFAULT_KEY_FILES = (
    '~/.config/mybot/deepseek_api_key',
    '~/.deepseek_api_key',
    '.secrets/deepseek_api_key',
    '.env',
)

# How long a *failed* parse keeps its latched status alive, so the supervisor
# and the dashboard always get a reason instead of a silent IDLE run.
FAILURE_HOLD_MIN_SEC = 10.0
FAILURE_HOLD_MAX_SEC = 20.0
# Endpoint matching on WSL/Fast DDS can take several seconds, so a status
# published once before the supervisor matched is simply lost. Re-sending the
# last status while holding is idempotent (the supervisor ignores it once it
# has acted) and makes the reason reliable.
STATUS_REPUBLISH_SEC = 2.0


def _key_from_file(path: Path, key_env: str) -> str:
    """Return the key stored in ``path``, or '' when the file has none.

    A file is accepted in both shapes: a bare key on the first useful line, or
    ``KEY=VALUE`` lines as in a ``.env`` file (only ``key_env`` is honoured
    there, and ``export`` prefixes / surrounding quotes are tolerated).
    """
    try:
        text = path.read_text(encoding='utf-8')
    except (OSError, UnicodeDecodeError):
        return ''
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        if '=' in line:
            name, _, raw = line.partition('=')
            if name.strip().removeprefix('export').strip() != key_env:
                continue
            line = raw.strip()
        return line.strip().strip('"').strip("'")
    return ''


def resolve_api_key(key_env: str, key_file: str = '') -> tuple:
    """Return ``(api_key, source)``; ``source`` is safe to log, the key is not.

    Lookup order, first hit wins:

    1. ``$<key_env>`` already exported in the environment;
    2. the ``api_key_file`` node parameter, then ``$DEEPSEEK_API_KEY_FILE``;
    3. ``~/.config/mybot/deepseek_api_key`` (written by ``set_deepseek_key.sh``);
    4. ``~/.deepseek_api_key``;
    5. ``.secrets/deepseek_api_key`` and ``.env`` relative to the directory the
       launch was started from, so a gitignored workspace file also works.
    """
    value = os.environ.get(key_env, '').strip()
    if value:
        return value, f'${key_env}'

    candidates = []
    if key_file.strip():
        candidates.append(Path(key_file.strip()).expanduser())
    env_file = os.environ.get('DEEPSEEK_API_KEY_FILE', '').strip()
    if env_file:
        candidates.append(Path(env_file).expanduser())
    candidates += [Path(p).expanduser() for p in DEFAULT_KEY_FILES]

    for path in candidates:
        value = _key_from_file(path, key_env)
        if value:
            return value, str(path)
    return '', ''


class TaskParser(Node):
    """Generate a problem, parse it with the cloud LLM, validate the JSON."""

    def __init__(self) -> None:
        super().__init__('task_parser')

        # 2026-10-08: default was an absolute /home/yaowei/dev_ws/... path, so a
        # fresh clone could not find the problem generator. Resolve against the
        # user's home instead, and expand "~"/"$HOME" in whatever the YAML or the
        # CLI supplies so the shipped config stays machine independent. See
        # _resolved_generator_path() for the expansion.
        self.declare_parameter(
            'generator_path',
            os.path.join(os.path.expanduser('~'), 'dev_ws',
                         'TMSCQtest_x86_x64.bin'),
        )
        self.declare_parameter('generator_timeout_sec', 10.0)
        self.declare_parameter('problem_override', '')
        # The structured task is published transient-local, but a publisher that
        # exits immediately takes its durability cache with it. Keeping the node
        # alive briefly guarantees a late-starting supervisor still latches the
        # task. 0 restores the historical "parse once and exit" behaviour.
        # On failure the hold is clamped to [FAILURE_HOLD_MIN_SEC,
        # FAILURE_HOLD_MAX_SEC] so the reason is never lost either.
        self.declare_parameter('linger_sec', 0.0)

        self.declare_parameter('api_key_env', 'DEEPSEEK_API_KEY')
        # Optional explicit key file; empty falls back to DEFAULT_KEY_FILES.
        self.declare_parameter('api_key_file', '')
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
        # Source (path or env name) the key was last read from, logged once.
        self._key_source = None
        # Last status text, re-sent while lingering so a late-matching
        # supervisor or dashboard still receives it.
        self._last_status = ''
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
        self._last_status = text
        self.get_logger().info(f'Task status: {text}')
        self._publish(self._status_pub, text)

    def republish_status(self) -> None:
        """Re-send the last status quietly, for a subscriber that matched late."""
        if self._last_status:
            self._publish(self._status_pub, self._last_status)

    # ------------------------------------------------------------ generate
    def _resolved_generator_path(self) -> str:
        """Expand ~ and $HOME in generator_path so the config stays portable.

        The shipped config uses "$HOME/dev_ws/TMSCQtest_x86_x64.bin" instead of a
        hard-coded /home/<user>/... path (2026-10-08), because the absolute form
        only worked on the machine it was written on and a fresh clone silently
        failed with "Problem generator not found". ROS parameters are plain
        strings, so the shell never expands them for us and the expansion has to
        happen here.
        """
        raw = str(self.get_parameter('generator_path').value).strip()
        return os.path.expanduser(os.path.expandvars(raw))

    def _generate_problem(self) -> str | None:
        override = str(self.get_parameter('problem_override').value).strip()
        if override:
            self.get_logger().info('Using problem_override instead of the generator.')
            return override

        path = self._resolved_generator_path()
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
        api_key, source = resolve_api_key(
            key_env, str(self.get_parameter('api_key_file').value)
        )
        if not api_key:
            raise RuntimeError(
                f'No DeepSeek API key: ${key_env} is empty and none of the '
                'secret files were found. Run "ros2 run mybot '
                'set_deepseek_key.sh" once (stores it in '
                '~/.config/mybot/deepseek_api_key), or export '
                f'{key_env}=sk-... before launching.'
            )
        if source != self._key_source:
            self._key_source = source
            # Never log the key itself, only where it came from.
            self.get_logger().info(f'Using DeepSeek API key from {source}.')

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
    linger = float(node.get_parameter('linger_sec').value)
    try:
        exit_code = 0 if node.run_once() else 1
    except KeyboardInterrupt:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        return
    except Exception as exc:
        # An unhandled crash used to exit within milliseconds, so neither the
        # supervisor nor the Foxglove dashboard ever latched a FAILED status:
        # the run just sat in IDLE and looked like "the robot does not move".
        node.get_logger().error(f'Task parser crashed: {exc}')
        exit_code = 1
        try:
            node._status('FAILED:CRASH')
        except Exception:
            pass

    # Keep the transient-local samples alive so a supervisor or dashboard that
    # starts later still sees the task (on success) or the reason (on failure).
    if exit_code == 0:
        hold = linger
    else:
        hold = min(max(linger, FAILURE_HOLD_MIN_SEC), FAILURE_HOLD_MAX_SEC)
    if hold > 0.0:
        node.get_logger().info(
            f'Holding the last /competition/* status for {hold:.0f}s so late '
            'subscribers latch it.'
        )
        deadline = time.monotonic() + hold
        next_republish = time.monotonic() + STATUS_REPUBLISH_SEC
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.2)
            if time.monotonic() >= next_republish:
                next_republish = time.monotonic() + STATUS_REPUBLISH_SEC
                node.republish_status()

    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()
    sys.exit(exit_code)


if __name__ == '__main__':
    main()
