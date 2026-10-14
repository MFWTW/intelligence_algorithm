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
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool, String

# Sibling module, resolved the same way the executor resolves obstacle_yield:
# `ros2 run` puts this script's own directory on sys.path. The start-button QoS
# contract must be identical to the supervisor's, so it is imported rather than
# copied.
from task_state_machine import start_button_fallback_qos


# Appended to the generator's argv to switch it into competition mode,
# where the problem changes every 30 minutes.
COMPETITION_CODE = '11470507'
SEED_PREFIX = '使用种子'


def parse_generator_output(stdout):
    """Return (problem, seed) from the generator's stdout.

    Practice mode prints exactly one line::
        题目: 每个衣柜有5件衣物;...

    Competition mode (the 11470507 argument) prints four::
        使用种子: 995389
        ------------------------------------------------------------
        题目: 每个书箱有5本书籍;...
        ------------------------------------------------------------

    Taking the first non-empty line, as this used to, returns the SEED line in
    competition mode and the model is then asked to parse "使用种子: 995389".
    Skip the seed and the separator rules, and join any wrapped continuation
    lines that follow the 题目 line.
    """
    problem_parts = []
    seed = None
    for raw in str(stdout or '').splitlines():
        line = raw.strip()
        if not line or set(line) == {'-'}:
            continue
        if line.startswith(SEED_PREFIX):
            _, _, value = line.partition(':')
            seed = value.strip() or None
            continue
        problem_parts.append(line)
        if problem_parts and len(problem_parts) > 1:
            break          # a second rule block would start a new problem
    problem = ' '.join(problem_parts) if problem_parts else None
    return problem, seed

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

        # 2026-10-08: default was an absolute /home/<user>/dev_ws/... path, so a
        # fresh clone could not find the problem generator. 2026-10-13: the
        # "$HOME/dev_ws/..." replacement was still a guess about WHERE the repo
        # was cloned, so the default is now just the file name and
        # _generator_candidates() finds the copy that ships with the workspace on
        # whatever machine this runs. An explicit path (YAML or CLI) still wins.
        self.declare_parameter('generator_path', 'TMSCQtest_x86_x64.bin')
        self.declare_parameter('generator_timeout_sec', 10.0)
        # The problem generator takes an extra argument to switch into
        # competition mode (the problem then changes every 30 minutes):
        #   <generator> 11470507
        # `competition_mode` appends that code; `generator_args` passes anything
        # else verbatim, so both are visible in the round's log line.
        # Default True so every path (launch, or a bare `ros2 run`) uses the
        # judges' 30-minute problem; task_parser.yaml also sets it true.
        self.declare_parameter('competition_mode', True)
        # A space-separated string, not an array: an array parameter has no
        # declaration form that survives both "a file supplies it" and "nothing
        # supplies it" (see the note on slow_zones in the executor). A string
        # has no such ambiguity.
        self.declare_parameter('generator_args', '')
        self.declare_parameter('problem_override', '')
        # The structured task is published transient-local, but a publisher that
        # exits immediately takes its durability cache with it. Keeping the node
        # alive briefly guarantees a late-starting supervisor still latches the
        # task. 0 restores the historical "parse once and exit" behaviour.
        # On failure the hold is clamped to [FAILURE_HOLD_MIN_SEC,
        # FAILURE_HOLD_MAX_SEC] so the reason is never lost either.
        self.declare_parameter('linger_sec', 0.0)
        # Gate the whole parse - generator AND LLM call - on the operator's start
        # button. The rule sheet hands the problem over at the start signal, so a
        # parse that ran at launch would answer a problem nobody has been given
        # yet. Measured cost when the button comes first: the generator plus the
        # LLM round trip took 6.2 s (4.6 s + 1.5 s) on 2026-10-14, and the
        # executor is pre-warmed meanwhile, so the press releases the queue and
        # the robot leaves almost immediately after the reply arrives.
        self.declare_parameter('wait_for_start', False)
        self.declare_parameter('start_topic', '/competition/start')

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
        # The competition publishes THREE mapping rules and the judge picks one
        # before timing, so all three are implemented and selected by
        # `mapping_rule`. They differ structurally, not just in constants:
        #   1. colour by LETTER (x=red, y=blue); zone by COUNT (>=3 -> A, else C)
        #   2. colour by COUNT (>=red_min_count -> red, else blue); zone by LETTER
        #      (x -> zone_x, y -> zone_y)
        #   3. colour by COMPARISON (larger count -> blue, smaller -> red);
        #      zone by COLOUR (blue -> zone_blue, red -> zone_red)
        # Rule 1 reuses color_x/color_y/zone_by_count; rules 2 and 3 need the
        # parameters below. Switchable while the stack runs:
        #   ros2 param set /task_parser mapping_rule 3
        self.declare_parameter('mapping_rule', 2)
        self.declare_parameter('red_min_count', 3)
        self.declare_parameter('zone_x', 'B')
        self.declare_parameter('zone_y', 'A')
        self.declare_parameter('zone_blue', 'B')
        self.declare_parameter('zone_red', 'C')
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
        # Latch the operator's press, including one made before this node was
        # ready: the same two readers the supervisor uses, because the Foxglove
        # button declares no QoS and a VOLATILE press never reaches a latched
        # reader (see start_button_fallback_qos in task_state_machine).
        self._start_requested = False
        if bool(self.get_parameter('wait_for_start').value):
            start_topic = str(self.get_parameter('start_topic').value)
            self.create_subscription(
                Bool, start_topic, self._on_start_button, result_qos
            )
            self.create_subscription(
                Bool, start_topic, self._on_start_button,
                start_button_fallback_qos()
            )

    # ---------------------------------------------------------------- start gate
    def _on_start_button(self, msg: Bool) -> None:
        if bool(msg.data) and not self._start_requested:
            self._start_requested = True
            self.get_logger().warning(
                'Start button pressed; running the generator and the LLM now.'
            )

    def wait_for_start(self, poll_sec: float = 0.2) -> bool:
        """Block until the operator starts the run, when configured to."""
        if not bool(self.get_parameter('wait_for_start').value):
            return True
        self._status('WAITING_FOR_START')
        self.get_logger().warning(
            'Parsing is gated on '
            f'{self.get_parameter("start_topic").value}; the problem is fetched '
            'only after the operator starts the run.'
        )
        while rclpy.ok() and not self._start_requested:
            rclpy.spin_once(self, timeout_sec=poll_sec)
        return self._start_requested

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
    def _generator_candidates(self) -> list[str]:
        """Every place the problem generator may live, best guess first.

        The binary travels with the repository (it sits in the workspace root), so
        the workspace root is DISCOVERED here rather than assumed to be ~/dev_ws:
        that assumption held only while every machine cloned to the same place
        (2026-10-13). Order: the configured value as given, $MYBOT_WS, the
        workspace this package was installed from, the working directory, and
        finally the old ~/dev_ws layout so existing setups keep working.
        """
        raw = str(self.get_parameter('generator_path').value).strip()
        given = os.path.expanduser(os.path.expandvars(raw))
        name = os.path.basename(given) or 'TMSCQtest_x86_x64.bin'
        candidates = [given] if given else []
        roots = []
        env_ws = os.environ.get('MYBOT_WS', '').strip()
        if env_ws:
            roots.append(Path(env_ws))
        try:
            share = Path(get_package_share_directory('mybot'))
            # <ws>/install/mybot/share/mybot -> <ws>
            if len(share.parents) >= 4:
                roots.append(share.parents[3])
        except Exception:                                  # noqa: BLE001
            # ament index does not know this package (running from a bare source
            # tree): the remaining roots still cover it.
            pass
        roots.append(Path.cwd())
        roots.append(Path.home() / 'dev_ws')
        for root in roots:
            candidates.append(str(root / name))
            if given and not os.path.isabs(given):
                # A configured relative sub-path ("tools/gen.bin") is relative to
                # each root as well, not only to the process CWD.
                candidates.append(str(root / given))
        seen, unique = set(), []
        for candidate in candidates:
            if candidate and candidate not in seen:
                seen.add(candidate)
                unique.append(candidate)
        return unique

    def _resolved_generator_path(self) -> str:
        """First candidate that actually exists, else the best guess.

        The shipped config names the generator without a directory
        ("TMSCQtest_x86_x64.bin"); the directory is found per machine. "~" and
        "$HOME" still expand, so an explicit absolute path in the YAML or on the
        command line keeps working - and keeps winning, because the configured
        value is the first candidate.
        """
        candidates = self._generator_candidates()
        for candidate in candidates:
            if os.path.isfile(candidate):
                # A bare relative filename exists in the current directory,
                # but subprocess.run([name]) searches PATH, not cwd. Return an
                # absolute executable path even for that first candidate.
                return os.path.abspath(candidate)
        return candidates[0] if candidates else ''

    def _generate_problem(self) -> str | None:
        override = str(self.get_parameter('problem_override').value).strip()
        if override:
            self.get_logger().info('Using problem_override instead of the generator.')
            return override

        path = self._resolved_generator_path()
        timeout = float(self.get_parameter('generator_timeout_sec').value)
        if not os.path.isfile(path):
            tried = self._generator_candidates()
            self.get_logger().error(
                f'Problem generator not found: {path}'
                + (f'  (also tried: {", ".join(tried[1:])})' if len(tried) > 1 else '')
                + '  -- set MYBOT_WS to the workspace root, or pass'
                  ' generator_path:=/abs/path.'
            )
            return None
        argv = [path] + str(
            self.get_parameter('generator_args').value or ''
        ).split()
        if bool(self.get_parameter('competition_mode').value)                 and COMPETITION_CODE not in argv:
            argv.append(COMPETITION_CODE)
        self.get_logger().warning(
            f'Problem generator argv: {argv}'
            + ('  (competition mode: the problem changes every 30 minutes)'
               if COMPETITION_CODE in argv else '  (practice mode)')
        )
        try:
            result = subprocess.run(
                argv, capture_output=True, text=True, timeout=timeout, check=False
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
        problem, seed = parse_generator_output(result.stdout)
        if seed is not None:
            # The seed identifies the 30-minute competition problem; record it so
            # a round can be tied back to what the judge announced.
            self.get_logger().warning(f'Generator seed: {seed}')
            self._publish(self._problem_pub, f'{problem}\n(种子 {seed})')
        if problem is None:
            self.get_logger().error('Problem generator produced no output.')
            return None
        return problem

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

    def _ask_model(self, problem: str, force_model: str = '') -> str | None:
        primary = str(self.get_parameter('model').value)
        fallback = str(self.get_parameter('fallback_model').value).strip()
        if force_model:
            # Used when the primary model's answer failed validation: a stronger
            # model is worth one more attempt before the round is lost.
            primary = force_model
            fallback = ''
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
        self.get_logger().warning(
            'All LLM attempts failed; checking deterministic offline fallback.'
        )
        return None


    @staticmethod
    def _oracle_totals(problem: str) -> dict | None:
        """Deterministically compute the totals from the generator templates.

        Returns None whenever the text does not match the known shapes, so an
        unfamiliar problem never produces a false rejection. When it does
        parse, the LLM's totals and x/y are checked against real arithmetic,
        which catches miscounting (e.g. reading 9 cats as 12).
        """
        # The measure word must be listed explicitly: "4本故事书" used to parse
        # the item as "本故事书" because 本 was not in the class, so the name never
        # matched a category and the oracle silently gave up (returned None) for
        # the whole 书箱/书籍 template family.
        measure = r'[件只条个本张把台辆朵棵份盒瓶支块片]'
        capacity = re.search(r'有\s*(\d+)\s*' + measure, problem)
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
        # Per-person amounts, so a later clause can inherit them ("小静和小文需求
        # 一样,但少一本故事书") instead of restating them.
        per_person: dict[str, dict[str, int]] = {}
        # Names are 2-3 characters here. {2,4} greedily swallowed the word
        # "需求" into the name ("小文需求"), so the reference never resolved.
        # "小华说和小芳一样" - the 说 belongs to the speaker, not the name, and
        # "小美也和小芳一样但是换成橙子" changes the KIND as well as inheriting the
        # amount; ignoring 换成 made the oracle report 5 apples where the题面 has
        # 4, which then vetoed the model's CORRECT answer (measured 2026-10-12,
        # seed 995433 -> TASK_PARSE:FAILED:VALIDATION and the round never ran).
        backref = re.compile(
            r'^([\u4e00-\u9fa5]{2,3}?)(?:也)?(?:说)?(?:和|跟)'
            r'([\u4e00-\u9fa5]{2,3})(?:的)?(?:需求)?一样')
        switch = re.compile(r'换成([A-Za-z\u4e00-\u9fa5]+)')
        doubled = re.compile(r'双倍|两倍|翻倍')
        # "少一本故事书" uses a Chinese numeral; "少1本" uses a digit.
        numerals = {'一': 1, '二': 2, '两': 2, '三': 3, '四': 4, '五': 5,
                    '六': 6, '七': 7, '八': 8, '九': 9, '十': 10}
        fewer = re.compile(
            r'少\s*(\d+|[\u4e00-\u9fa5])?\s*' + measure
            + r'?\s*([A-Za-z\u4e00-\u9fa5]+)')

        def _amount(text):
            if text is None:
                return 1
            if text.isdigit():
                return int(text)
            return numerals.get(text, 0)

        def _kind_of(word):
            word = word.lstrip(measure)
            for kind in (first, second):
                if word.startswith(kind):
                    return kind
            return None

        for clause in clauses:
            clause = clause.strip()
            if not clause or clause.startswith('设'):
                continue
            inherited = None
            ref = backref.match(clause)
            if ref is not None:
                # "小静和小文需求一样,但少一本故事书"
                source = per_person.get(ref.group(2))
                if source is None:
                    return None
                inherited = dict(source)
                rest = clause[ref.end():]
                shrink = fewer.search(rest)
                if shrink is not None:
                    kind = _kind_of(shrink.group(2))
                    step = _amount(shrink.group(1))
                    if kind is None or step <= 0:
                        return None
                    inherited[kind] -= step
                    if inherited[kind] < 0:
                        return None
                swap = switch.search(rest)
                if swap is not None:
                    target = _kind_of(swap.group(1))
                    if target is None:
                        return None
                    moved = sum(inherited.values())
                    inherited = {target: moved}
                elif switch.search(clause) is not None:
                    return None
                people_names = [ref.group(1)]
                items = [(str(v), k) for k, v in inherited.items()]
            else:
                # The name is LAZY on purpose. Greedy {2,4} matched "小文想" in
                # "小文想要4本书" (the verb list then matched the leftover "要"),
                # so the person was recorded as 小文想 and a later back-reference
                # to 小文 could not resolve. 想要 is listed before 要 so the lazy
                # name stops at the right place.
                head = re.match(
                    r'^([\u4e00-\u9fa5]{2,3}?(?:、[\u4e00-\u9fa5]{2,3}?)*)'
                    r'(?:都)?(?:需要|想要|领养|要|买|拿)',
                    clause,
                )
                if head is None:
                    return None
                people_names = head.group(1).split('、')
                rest = clause[head.end():]
                items = re.findall(
                    r'(\d+)\s*' + measure + r'?\s*([A-Za-z\u4e00-\u9fa5]+)',
                    rest,
                )
                if not items:
                    return None
                if doubled.search(rest):
                    # "他是双胞胎所以需要双倍的书" - the generator states the
                    # doubling in words, and the model does not always apply it.
                    items = [(str(int(n) * 2), w) for n, w in items]
            for number, word in items:
                kind = _kind_of(word)
                if kind is None:
                    return None
                totals[kind] += int(number) * len(people_names)
                for who in people_names:
                    per_person.setdefault(who, {})[kind] = (
                        per_person.setdefault(who, {}).get(kind, 0) + int(number)
                    )
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

    @classmethod
    def _offline_oracle_payload(cls, problem: str) -> dict | None:
        """Fallback only for generator templates the existing oracle parses.

        This is not a cached answer: amounts and capacity are calculated afresh
        from the actual题面. Unknown wording returns None and fails closed.
        The normal _check/_cross_check and count invariants still gate success.
        """
        oracle = cls._oracle_totals(problem)
        if oracle is None:
            return None
        return {
            'capacity': oracle['capacity'],
            'items': [{
                'who': '离线核算',
                'first': oracle['first'],
                'second': oracle['second'],
            }],
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

    def _repair(self, problem: str, reply: str, errors: list[str],
                model: str = '') -> dict | None:
        """Ask the model once to fix a reply that failed validation."""
        feedback = (
            '你上一次的回复未通过校验，错误如下：\n- '
            + '\n- '.join(errors)
            + f'\n\n上一次回复：\n{reply}'
            + '\n\n原始题目：\n' + problem
            + '\n\n请重新逐句抽取每个人需要的数量（“甲、乙都需要N个X”要拆成'
            + '甲、乙两行），不要自己求和，只输出修正后的 JSON。'
        )
        corrected = self._ask_model(feedback, force_model=model)
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
    def _zone_by_count(self, count: int) -> str:
        """Rule 1 only: look the count up in the published table."""
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
                'count %s exceeds the zone table %s; using the last entry.'
                % (count, table)
            )
            return table[-1]
        return table[count - 1]

    def _checked_zone(self, zone: object, what: str) -> str:
        zone = str(zone)
        if zone not in ZONES:
            raise ValueError(f'{what} must be in {sorted(ZONES)}, got {zone!r}')
        return zone

    def _rule1(self, x_count: int, y_count: int) -> list[dict]:
        """x=red, y=blue; a count >= 3 goes to A, the other colour to C."""
        color_x = str(self.get_parameter('color_x').value)
        color_y = str(self.get_parameter('color_y').value)
        if color_x not in COLORS or color_y not in COLORS:
            raise ValueError(
                f'color_x/color_y must be in {sorted(COLORS)}; '
                f'got {color_x!r}/{color_y!r}'
            )
        return [
            {'color': color_x, 'count': x_count,
             'zone': self._zone_by_count(x_count)},
            {'color': color_y, 'count': y_count,
             'zone': self._zone_by_count(y_count)},
        ]

    def _rule2(self, x_count: int, y_count: int) -> list[dict]:
        """A count >= red_min_count is red, otherwise blue; x->zone_x, y->zone_y."""
        minimum = int(self.get_parameter('red_min_count').value)
        if minimum < 1:
            raise ValueError(f'red_min_count must be >= 1, got {minimum}')
        zone_x = self._checked_zone(self.get_parameter('zone_x').value, 'zone_x')
        zone_y = self._checked_zone(self.get_parameter('zone_y').value, 'zone_y')
        color = lambda count: 'red' if count >= minimum else 'blue'  # noqa: E731
        return [
            {'color': color(x_count), 'count': x_count, 'zone': zone_x},
            {'color': color(y_count), 'count': y_count, 'zone': zone_y},
        ]

    def _rule3(self, x_count: int, y_count: int) -> list[dict]:
        """The larger count is blue and the smaller is red; blue->zone_blue, red->zone_red."""
        zone_blue = self._checked_zone(
            self.get_parameter('zone_blue').value, 'zone_blue')
        zone_red = self._checked_zone(
            self.get_parameter('zone_red').value, 'zone_red')
        if x_count == y_count:
            # The generator guarantees x + y = 5 with both >= 1, so this cannot
            # happen; refuse rather than guess a colour for a tie.
            raise ValueError(
                f'mapping rule 3 needs a strict ordering but x == y == {x_count}; '
                'the problem does not determine which colour is which.'
            )
        x_color = 'blue' if x_count > y_count else 'red'
        y_color = 'red' if x_color == 'blue' else 'blue'
        return [
            {'color': x_color, 'count': x_count,
             'zone': zone_blue if x_color == 'blue' else zone_red},
            {'color': y_color, 'count': y_count,
             'zone': zone_blue if y_color == 'blue' else zone_red},
        ]

    def rule_summary(self) -> str:
        """One line naming the active rule, for the operator and the topic."""
        rule = self._mapping_rule()
        if rule == 1:
            return ('规则1: x=红, y=蓝; 数值>=3 的颜色去 A 区, 另一色去 C 区'
                    f'  [执行: {self.get_parameter("color_x").value}->'
                    f'{self.get_parameter("color_y").value}, 表='
                    f'{list(self.get_parameter("zone_by_count").value)}]')
        if rule == 2:
            return ('规则2: xy 中数值>='
                    f'{int(self.get_parameter("red_min_count").value)}'
                    ' 的代表红色, 否则蓝色; x 去 '
                    f'{self.get_parameter("zone_x").value} 区, y 去 '
                    f'{self.get_parameter("zone_y").value} 区')
        return ('规则3: 蓝色去 '
                f'{self.get_parameter("zone_blue").value} 区, 红色去 '
                f'{self.get_parameter("zone_red").value} 区; '
                'xy 中数值大的是蓝色, 数值少的是红色')

    def _mapping_rule(self) -> int:
        try:
            rule = int(self.get_parameter('mapping_rule').value)
        except (TypeError, ValueError):
            raise ValueError(
                'mapping_rule must be 1, 2 or 3; got '
                f'{self.get_parameter("mapping_rule").value!r}'
            )
        if rule not in (1, 2, 3):
            raise ValueError(f'mapping_rule must be 1, 2 or 3, got {rule}')
        return rule

    def _build_tasks(self, payload: dict) -> list[dict]:
        """Map the solved counts onto colours and zones deterministically.

        The LLM is only trusted to extract the two counts; which colour each
        becomes and which zone it goes to is decided here, from the selected
        competition mapping rule, so a mis-read of the rule cannot reach the
        task.
        """
        rule = self._mapping_rule()
        x_count = int(payload['x'])
        y_count = int(payload['y'])
        # Class-level dispatch so the rule functions do not depend on bound
        # attribute lookup (which also keeps them unit-testable with a stub).
        builder = {1: TaskParser._rule1, 2: TaskParser._rule2,
                   3: TaskParser._rule3}[rule]
        tasks = [task for task in builder(self, x_count, y_count)
                 if task['count'] > 0]
        self.get_logger().warning(
            f'映射规则 {rule} 生效：x={x_count}, y={y_count} -> '
            + ', '.join(f"{t['count']}件{t['color']}去{t['zone']}区" for t in tasks)
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
        # Order matters: the first line a judge reads must be the COMPETITION
        # mapping rule, not the extraction prompt. `mapping_rules` is really the
        # LLM's extraction instruction (capacity/items/first/second) despite its
        # name, so it is demoted to a clearly-labelled trailing block.
        self._publish(
            self._rules_pub,
            '生效映射规则: ' + self.rule_summary()
            + '\n区域规则: ' + str(self.get_parameter('zone_rule_text').value)
            + '\n颜色映射: x='
            + str(self.get_parameter('color_x').value)
            + ', y=' + str(self.get_parameter('color_y').value)
            + '\n\n--- 以下为解析器内部抽取提示，不是比赛规则 ---\n'
            + str(self.get_parameter('mapping_rules').value),
        )

        self._status('PARSING')
        reply = self._ask_model(problem)
        if reply is None:
            offline = self._offline_oracle_payload(problem)
            if offline is not None:
                errors, _ = self._check(offline, problem)
                if not errors:
                    self.get_logger().warning(
                        'LLM unavailable; using validated offline arithmetic '
                        'for this recognised generator template.'
                    )
                    reply = json.dumps(offline, ensure_ascii=False)
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
            used_stronger = False
            repaired = self._repair(problem, reply, errors)
            if repaired is None or self._last_solved is None:
                # The same model at temperature 0 tends to repeat its mistake, so
                # spend one attempt on the stronger fallback model before losing
                # the whole round. Measured 2026-10-12 (seed 995411): the primary
                # answered x+y=4 and its repair answered x+y=6 for a problem the
                # generator guarantees to be 5, and the round never started.
                stronger = str(self.get_parameter('fallback_model').value).strip()
                if stronger and stronger != str(self.get_parameter('model').value):
                    self.get_logger().warning(
                        f'Repair still failed validation; retrying with '
                        f'{stronger}.')
                    self._status('REPAIRING')
                    repaired = self._repair(problem, reply, errors, model=stronger)
                if repaired is None or self._last_solved is None:
                    # Last resort: the generator templates are arithmetic, so the
                    # deterministic oracle can settle what the models could not.
                    # Measured 2026-10-12 (seed 995411): deepseek-chat extracted
                    # 4 for a clause that says "双胞胎所以需要双倍" and its repair
                    # was wrong too; the oracle computes it correctly, and without
                    # this path the whole round is lost before the robot moves.
                    offline = self._offline_oracle_payload(problem)
                    if offline is not None:
                        remaining, solved = self._check(offline, problem)
                        if not remaining:
                            self.get_logger().warning(
                                'Both models failed validation; using the '
                                'deterministic template arithmetic instead.')
                            repaired = json.dumps(offline, ensure_ascii=False)
                            self._last_solved = solved
                    if repaired is None or self._last_solved is None:
                        self._status('FAILED:VALIDATION')
                        return False
                    used_stronger = True
            payload = repaired
            solved = self._last_solved
            self._status('REPAIRED_STRONGER' if used_stronger else 'REPAIRED')
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
        if not node.wait_for_start():
            exit_code = 1
        else:
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
