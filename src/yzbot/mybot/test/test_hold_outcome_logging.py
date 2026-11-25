"""The hold exits must say why they gave up.

A hold that never gets a verified release used to fail the leg silently: the
only trace was a TIMING line, so "the policy kept holding" and "the policy
released, but the corridor was never verified" looked identical. These tests
pin the four fields and the fact that every give-up exit prints them.
"""
import ast
import pathlib
import textwrap

SCRIPT = (pathlib.Path(__file__).resolve().parents[1]
          / 'scripts' / 'fixed_joint_pick_place.py')


def _source() -> str:
    return SCRIPT.read_text(encoding='utf-8')


def _method_source(name: str) -> str:
    source = _source()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            segment = ast.get_source_segment(source, node)
            assert segment is not None
            return segment
    raise AssertionError(f'{name} is missing from the executor')


def _node_type(name: str):
    namespace = {}
    exec(textwrap.dedent(_method_source(name)), namespace)  # noqa: S102
    return type('Node', (), {
        '_hold_outcome_text': namespace[name],
    })


def test_a_held_hold_reports_the_policy_still_holding():
    node = _node_type('_hold_outcome_text')()
    node._last_hold_report = (4.2, False, True,
                              'the mover still occupies the corridor')
    text = node._hold_outcome_text()

    assert 'held_for=4.2s' in text
    assert 'holds=True' in text
    assert 'cleared=False' in text
    assert 'reason=' in text and 'still occupies' in text


def test_a_released_but_unverified_hold_is_distinguishable():
    node = _node_type('_hold_outcome_text')()
    node._last_hold_report = (0.4, False, False, 'clear')
    text = node._hold_outcome_text()

    assert 'holds=False' in text
    assert 'cleared=False' in text
    assert "reason='clear'" in text


def test_a_missing_report_is_visible_not_silent():
    node = _node_type('_hold_outcome_text')()

    assert node._hold_outcome_text() == 'held_for=? holds=? cleared=? reason=?'


def _method_body(name: str) -> str:
    source = _source()
    start = source.index(f'def {name}(')
    end = source.index('\n    def ', start + 1)
    return source[start:end]


def test_every_give_up_exit_prints_the_outcome():
    navigate = _method_body('_navigate_once__timed')

    assert navigate.count('self._hold_outcome_text()') >= 5


def test_the_report_is_dropped_when_a_leg_starts():
    navigate = _method_body('_navigate_to')

    reset = navigate.index('self._wedged_retreat_used = False')
    assert navigate.index('self._last_hold_report = None') > reset
