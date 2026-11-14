"""The decode-route vocabulary is closed and covers every status the frame engine can answer."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from src.domain.decode_route import DECODE_ROUTES, PREP_STATUSES, check_route, prep_status_route


ENGINE_DIR = Path(__file__).resolve().parents[1] / 'frame_engine'


def _returned_reasons() -> set[str]:
    """The status strings ``prep_geometry.route`` returns (all but ``gpu``)."""
    tree = ast.parse((ENGINE_DIR / 'prep_geometry.py').read_text())
    (func,) = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'route']
    reasons: set[str] = set()
    for node in ast.walk(func):
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Tuple):
            first = node.value.elts[0]
            literals = [c for c in ast.walk(first) if isinstance(c, ast.Constant)]
            reasons.update(str(c.value) for c in literals)
    return reasons - {'gpu', 'jpeg', 'other'}


def _engine_statuses() -> set[str]:
    """The literals ``engine.py`` answers a photo with: the status it passes to ``_failed`` and
    the reasons ``_decode`` returns."""
    tree = ast.parse((ENGINE_DIR / 'engine.py').read_text())
    found: set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == '_failed'
            and isinstance(node.args[1], ast.Constant)
        ):
            found.add(str(node.args[1].value))
        if isinstance(node, ast.FunctionDef) and node.name == '_decode':
            found.update(
                str(r.value.value)
                for r in ast.walk(node)
                if isinstance(r, ast.Return) and isinstance(r.value, ast.Constant)
            )
    return found


def test_every_status_the_engine_answers_has_a_route() -> None:
    engine = _engine_statuses()
    assert {'over_budget', 'gpu_memory', 'decoder'} <= engine, engine
    answered = _returned_reasons() | engine
    assert answered, 'the scan found no statuses'
    assert answered <= set(PREP_STATUSES), answered - set(PREP_STATUSES)
    for status in answered:
        assert prep_status_route(status) in DECODE_ROUTES


def test_the_names_are_unique_and_cpu_routes_are_prefixed() -> None:
    assert len(set(DECODE_ROUTES)) == len(DECODE_ROUTES)
    assert all(r == 'gpu' or r.startswith('cpu_') for r in DECODE_ROUTES)


@pytest.mark.parametrize('status', ['telepathy', 'ok', ''])
def test_a_status_outside_the_vocabulary_is_rejected(status: str) -> None:
    with pytest.raises(ValueError, match='unknown decode route'):
        prep_status_route(status)
    with pytest.raises(ValueError, match='unknown decode route'):
        check_route(f'cpu_{status}')


def test_the_call_failure_word_is_in_the_vocabulary() -> None:
    assert prep_status_route('call_failed') == 'cpu_call_failed'
