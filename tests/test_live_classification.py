"""The live suite's parallel/serial classification cannot rot (tests/live/_classes.py, README)."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from tests.live import _classes as classes


LIVE_DIR = Path(__file__).parent / 'live'
# Generic verbs are stack-acting only as `_compose.<verb>` or imported from `_compose`.
COMPOSE_ONLY = {'kill', 'stop', 'start', 'up'}


def live_modules() -> set[str]:
    return {p.stem for p in LIVE_DIR.glob('test_*.py')}


def stack_acting_uses(source: str) -> list[str]:
    """Stack-acting helpers a module's code uses, by name (see ``_classes.STACK_ACTING_NAMES``)."""
    found: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom):
            from_compose = (node.module or '').endswith('_compose')
            found.extend(
                alias.name
                for alias in node.names
                if alias.name in classes.STACK_ACTING_NAMES
                and (alias.name not in COMPOSE_ONLY or from_compose)
            )
        elif isinstance(node, ast.Name) and node.id in classes.STACK_ACTING_NAMES - COMPOSE_ONLY:
            found.append(node.id)
        elif isinstance(node, ast.arg) and node.arg in classes.STACK_ACTING_NAMES - COMPOSE_ONLY:
            found.append(node.arg)
        elif isinstance(node, ast.Attribute) and node.attr in classes.STACK_ACTING_NAMES:
            if node.attr not in COMPOSE_ONLY or (
                isinstance(node.value, ast.Name) and node.value.id == '_compose'
            ):
                found.append(node.attr)
        elif isinstance(node, ast.Call):
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, 'id', '')
            first = node.args[0] if node.args else None
            if (
                name == 'compose'
                and isinstance(first, ast.Constant)
                and first.value in classes.STACK_ACTING_COMPOSE_VERBS
            ):
                found.append(f'compose({first.value!r})')
        elif isinstance(node, ast.Constant) and node.value == '/__control':
            found.append('/__control')
    return found


def test_every_live_module_is_classified_exactly_once() -> None:
    sets = [classes.ISOLATED, classes.STACK_ACTING, classes.UNSURE]
    modules = live_modules()
    classified = set().union(*sets)
    assert modules - classified == set(), 'add to a set in tests/live/_classes.py and the README'
    assert classified - modules == set(), 'a classified module no longer exists'
    assert sum(len(s) for s in sets) == len(classified), 'a module is in two sets'


@pytest.mark.parametrize('module', sorted(classes.ISOLATED))
def test_an_isolated_module_uses_no_stack_acting_helper(module: str) -> None:
    uses = stack_acting_uses((LIVE_DIR / f'{module}.py').read_text())
    assert uses == [], f'{module} acts on the stack ({uses}); move it to STACK_ACTING'


@pytest.mark.parametrize('module', sorted(classes.STACK_ACTING))
def test_a_stack_acting_module_really_uses_one(module: str) -> None:
    source = (LIVE_DIR / f'{module}.py').read_text()
    if stack_acting_uses(source):
        return
    # Stack-wide assertions without a helper (all exports, the whole queue) are named here.
    assert module in {'test_live_models_promote', 'test_live_train_promote'}, module


def test_the_guard_sees_a_stack_acting_helper() -> None:
    src = 'from . import _compose\nfrom ._engine import engine_refuses\n_compose.kill("x")\n'
    assert 'engine_refuses' in stack_acting_uses(src)
    assert 'kill' in stack_acting_uses(src)
    assert stack_acting_uses('compose("up", "-d")') == ["compose('up')"]
    assert stack_acting_uses('compose("logs", "x")\nstop = 1\nstart = 2') == []
