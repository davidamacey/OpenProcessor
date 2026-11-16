"""The marker ``differential`` is registered and kept out of the offline suite, as ``live`` is."""

from __future__ import annotations

import tomllib
from types import SimpleNamespace
from typing import Any

import pytest

from . import conftest as guard, families as fam


def pytest_options() -> dict[str, object]:
    pyproject = tomllib.loads((fam.REPO_ROOT / 'pyproject.toml').read_text(encoding='utf-8'))
    options: dict[str, object] = pyproject['tool']['pytest']['ini_options']
    return options


def test_the_marker_is_registered() -> None:
    markers = pytest_options()['markers']
    assert isinstance(markers, list)
    assert any(str(m).startswith('differential:') for m in markers)


def test_the_default_run_deselects_it() -> None:
    addopts = pytest_options()['addopts']
    assert isinstance(addopts, list)
    expression = addopts[addopts.index('-m') + 1]
    assert 'not differential' in str(expression)
    assert 'not live' in str(expression)


class Item:
    def __init__(self, marked: bool) -> None:
        self.marked = marked

    def get_closest_marker(self, name: str) -> object | None:
        return object() if self.marked and name == 'differential' else None


def collect(expression: str | None) -> tuple[list[Item], list[Item]]:
    """The items left and the items deselected by the guard, for a ``-m`` expression."""
    deselected: list[Item] = []
    hook = SimpleNamespace(pytest_deselected=lambda items: deselected.extend(items))
    config: Any = SimpleNamespace(getoption=lambda _name: expression, hook=hook)
    items = [Item(False), Item(True), Item(False)]
    guard.pytest_collection_modifyitems(config, items)  # type: ignore[arg-type]
    return items, deselected


@pytest.mark.parametrize(
    'expression',
    [
        None,
        '',
        'not live and not families',
        'not live and not families and not db and not native',
        'not live and not families and not db and not native and not differential',
        'live',
    ],
)
def test_an_explicit_expression_that_does_not_name_the_marker_still_deselects_it(
    expression: str | None,
) -> None:
    left, dropped = collect(expression)
    assert [i.marked for i in left] == [False, False]
    assert [i.marked for i in dropped] == [True]


@pytest.mark.parametrize(
    'expression', ['differential', 'differential and not live', 'live or differential']
)
def test_asking_for_the_marker_by_name_keeps_it(expression: str) -> None:
    left, dropped = collect(expression)
    assert [i.marked for i in left] == [False, True, False]
    assert dropped == []
