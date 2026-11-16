"""Keeps the ``differential`` tests out of every default run, whatever ``-m`` expression it passes.

The offline suite is started with explicit expressions (``-m 'not live and not families ...'`` in the
Makefile, CI, CONTRIBUTING.md and CLAUDE.md), and an explicit ``-m`` replaces the one in pyproject.toml.
A test marked ``differential`` therefore runs only when the expression asks for the marker by name.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    import pytest

ASKED_FOR = re.compile(r'(?<!not )\bdifferential\b')


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if ASKED_FOR.search(config.getoption('markexpr') or ''):
        return
    kept = [i for i in items if i.get_closest_marker('differential') is None]
    dropped = [i for i in items if i.get_closest_marker('differential') is not None]
    if dropped:
        config.hook.pytest_deselected(items=dropped)
        items[:] = kept
