"""The pending list: tests that are red until a named slice of the Rust port.

``tests/pending_rust.txt`` holds one ``<node id> <slice>`` line per test. Both conftests (``tests/`` and
``frontend/e2e/``) call :func:`mark_pending`, which marks each listed test
``xfail(strict=True)``: a listed test that fails is an expected failure, and one that passes fails the
run (https://docs.pytest.org/en/stable/how-to/skipping.html#strict-parameter), so an entry cannot
outlive the slice that fixes it. A node id is relative to the repository root in both suites, and a
function id also covers every parametrization of it.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest


if TYPE_CHECKING:
    from collections.abc import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
PENDING_FILE = Path(__file__).with_name('pending_rust.txt')


def read_pending(path: Path | None = None) -> dict[str, str]:
    """Node id to slice. A malformed or duplicate line is an error, never a skipped one."""
    path = path or PENDING_FILE
    pending: dict[str, str] = {}
    for number, raw in enumerate(path.read_text(encoding='utf-8').splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith('#'):
            continue
        parts = line.rsplit(None, 1)
        if (
            len(parts) != 2
            or '::' not in parts[0]
            or any(c.isspace() for c in parts[0].partition('[')[0])
        ):
            raise ValueError(f'pending list {path}:{number}: want "<node id> <slice>", got {raw!r}')
        node_id, slice_id = parts
        if node_id in pending:
            raise ValueError(f'pending list {path}:{number}: {node_id} is listed twice')
        pending[node_id] = slice_id
    return pending


def root_relative_id(item: pytest.Item, root: Path) -> str:
    """The node id of ``item`` from ``root``, whatever rootdir the running suite has."""
    _, _, rest = item.nodeid.partition('::')
    path = Path(str(item.path)).resolve()
    try:
        shown = path.relative_to(root).as_posix()
    except ValueError:
        shown = item.nodeid.partition('::')[0]
    return f'{shown}::{rest}'


def _slice_for(node_id: str, pending: dict[str, str]) -> str | None:
    if node_id in pending:
        return pending[node_id]
    function_id = node_id.partition('[')[0]
    return pending.get(function_id)


def mark_pending(
    items: Iterable[pytest.Item], pending: dict[str, str], root: Path = REPO_ROOT
) -> list[str]:
    """Mark every listed item ``xfail(strict=True)``; returns the ids marked."""
    marked: list[str] = []
    for item in items:
        node_id = root_relative_id(item, root)
        slice_id = _slice_for(node_id, pending)
        if slice_id is None:
            continue
        item.add_marker(pytest.mark.xfail(strict=True, reason=f'pending {slice_id}'))
        marked.append(node_id)
    return marked
