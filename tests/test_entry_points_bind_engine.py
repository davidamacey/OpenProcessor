"""Every script entry point that starts an event loop binds a database engine first.

Only the API sets ``WEB_CONCURRENCY``, so ``async_session_factory`` fails closed in any other
process unless the entry point bound its own engine (``run_with_engine``, src/db/engine.py).
A new ``asyncio.run`` in these folders must wrap its coroutine in it, or be listed below with
the reason it never opens an async database session.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
FOLDERS = ('scripts/curation', 'scripts/db', 'scripts/opensearch')

NO_ASYNC_SESSION = {
    'scripts/curation/ingest_upload.py': 'talks to the API over HTTP',
    'scripts/curation/prune_exports.py': 'works on the export files only',
    'scripts/curation/prune_training_runs.py': 'works on the run files only',
    'scripts/db/rollback_to_060.py': 'restores a bundle with its own connections',
    'scripts/db/upgrade_from_060.py': 'builds its own owner-role engine and passes sessions in',
    'scripts/purge_face_data.py': 'OpenSearch and the face index files only',
}


def _loop_starts() -> list[tuple[str, int, ast.Call]]:
    files = [
        *(p for folder in FOLDERS for p in (ROOT / folder).rglob('*.py')),
        ROOT / 'scripts/purge_face_data.py',
    ]
    return [
        (str(path.relative_to(ROOT)), node.lineno, node)
        for path in sorted(files)
        for node in ast.walk(ast.parse(path.read_text()))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == 'run'
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == 'asyncio'
    ]


def _wrapped(call: ast.Call) -> bool:
    arg = call.args[0] if call.args else None
    return (
        isinstance(arg, ast.Call)
        and isinstance(arg.func, ast.Name)
        and arg.func.id == 'run_with_engine'
    )


def test_the_scan_finds_the_entry_points() -> None:
    assert len(_loop_starts()) >= 20


@pytest.mark.parametrize(
    ('path', 'line', 'call'),
    _loop_starts(),
    ids=[f'{path}:{line}' for path, line, _ in _loop_starts()],
)
def test_an_event_loop_start_binds_an_engine_or_is_listed_as_not_needing_one(
    path: str, line: int, call: ast.Call
) -> None:
    if path in NO_ASYNC_SESSION:
        assert not _wrapped(call), f'{path} is listed as not needing an engine; drop the entry'
    else:
        assert _wrapped(call), f'{path}:{line} starts a loop without run_with_engine'
