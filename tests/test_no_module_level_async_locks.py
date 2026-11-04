"""No module-level asyncio lock pretends to protect shared state.

The API runs many worker processes plus separate worker containers, so an in-process
``asyncio.Lock`` gives no exclusion across them (and binds to one event loop). State
shared between processes is protected by the datastore (idempotent writes, optimistic
concurrency); an in-process primitive is allowed only as a documented local throttle.
"""

from __future__ import annotations

import ast
from pathlib import Path


SRC = Path(__file__).resolve().parents[1] / 'src'
PRIMITIVES = {'Lock', 'Semaphore', 'BoundedSemaphore', 'Condition', 'Event'}

#: module -> name, each a documented local throttle (see its comment).
ALLOWED = {('services/curation/open_vocab_ingest.py', '_PASS_LOCK')}


def _module_level_primitives(path: Path) -> list[str]:
    found = []
    for node in ast.parse(path.read_text(encoding='utf-8')).body:
        if not isinstance(node, ast.Assign | ast.AnnAssign):
            continue
        value = node.value
        if not (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Attribute)
            and value.func.attr in PRIMITIVES
            and isinstance(value.func.value, ast.Name)
            and value.func.value.id == 'asyncio'
        ):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        found.extend(t.id for t in targets if isinstance(t, ast.Name))
    return found


def test_no_module_level_asyncio_primitive_outside_the_allowlist() -> None:
    offenders = [
        f'{path.relative_to(SRC)}:{name}'
        for path in sorted(SRC.rglob('*.py'))
        for name in _module_level_primitives(path)
        if (str(path.relative_to(SRC)), name) not in ALLOWED
    ]
    assert not offenders, offenders
