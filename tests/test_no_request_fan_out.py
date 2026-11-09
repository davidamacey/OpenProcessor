"""No request handler fans out: no ``asyncio.gather``, no ``create_task`` in the routers or in the
services only a handler reaches.

The ingest data plane is the one exemption: four modules, request-scoped, bounded by the batch,
awaited before the response, written as ``asyncio.TaskGroup`` (Python docs, "Task Groups": a
failing child cancels its siblings; ``gather`` leaves them running). The per-process helpers below
are not job work: they end with the request or the process, each with its reason.
"""

from __future__ import annotations

import ast
from pathlib import Path


SRC = Path(__file__).resolve().parents[1] / 'src'
FAN_OUT_CALLS = frozenset({'gather', 'create_task', 'ensure_future'})

# Services reached only from a request handler, beside every module under src/routers.
HANDLER_ONLY_SERVICES = (
    'services/curation/semantic_search.py',
    'services/curation/new_class_resolve.py',
    'services/curation/edit_undo.py',
    'services/curation/class_undo.py',
    'services/resource_links.py',
)

# The ingest data plane: TaskGroup fan-out is allowed here and only here.
INGEST_DATA_PLANE = (
    'services/curation/ingest_batch.py',
    'services/curation/ingest_detect.py',
    'services/curation/ingest_index.py',
    'services/visual_search/ingest_batch.py',
)

# Per-process helpers: a task that lives no longer than the request or the process.
ROUTER_HELPERS = {
    'routers/curation/pipeline_events.py': (
        'the dataset-stats aggregation shared by the open SSE streams of one process, shielded '
        'so a closing tab does not cancel it for the others; bounded by a TTL cache'
    ),
    'routers/curation/ingest_bulk_preview.py': (
        'the client-disconnect watcher of one preview request, cancelled in its finally block'
    ),
}

# A handler-only service whose fan-out another lane is replacing (one ``_msearch`` instead of two
# parallel requests); that lane deletes this entry (a stale entry fails below).
PENDING_REPLACEMENT = {
    'services/curation/semantic_search.py': 'kNN and unembedded count in parallel',
}

# Left in an exempt module until the core near-duplicate grouping leaves the ingest request; the
# lane that deletes the function deletes this entry (a stale entry fails below).
INGEST_LEFTOVER_GATHER = {
    'services/visual_search/ingest_batch.py': 'near-duplicate grouping inside the request',
}


def fan_out_calls(source: str) -> list[tuple[str, int]]:
    """``(name, line)`` of every gather, create_task or ensure_future call in ``source``."""
    found: list[tuple[str, int]] = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, 'id', '')
        if name in FAN_OUT_CALLS:
            found.append((name, node.lineno))
    return sorted(found, key=lambda call: call[1])


def _modules() -> list[str]:
    routers = sorted(str(p.relative_to(SRC)) for p in (SRC / 'routers').rglob('*.py'))
    return [*routers, *HANDLER_ONLY_SERVICES]


def test_no_handler_module_fans_out() -> None:
    offenders = []
    for rel in _modules():
        if rel in ROUTER_HELPERS or rel in PENDING_REPLACEMENT:
            continue
        calls = fan_out_calls((SRC / rel).read_text(encoding='utf-8'))
        offenders += [f'src/{rel}:{line} {name}' for name, line in calls]
    assert not offenders, f'a request handler fans out: {offenders}'


def test_the_ingest_data_plane_has_no_create_task_or_gather_outside_task_groups() -> None:
    for rel in INGEST_DATA_PLANE:
        names = [n for n, _ in fan_out_calls((SRC / rel).read_text(encoding='utf-8'))]
        assert 'ensure_future' not in names, rel
        gathers = names.count('gather')
        assert gathers == (1 if rel in INGEST_LEFTOVER_GATHER else 0), rel


def test_every_helper_and_leftover_entry_still_applies() -> None:
    for rel in (*ROUTER_HELPERS, *PENDING_REPLACEMENT):
        assert fan_out_calls((SRC / rel).read_text(encoding='utf-8')), f'{rel} no longer needs it'
    for rel in INGEST_LEFTOVER_GATHER:
        names = [n for n, _ in fan_out_calls((SRC / rel).read_text(encoding='utf-8'))]
        assert 'gather' in names, f'{rel} no longer needs it'


def test_the_scan_finds_a_planted_violation() -> None:
    planted = (
        'import asyncio\n'
        'async def handler(items):\n'
        '    await asyncio.gather(*(one(i) for i in items))\n'
        '    asyncio.get_running_loop().create_task(background())\n'
    )
    assert fan_out_calls(planted) == [('gather', 3), ('create_task', 4)]
