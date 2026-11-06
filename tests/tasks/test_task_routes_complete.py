"""Every registered task has a route and a job kind; no decorator picks its own queue.

Celery merges a route *under* the task's own options, so ``queue=`` in a decorator would
silently override the route table and stop it being the single source of truth. The same
check the worker runs at ``worker_ready`` is used here, so the runtime check and the test
cannot disagree. Pattern from OpenTranscribe ``tests/unit/test_task_routes_complete.py``.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from src.config.job_kinds import KIND_BY_TASK
from src.config.queue import TaskQueues
from src.config.task_routes import ROUTES
from src.tasks.runtime.gpu_app import task_route_problems


REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def worker_app(queue_env: None):
    from src.tasks.app import app

    app.loader.import_default_modules()
    return app


def test_the_registry_is_populated_so_the_check_is_not_vacuous(worker_app) -> None:
    assert 'op.infra.noop' in worker_app.tasks


def test_every_registered_task_has_a_route_and_a_kind(worker_app) -> None:
    assert task_route_problems(worker_app) == ([], [])


def test_the_noop_task_routes_to_jobs_and_has_a_kind() -> None:
    assert ROUTES['op.infra.noop'] == TaskQueues.JOBS
    assert KIND_BY_TASK['op.infra.noop'].queue == TaskQueues.JOBS


def test_no_task_pins_a_queue_on_its_object(worker_app) -> None:
    pinned = sorted(
        name
        for name in worker_app.tasks
        if not name.startswith('celery.') and getattr(worker_app.tasks[name], 'queue', None)
    )
    assert pinned == []


def _queue_keywords(source: str) -> list[int]:
    """Line numbers of ``.task(...)`` / ``shared_task(...)`` calls that pass ``queue=``."""
    lines = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, 'id', '')
        if name in {'task', 'shared_task'} and any(kw.arg == 'queue' for kw in node.keywords):
            lines.append(node.lineno)
    return lines


def test_no_decorator_in_the_tree_passes_queue() -> None:
    offenders = [
        f'{path.relative_to(REPO_ROOT)}:{line}'
        for root in ('src', 'docker', 'scripts')
        for path in (REPO_ROOT / root).rglob('*.py')
        for line in _queue_keywords(path.read_text(encoding='utf-8'))
    ]
    assert offenders == []


def test_the_scan_finds_a_planted_decorator_queue() -> None:
    assert _queue_keywords("@app.task(name='x', queue='jobs')\ndef f(): ...\n") == [1]
