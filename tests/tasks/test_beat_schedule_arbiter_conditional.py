"""The ``gpu_arbiter`` beat entry exists only when ``OP_GPU_ARBITER_CONTAINERS`` is set, and it
is routed to the ``arbiter`` queue that only ``gpu-arbiter-worker`` consumes.

A beat message with ``expires`` is dropped by the worker that receives it, so the expiry holds
only for a queue that has a live consumer; without the overlay nothing consumes ``arbiter`` and
an entry would pile messages up (review RC-3, 03f lane 6A).
https://docs.celeryq.dev/en/stable/userguide/periodic-tasks.html
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.config.queue import TaskQueues
from src.config.task_routes import ROUTES
from src.tasks.beat_schedule import build_schedule


if TYPE_CHECKING:
    import pytest


TASK = 'op.periodic.gpu_arbiter'


def test_without_the_overlay_the_schedule_has_no_arbiter_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv('OP_GPU_ARBITER_CONTAINERS', raising=False)
    assert 'gpu_arbiter' not in build_schedule()


def test_an_empty_setting_is_the_same_as_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('OP_GPU_ARBITER_CONTAINERS', '  ')
    assert 'gpu_arbiter' not in build_schedule()


def test_with_the_overlay_the_entry_runs_every_15_seconds_and_expires_with_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv('OP_GPU_ARBITER_CONTAINERS', 'vlm,triton-server@2')
    entry = build_schedule()['gpu_arbiter']
    assert entry == {'task': TASK, 'schedule': 15.0, 'options': {'expires': 15}}


def test_the_tick_is_routed_to_the_arbiter_queue_not_periodic() -> None:
    assert ROUTES[TASK] == TaskQueues.ARBITER


def test_the_tick_is_a_registered_task_with_the_tick_limits(queue_env) -> None:
    from src.tasks.app import app

    app.loader.import_default_modules()
    task = app.tasks[TASK]
    assert (task.soft_time_limit, task.time_limit) == (120, 150)
