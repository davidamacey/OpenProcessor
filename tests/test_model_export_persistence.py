"""Tests for model-export task-state durability (persistence-hardening
Gap 2).

``src.services.model_export`` predates the curation subsystem and isn't
curation-specific -- see ``docs/design/curation_design_rationale.md``.
Before this fix, ``export_tasks`` was a bare in-memory dict with no disk
backing at all; every test here proves the on-disk ``<task_id>.json`` file
(not the in-memory dict) is now the source of truth by clearing the
in-memory cache mid-test and confirming state survives.

Each test redirects ``EXPORT_TASK_DIR`` to a per-test ``tmp_path`` so
nothing here touches a real ``/jobs`` mount (same convention the curation
job-runner tests use for their own state dirs).
"""

from __future__ import annotations

import json

import pytest


@pytest.fixture(autouse=True)
def _isolated_export_tasks(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """Point EXPORT_TASK_DIR at a tmp dir and clear the in-memory cache
    before and after every test -- export_tasks is a module-level global
    shared across the whole test session otherwise."""
    monkeypatch.setenv('EXPORT_TASK_DIR', str(tmp_path / 'model_export'))
    from src.services import model_export

    model_export.export_tasks.clear()
    yield
    model_export.export_tasks.clear()


def _make_task(model_export, *, task_id: str = 'abc12345', status=None) -> str:
    tid = model_export.create_export_task(
        filename='my_model.pt',
        triton_name='my_model',
        model_info={'num_classes': 3, 'class_names': ['a', 'b', 'c'], 'end2end': False},
        formats=['trt'],
    )
    if status is not None:
        model_export.export_tasks[tid]['status'] = status
        model_export.persist_task(tid)
    return tid


def test_create_export_task_writes_a_task_file(tmp_path) -> None:
    from src.services import model_export

    task_id = _make_task(model_export)
    task_file = tmp_path / 'model_export' / f'{task_id}.json'
    assert task_file.exists()
    raw = json.loads(task_file.read_text())
    assert raw['task_id'] == task_id
    assert raw['triton_name'] == 'my_model'
    assert raw['status'] == 'pending'


def test_task_state_survives_a_simulated_restart(tmp_path) -> None:
    """Clear the in-memory dict (the "restart") and prove the task is
    still queryable with its correct status via the read-through cache."""
    from src.services import model_export

    task_id = _make_task(model_export)
    model_export.export_tasks[task_id]['status'] = model_export.ExportStatus.EXPORTING
    model_export.export_tasks[task_id]['progress'] = 42.0
    model_export.persist_task(task_id)

    # Simulate a process restart: the in-memory cache is gone, only the
    # file on disk survives.
    model_export.export_tasks.clear()
    assert model_export.export_tasks == {}

    task = model_export.get_export_task(task_id)
    assert task is not None
    assert task['status'] == 'exporting'
    assert task['progress'] == 42.0
    # get_export_task() must have repopulated the cache (read-through).
    assert task_id in model_export.export_tasks


def test_list_export_tasks_rebuilds_cache_from_disk(tmp_path) -> None:
    from src.services import model_export

    id_a = _make_task(model_export, task_id='aaa')
    id_b = _make_task(model_export, task_id='bbb')
    model_export.export_tasks.clear()

    tasks = model_export.list_export_tasks()
    ids = {t['task_id'] for t in tasks}
    assert id_a in ids
    assert id_b in ids


def test_load_tasks_from_disk_on_empty_dir_returns_zero(tmp_path) -> None:
    from src.services import model_export

    assert model_export.load_tasks_from_disk() == 0


def test_get_export_task_missing_id_returns_none(tmp_path) -> None:
    from src.services import model_export

    assert model_export.get_export_task('does-not-exist') is None
