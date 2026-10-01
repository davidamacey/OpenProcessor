"""A worker killed and restarted inside the liveness window leaves an import
``running`` with a heartbeat that then goes stale: it must be resumable and
undoable, not stuck. An undo excludes a concurrent import like a start does."""

from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING, Any

import pytest

from curation.dataset_import.harness import Harness, map_all, write_yolo
from src.services.curation.dataset_import import runner
from src.services.curation.dataset_import.store import ImportStore, imports_root, open_store
from src.services.curation.file_job import HEARTBEAT_STALE_S, FileJob


if TYPE_CHECKING:
    from pathlib import Path


async def _completed(tmp_path: Path, monkeypatch: Any) -> tuple[Harness, ImportStore]:
    h = Harness(tmp_path, monkeypatch)
    h.registry.add_class('car')
    root = tmp_path / 'ds'
    write_yolo(root, names=['car'], images={'a': ['0 0.3 0.3 0.2 0.2']})
    store, _ = await h.run(h.request(root, map_all(h.registry, 'car')))
    return h, store


def _age_heartbeat(store: ImportStore, seconds: float) -> None:
    old = time.time() - seconds
    os.utime(store.job.heartbeat_file, (old, old))


@pytest.mark.asyncio
async def test_a_stale_running_import_is_resumable_and_undoable(tmp_path, monkeypatch) -> None:
    _h, store = await _completed(tmp_path, monkeypatch)
    store.job.update(status='running')
    _age_heartbeat(store, 600)
    reopened = open_store(store.import_id)
    assert reopened is not None
    assert reopened.job.read()['status'] == 'interrupted'
    runner.check_resumable(reopened)
    runner.check_undoable(reopened)


@pytest.mark.asyncio
async def test_check_resumable_itself_repairs_a_stale_import(tmp_path, monkeypatch) -> None:
    _h, store = await _completed(tmp_path, monkeypatch)
    store.job.update(status='running')
    _age_heartbeat(store, 600)
    runner.check_resumable(store)
    assert store.job.read()['status'] == 'interrupted'


@pytest.mark.asyncio
async def test_a_live_running_import_is_neither_resumable_nor_undoable(
    tmp_path, monkeypatch
) -> None:
    _h, store = await _completed(tmp_path, monkeypatch)
    store.job.update(status='running')
    store.job.touch_heartbeat()
    with pytest.raises(runner.ImportNotResumableError):
        runner.check_resumable(store)
    with pytest.raises(runner.ImportNotUndoableError):
        runner.check_undoable(store)
    assert store.job.read()['status'] == 'running'


@pytest.mark.asyncio
async def test_undo_refuses_while_another_import_is_live(tmp_path, monkeypatch) -> None:
    _h, store = await _completed(tmp_path, monkeypatch)
    other = ImportStore(imports_root() / 'imp_20990101T000000_deadbeef')
    other.job.write({'status': 'running'})
    other.job.touch_heartbeat()
    with pytest.raises(runner.ImportBusyError) as busy:
        runner.claim_undo(store)
    assert busy.value.import_id == other.import_id
    assert store.job.read()['status'] == 'completed'


@pytest.mark.asyncio
async def test_undo_claim_marks_the_import_undoing_when_nothing_else_is_live(
    tmp_path, monkeypatch
) -> None:
    _h, store = await _completed(tmp_path, monkeypatch)
    runner.claim_undo(store)
    state = store.job.read()
    assert (state['status'], state['mode']) == ('undoing', 'undo')
    with pytest.raises(runner.ImportNotUndoableError):
        runner.claim_undo(store)


def test_repair_leaves_a_job_that_has_not_ticked_yet(tmp_path: Path) -> None:
    job = FileJob(tmp_path / 'j')
    job.write({'status': 'running'})
    assert job.repair_if_stale(frozenset({'running'}), error_prefix='x')['status'] == 'running'


def test_repair_leaves_a_fresh_heartbeat_and_a_terminal_state(tmp_path: Path) -> None:
    job = FileJob(tmp_path / 'j')
    job.write({'status': 'running'})
    job.touch_heartbeat()
    assert job.repair_if_stale(frozenset({'running'}), error_prefix='x')['status'] == 'running'
    job.write({'status': 'completed'})
    old = time.time() - 10 * HEARTBEAT_STALE_S
    os.utime(job.heartbeat_file, (old, old))
    assert job.repair_if_stale(frozenset({'running'}), error_prefix='x')['status'] == 'completed'


def test_repair_marks_a_stale_active_job_interrupted(tmp_path: Path) -> None:
    job = FileJob(tmp_path / 'j')
    job.write({'status': 'running'})
    job.touch_heartbeat()
    old = time.time() - 10 * HEARTBEAT_STALE_S
    os.utime(job.heartbeat_file, (old, old))
    state = job.repair_if_stale(frozenset({'running'}), error_prefix='thing')
    assert state['status'] == 'interrupted'
    assert state['error'] == 'thing heartbeat stale'
    assert job.read()['status'] == 'interrupted'
