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
    assert reopened.repaired_state()['status'] == 'interrupted'
    runner.check_resumable(reopened)
    runner.check_undoable(reopened)


@pytest.mark.asyncio
async def test_check_resumable_accepts_a_stale_import_without_writing(
    tmp_path, monkeypatch
) -> None:
    _h, store = await _completed(tmp_path, monkeypatch)
    store.job.update(status='running')
    _age_heartbeat(store, 600)
    before = store.job.state_file.stat().st_mtime_ns
    runner.check_resumable(store)
    assert store.job.read()['status'] == 'running'  # a reader never writes
    assert store.job.state_file.stat().st_mtime_ns == before


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
    assert job.served_state(frozenset({'running'}), error_prefix='x')['status'] == 'running'


def test_repair_leaves_a_fresh_heartbeat_and_a_terminal_state(tmp_path: Path) -> None:
    job = FileJob(tmp_path / 'j')
    job.write({'status': 'running'})
    job.touch_heartbeat()
    assert job.served_state(frozenset({'running'}), error_prefix='x')['status'] == 'running'
    job.write({'status': 'completed'})
    old = time.time() - 10 * HEARTBEAT_STALE_S
    os.utime(job.heartbeat_file, (old, old))
    assert job.served_state(frozenset({'running'}), error_prefix='x')['status'] == 'completed'


def test_repair_marks_a_stale_active_job_interrupted(tmp_path: Path) -> None:
    job = FileJob(tmp_path / 'j')
    job.write({'status': 'running'})
    job.touch_heartbeat()
    old = time.time() - 10 * HEARTBEAT_STALE_S
    os.utime(job.heartbeat_file, (old, old))
    state = job.served_state(frozenset({'running'}), error_prefix='thing')
    assert state['status'] == 'interrupted'
    assert state['error'] == 'thing heartbeat stale'
    assert job.read()['status'] == 'running'  # the view only: a reader never writes


def _stale_running_job(directory: Path) -> FileJob:
    job = FileJob(directory)
    job.write({'status': 'running'})
    job.touch_heartbeat()
    old = time.time() - 600
    os.utime(job.heartbeat_file, (old, old))
    return job


def test_a_stale_read_writes_nothing_so_a_claim_cannot_be_overwritten(tmp_path: Path) -> None:
    job = _stale_running_job(tmp_path / 'job')
    before = job.state_file.stat().st_mtime_ns

    seen = job.served_state(frozenset({'running'}), error_prefix='t')
    job.update(status='queued')  # a resume claims the job after the reader looked

    assert seen['status'] == 'interrupted'
    assert job.state_file.stat().st_mtime_ns != before
    assert job.read()['status'] == 'queued'
    assert not (job.directory / 'state.lock').exists()


def test_undo_keeps_a_box_whose_ledger_row_has_no_bbox() -> None:
    """Ledgers written before standalone items recorded their box have rows
    without ``bbox_norm``: undo must keep that box, not crash."""
    from src.domain.region_boxes import RegionBox
    from src.services.curation.dataset_import.undo import CANDIDATE_IMPORT, _import_boxes

    box = RegionBox(
        box_id='b1',
        bbox_norm=(0.1, 0.1, 0.5, 0.5),
        state='accepted',
        source=CANDIDATE_IMPORT,
        detector_version='imp1',
    )
    mine, edited = _import_boxes([box], 'imp1', [{'crop_id': 'c', 'box_id': 'b1'}])
    assert mine == []
    assert edited == [box]
