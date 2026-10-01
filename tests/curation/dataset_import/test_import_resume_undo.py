"""Idempotency, crash-resume and undo of an import batch (W10.11, W10.12)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from curation.dataset_import.harness import Harness, map_all, write_yolo
from src.services.curation.dataset_import import runner
from src.services.curation.dataset_import.undo import undo_import


if TYPE_CHECKING:
    from pathlib import Path


class SimulatedCrash(BaseException):
    """Not an ``Exception``: nothing in the job may catch it."""


def _dataset(root: Path) -> None:
    write_yolo(
        root,
        names=['car', 'truck'],
        images={
            'a': ['0 0.3 0.3 0.2 0.2', '1 0.7 0.7 0.2 0.2'],
            'b': ['0 0.5 0.5 0.3 0.3'],
            'c': [],
        },
    )


def _snapshot(h: Harness) -> dict[str, tuple]:
    return {
        cid: (d['class_id'], d['class_name'], d['class_validated'], d['dataset_split'])
        for cid, d in h.items.items()
    }


@pytest.mark.asyncio
async def test_same_key_completed_import_is_reused_and_writes_nothing(
    tmp_path: Path, monkeypatch
) -> None:
    h = Harness(tmp_path, monkeypatch)
    h.registry.add_class('car')
    h.registry.add_class('truck')
    root = tmp_path / 'ds'
    _dataset(root)
    request = h.request(root, map_all(h.registry, 'car', 'truck'))
    store, _ = await h.run(request)
    before = {k: dict(v) for k, v in h.items.items()}
    writes = h.os.bulk_calls
    again, reused = runner.claim_import(request, h.prepare(request))
    assert reused
    assert again.import_id == store.import_id
    assert h.os.bulk_calls == writes
    assert {k: dict(v) for k, v in h.items.items()} == before


@pytest.mark.asyncio
async def test_only_one_import_runs_per_project(tmp_path: Path, monkeypatch) -> None:
    h = Harness(tmp_path, monkeypatch)
    h.registry.add_class('car')
    h.registry.add_class('truck')
    root = tmp_path / 'ds'
    _dataset(root)
    request = h.request(root, map_all(h.registry, 'car', 'truck'))
    store, _ = runner.claim_import(request, h.prepare(request))
    other = tmp_path / 'ds2'
    write_yolo(other, names=['car'], images={'z': ['0 0.5 0.5 0.2 0.2']})
    req2 = h.request(other, map_all(h.registry, 'car'))
    with pytest.raises(runner.ImportBusyError) as busy:
        runner.claim_import(req2, h.prepare(req2))
    assert busy.value.import_id == store.import_id


@pytest.mark.asyncio
async def test_crash_then_resume_equals_an_uninterrupted_run(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv('OP_DATASET_IMPORT_CHUNK', '1')
    h = Harness(tmp_path / 'crash', monkeypatch, root=tmp_path)
    h.registry.add_class('car')
    h.registry.add_class('truck')
    root = tmp_path / 'ds'
    _dataset(root)
    request = h.request(root, map_all(h.registry, 'car', 'truck'))

    async def die_after_first_chunk(idx: int) -> None:
        if idx == 0:
            raise SimulatedCrash

    with pytest.raises(SimulatedCrash):
        await h.run(request, after_chunk=die_after_first_chunk)
    store = h.last_store
    assert store is not None
    assert store.job.read()['status'] == 'running'
    # The process died: its heartbeat goes stale, the next startup repairs it.
    store.job.heartbeat_file.unlink()
    assert runner.reconcile_orphaned_jobs() is True
    assert store.job.read()['status'] == 'interrupted'
    assert set(store.chunks_done()) == {0}

    await h.resume(store, request)
    assert store.job.read()['status'] == 'completed'

    clean = Harness(tmp_path / 'clean', monkeypatch, root=tmp_path)
    clean.registry.add_class('car')
    clean.registry.add_class('truck')
    clean_store, _ = await clean.run(clean.request(root, map_all(clean.registry, 'car', 'truck')))
    assert _snapshot(h) == _snapshot(clean)
    assert store.job.read()['report']['items_created'] == 3
    # The write-ahead ledger keeps the truth across the re-run.
    actions = {i['action'] for row in store.ledger_rows() for i in row.get('items') or []}
    assert actions == {'created'}
    assert {r['status'] for r in store.ledger_rows()} == {'ok'}
    assert {r['status'] for r in clean_store.ledger_rows()} == {'ok'}


@pytest.mark.asyncio
async def test_resume_consumes_the_persisted_mapping_not_the_resumers_registry(
    tmp_path: Path, monkeypatch
) -> None:
    """Cold worker: the resuming process has a different registry state (a
    class the API created at start is missing from it) and must still use
    the class ids the START resolved."""
    monkeypatch.setenv('OP_DATASET_IMPORT_CHUNK', '1')
    h = Harness(tmp_path / 'w', monkeypatch, root=tmp_path)
    h.registry.add_class('car')
    root = tmp_path / 'ds'
    _dataset(root)
    request = h.request(root, map_all(h.registry, 'car', 'truck'))

    async def die(idx: int) -> None:
        raise SimulatedCrash

    with pytest.raises(SimulatedCrash):
        await h.run(request, after_chunk=die)
    store = h.last_store
    assert store is not None
    truck_id = store.read_mapping()['created_classes']['truck']
    store.job.heartbeat_file.unlink()
    runner.reconcile_orphaned_jobs()
    # A cold worker: nothing but the registry file survives; a stray class
    # claims the id the interrupted run's 'truck' would take next.
    await h.resume(store, request)
    truck_docs = [d for d in h.items.values() if d['class_name'] == 'truck']
    assert truck_docs
    assert {d['class_id'] for d in truck_docs} == {truck_id}
    assert h.registry.get(truck_id).class_name == 'truck'


@pytest.mark.asyncio
async def test_resume_refuses_when_the_dataset_changed(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv('OP_DATASET_IMPORT_CHUNK', '1')
    h = Harness(tmp_path / 'w', monkeypatch, root=tmp_path)
    h.registry.add_class('car')
    h.registry.add_class('truck')
    root = tmp_path / 'ds'
    _dataset(root)
    request = h.request(root, map_all(h.registry, 'car', 'truck'))

    async def die(idx: int) -> None:
        raise SimulatedCrash

    with pytest.raises(SimulatedCrash):
        await h.run(request, after_chunk=die)
    store = h.last_store
    assert store is not None
    store.job.heartbeat_file.unlink()
    runner.reconcile_orphaned_jobs()
    (root / 'labels/train/b.txt').write_text('1 0.5 0.5 0.3 0.3\n')
    with pytest.raises(runner.DatasetChangedError):
        await h.resume(store, request)


@pytest.mark.asyncio
async def test_undo_removes_only_import_labels_and_keeps_human_edits(
    tmp_path: Path, monkeypatch
) -> None:
    h = Harness(tmp_path, monkeypatch)
    h.registry.add_class('car')
    root = tmp_path / 'ds'
    _dataset(root)
    store, _ = await h.run(h.request(root, map_all(h.registry, 'car', 'truck')))
    created_truck = store.read_mapping()['created_classes']['truck']
    cars = [d for d in h.items.values() if d['class_name'] == 'car']
    kept = cars[0]['crop_id']
    h.items[kept].update(
        {'class_id': 99, 'class_name': 'bus', 'class_source': 'human', 'label_source': 'human'}
    )
    ctx = h.undo_context(store.import_id)
    dry = await undo_import(ctx, store, dry_run=True, created_classes={'truck': created_truck})
    assert (dry.items_deleted, dry.items_kept_human_edited) == (2, 1)
    assert dry.classes_deprecated == ['truck']
    assert len(h.items) == 3  # a dry run changes nothing
    assert h.registry.get(created_truck).deprecated is False

    report = await undo_import(ctx, store, dry_run=False, created_classes={'truck': created_truck})
    assert (report.items_deleted, report.items_kept_human_edited) == (2, 1)
    assert list(h.items) == [kept]
    survivor = h.items[kept]
    assert (survivor['class_name'], survivor['class_source']) == ('bus', 'human')
    assert survivor['import_ids'] == []
    assert report.classes_deprecated == ['truck']
    assert h.registry.get(created_truck).deprecated is True
    assert report.images_deleted == 2  # b and c hold nothing now; a still holds the kept car
    again = await undo_import(ctx, store, dry_run=False, created_classes={})
    assert (again.items_deleted, again.items_restored, again.images_deleted) == (0, 0, 0)


@pytest.mark.asyncio
async def test_a_crash_inside_a_chunk_keeps_created_truthful_for_undo(
    tmp_path: Path, monkeypatch
) -> None:
    """The process dies after an image's docs were written but before its
    ledger row was completed. The resumed run sees those docs as existing,
    so it would call them ``noop``; the write-ahead row must keep them
    ``created`` or undo would leave them behind."""
    h = Harness(tmp_path / 'w', monkeypatch, root=tmp_path)
    h.registry.add_class('car')
    h.registry.add_class('truck')
    root = tmp_path / 'ds'
    _dataset(root)
    request = h.request(root, map_all(h.registry, 'car', 'truck'))

    real = h.service.index_items
    calls = 0

    async def crash_after_the_second_images_write(*args, **kwargs):
        nonlocal calls
        outcome = await real(*args, **kwargs)
        calls += 1
        if calls == 2:
            raise SimulatedCrash
        return outcome

    monkeypatch.setattr(h.service, 'index_items', crash_after_the_second_images_write)
    with pytest.raises(SimulatedCrash):
        await h.run(request)
    store = h.last_store
    assert store is not None
    pending = [r for r in store.ledger_rows() if r['status'] == 'pending']
    assert len(pending) == 1

    monkeypatch.setattr(h.service, 'index_items', real)
    store.job.heartbeat_file.unlink()
    runner.reconcile_orphaned_jobs()
    await h.resume(store, request)
    assert {r['status'] for r in store.ledger_rows()} == {'ok'}
    actions = [i['action'] for r in store.ledger_rows() for i in r['items']]
    assert actions == ['created'] * 3

    ctx = h.undo_context(store.import_id)
    report = await undo_import(ctx, store, dry_run=False)
    assert report.items_deleted == 3
    assert h.items == {}
