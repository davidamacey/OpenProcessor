"""Automated class writers never overwrite a class state that changed after
they read it.

Live case (crop 5ae9c511...): a human undo restored the classifier's label;
17 s later the VLM label-batch write — decided on the state read before the
undo — replaced it and moved the item to another cluster. Every VLM /
auto-label class writer now records the class state it decided on and its
OCC merger drops the class write when the write-time state differs
(:mod:`src.services.curation.class_write_guard`).

Each test injects the human write between the writer's read and its
write-time re-read (the ``mget`` the OCC bulk writer issues), against the
query-evaluating OpenSearch fake.
"""

from __future__ import annotations

import io
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock

import pytest
from PIL import Image

from curation.query_fakes import QueryFakeOpenSearch
from src.clients.curation_opensearch import ClassRegistry
from src.config import CurationConfig, get_curation_config, get_region_fields
from src.services.curation.class_write_guard import (
    CLASS_GUARD_SOURCE_FIELDS,
    ClassWriteGuard,
    class_state_token,
)
from src.services.labeling.vlm_labeler import VlmClassPrediction


if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


ITEMS = get_curation_config().items_index
F = get_region_fields()

UNDO_ENTRY = {'writer': 'human:unlabel_crop', 'at': '2026-09-24T10:56:34+00:00'}


def _restored_by_undo(doc: dict[str, Any]) -> None:
    """A human undo puts the classifier label back (same as the pre-discard
    state) and appends its own history entry."""
    doc.update(
        class_id=12,
        class_name='coupe',
        class_source='det_model',
        label_source='det_model',
        class_validated=False,
        cluster_id=12,
    )
    doc['class_id_history'] = [*(doc.get('class_id_history') or []), dict(UNDO_ENTRY)]


def _discarded(crop_id: str) -> dict[str, Any]:
    return {
        'crop_id': crop_id,
        'image_path': f'/data/{crop_id}.jpg',
        'bbox_norm': [0.1, 0.1, 0.5, 0.5],
        'updated_at': '2026-09-01T00:00:00+00:00',
        'class_id': None,
        'class_source': None,
        'class_validated': False,
        'class_id_history': [{'writer': 'human:discard_crop', 'at': '2026-09-24T10:56:31+00:00'}],
    }


def _plain(crop_id: str) -> dict[str, Any]:
    return {**_discarded(crop_id), 'class_id_history': []}


def _hook_mget(fake: QueryFakeOpenSearch, mutate: Callable[[], None]) -> None:
    """Run ``mutate`` once, right before the first write-time ``mget``."""
    orig = fake.mget
    fired = False

    async def _mget(**kw: Any) -> dict[str, Any]:
        nonlocal fired
        if not fired:
            fired = True
            mutate()
        return await orig(**kw)

    fake.mget = _mget  # type: ignore[method-assign]


# ---------------------------------------------------------------- the guard


def test_token_sees_a_round_trip() -> None:
    doc = {'class_id': 3, 'class_source': 'det_model', 'class_id_history': []}
    before = class_state_token(doc)
    doc['class_id_history'] = [{'writer': 'human:label_crop'}, {'writer': 'human:unlabel_crop'}]
    assert class_state_token(doc) != before


def test_guard_refuses_unread_and_locked_items() -> None:
    guard = ClassWriteGuard('w')
    doc = {'class_id': 3, 'class_source': 'det_model'}
    assert not guard.allows('never-read', doc)
    guard.remember('a', doc)
    assert guard.allows('a', dict(doc))
    assert not guard.allows('a', {**doc, 'class_validated': True})
    assert not guard.allows('a', {**doc, 'class_source': 'human'})


# ---------------------------------------------------------------- VLM label_batch


@pytest.mark.asyncio
async def test_label_batch_skips_item_restored_during_vlm_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import src.routers.curation.vlm as vlm_mod
    from src.routers.curation.vlm import VlmLabelBatchRequest

    buf = io.BytesIO()
    Image.new('RGB', (32, 32), (1, 2, 3)).save(buf, format='JPEG')
    for cid in ('undone', 'plain'):
        (tmp_path / f'{cid}.jpg').write_bytes(buf.getvalue())
    monkeypatch.setattr(
        vlm_mod, 'get_curation_config', lambda: SimpleNamespace(crop_cache_dir=tmp_path)
    )
    reg = ClassRegistry(path=tmp_path / 'class_registry.json')
    reg.add_class('sportscar')
    monkeypatch.setattr(vlm_mod, 'get_class_registry', lambda: reg)
    monkeypatch.setattr(vlm_mod, '_default_pack_name', AsyncMock(return_value=None))

    fake = QueryFakeOpenSearch({ITEMS: {'undone': _discarded('undone'), 'plain': _plain('plain')}})

    class _Labeler:
        async def label_or_propose_batch(self, crops: list[Any], _names: list[str]) -> list[Any]:
            _restored_by_undo(fake.docs(ITEMS)['undone'])
            return [
                VlmClassPrediction(img_id=c.img_id, class_name='sportscar', confidence='high')
                for c in crops
            ]

    monkeypatch.setattr(vlm_mod, '_get_vlm_labeler', lambda *_a, **_k: _Labeler())

    await vlm_mod.vlm_label_batch(VlmLabelBatchRequest(crop_ids=['undone', 'plain']), fake)

    docs = fake.docs(ITEMS)
    assert (docs['undone']['class_id'], docs['undone']['class_source']) == (12, 'det_model')
    assert docs['undone']['cluster_id'] == 12
    assert docs['plain']['class_name'] == 'sportscar'
    assert docs['plain']['class_source'] == 'vlm'


# ---------------------------------------------------------------- auto-label VLM stage


@pytest.mark.asyncio
async def test_pipeline_vlm_stage_skips_item_restored_during_vlm_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from curation.test_pipeline import _FakeClassEntry, _FakeRegistry
    from src.routers.curation import pipeline, pipeline_health
    from src.services.curation import image_serving
    from src.services.curation.autolabel import selection

    fake = QueryFakeOpenSearch({ITEMS: {'undone': _discarded('undone'), 'plain': _plain('plain')}})

    class _Labeler:
        model = 'fake-vlm'
        _pack = None

        async def label_or_propose_batch(self, crops: list[Any], *_a: Any, **_k: Any) -> list[Any]:
            _restored_by_undo(fake.docs(ITEMS)['undone'])
            return [
                VlmClassPrediction(img_id=c.img_id, class_name='sportscar', confidence='high')
                for c in crops
            ]

    monkeypatch.setattr(selection, 'classifier_class_sources', lambda: frozenset({'det_model'}))
    monkeypatch.setattr(pipeline, '_get_vlm_labeler', lambda _pack=None: _Labeler())
    monkeypatch.setattr(pipeline, 'resolve_run_prompt_pack', AsyncMock(return_value=None))
    monkeypatch.setattr(pipeline_health, 'pipeline_health_snapshot', AsyncMock(return_value={}))
    monkeypatch.setattr(
        'src.routers.curation.get_class_registry',
        lambda: _FakeRegistry([_FakeClassEntry(7, 'sportscar')]),
    )
    monkeypatch.setattr(
        'src.services.labeling.vlm_labeler.format_class_catalog', lambda *_a, **_k: ''
    )
    monkeypatch.setattr(image_serving, 'resolve_crop_root', lambda _p: '/data')
    monkeypatch.setattr(image_serving, 'resolve_safe_path', lambda p, _r: p)
    monkeypatch.setattr(
        image_serving, 'THUMBNAIL_CACHE', SimpleNamespace(get_or_compute=lambda *_a, **_k: b'jpeg')
    )

    await pipeline.pipeline_auto_label(opensearch=fake, train_clusters=False, run_vlm=True)

    docs = fake.docs(ITEMS)
    assert (docs['undone']['class_id'], docs['undone']['class_source']) == (12, 'det_model')
    assert docs['plain']['class_id'] == 7
    assert docs['plain']['class_source'] == 'vlm'


# ---------------------------------------------------------------- registry reclassify


@pytest.mark.asyncio
async def test_registry_reclassify_skips_item_touched_after_read(tmp_path: Path) -> None:
    from src.services.curation.registry_reclassify import UnmatchedLabelSource, reclassify_unmatched
    from src.services.labeling.vlm_prompts import GENERIC_ITEM_PACK

    index = 'test_items'
    reg = ClassRegistry(path=tmp_path / 'class_registry.json')
    reg.add_class('widget')
    base = {'class_source': 'vlm_unmatched', 'label_source': 'vlm', 'vlm_raw_label': 'widget'}
    fake = QueryFakeOpenSearch(
        {index: {'t': {'crop_id': 't', **base}, 'u': {'crop_id': 'u', **base}}}
    )

    def _label_then_undo() -> None:
        # A human labeled 't' and undid it after the reclassifier read it:
        # same class fields, two new history entries.
        fake.docs(index)['t']['class_id_history'] = [
            {'writer': 'human:label_crop', 'at': '2026-09-24T10:00:00+00:00'},
            dict(UNDO_ENTRY),
        ]

    _hook_mget(fake, _label_then_undo)
    await reclassify_unmatched(
        fake,
        source=UnmatchedLabelSource('vlm'),
        registry=reg,
        pack=GENERIC_ITEM_PACK,
        config=CurationConfig(items_index=index),
        dry_run=False,
    )
    docs = fake.docs(index)
    assert docs['t']['class_source'] == 'vlm_unmatched'
    assert docs['u']['class_source'] == 'vlm_reclassified'


# ---------------------------------------------------------------- cluster auto-promote


@pytest.mark.asyncio
async def test_auto_promote_skips_item_restored_to_another_class(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.services.curation.clustering import orchestrator

    _ = orchestrator.ITEMS_INDEX
    from src.services.curation.clustering import auto_promote as ap

    monkeypatch.setattr(ap, 'classifier_class_sources', lambda: frozenset({'det_model'}))
    cluster = 10001
    members = {
        f'm{i}': {
            'crop_id': f'm{i}',
            'cluster_id': cluster,
            'class_name': 'coupe',
            'class_source': 'det_model',
            'class_validated': False,
        }
        for i in range(6)
    }
    fake = QueryFakeOpenSearch({ap.ITEMS_INDEX: members})

    def _undo_to_other_class() -> None:
        doc = fake.docs(ap.ITEMS_INDEX)['m0']
        doc.update(class_name='sedan', class_id=4)
        doc['class_id_history'] = [dict(UNDO_ENTRY)]

    _hook_mget(fake, _undo_to_other_class)
    await ap.auto_promote_clusters(fake, min_purity=0.5, min_members=2, dry_run=False)

    docs = fake.docs(ap.ITEMS_INDEX)
    assert docs['m0'].get('class_validated') is False
    assert docs['m0']['class_name'] == 'sedan'
    assert docs['m1']['class_validated'] is True


# ---------------------------------------------------------------- detection worker


@pytest.mark.asyncio
async def test_worker_fetch_requests_guard_fields_and_records_token() -> None:
    from scripts.curation.worker.cascade import _fetch_pending

    doc = {
        'crop_id': 'w1',
        'image_path': '/x.jpg',
        'bbox_norm': [0.1, 0.1, 0.5, 0.5],
        'created_at': '2026-01-01',
        F.status: 'pending_detection',
        'class_id': 3,
        'class_source': 'det_model',
    }
    fake = QueryFakeOpenSearch({ITEMS: {'w1': doc}})
    bodies: list[dict[str, Any]] = []
    orig = fake.search

    async def _search(**kw: Any) -> dict[str, Any]:
        bodies.append(kw['body'])
        return await orig(**kw)

    fake.search = _search  # type: ignore[method-assign]
    [task] = await _fetch_pending(fake, batch_size=5)
    assert set(CLASS_GUARD_SOURCE_FIELDS) <= set(bodies[0]['_source'])
    assert task.class_token == class_state_token(doc)


@pytest.mark.asyncio
async def test_worker_drops_class_fields_when_class_changed_after_read() -> None:
    from scripts.curation.worker.bulk_writer import _bulk_update
    from scripts.curation.worker.state import CURATION_ITEMS_INDEX, _ItemTask

    read = _discarded('w1') | {F.status: 'pending_detection'}
    fake = QueryFakeOpenSearch({CURATION_ITEMS_INDEX: {'w1': dict(read)}})
    task = _ItemTask(
        crop_id='w1',
        image_path='/x.jpg',
        item_bbox_norm=(0.1, 0.1, 0.5, 0.5),
        region_status='pending_detection',
        class_name='',
        group='',
        class_token=class_state_token(read),
    )
    task.update_doc = {
        F.status: 'detected',
        F.bbox_norm: [0.2, 0.2, 0.3, 0.3],
        'class_id': 7,
        'class_name': 'sportscar',
        'class_source': 'vlm',
        'label_source': 'vlm',
    }
    _hook_mget(fake, lambda: _restored_by_undo(fake.docs(CURATION_ITEMS_INDEX)['w1']))
    written, _skipped = await _bulk_update(fake, [task])

    doc = fake.docs(CURATION_ITEMS_INDEX)['w1']
    assert written == 1
    assert doc[F.status] == 'detected'
    assert (doc['class_id'], doc['class_source']) == (12, 'det_model')


@pytest.mark.asyncio
async def test_worker_writes_class_when_unchanged() -> None:
    from scripts.curation.worker.bulk_writer import _bulk_update
    from scripts.curation.worker.state import CURATION_ITEMS_INDEX, _ItemTask

    read = _plain('w2') | {F.status: 'pending_detection'}
    fake = QueryFakeOpenSearch({CURATION_ITEMS_INDEX: {'w2': dict(read)}})
    task = _ItemTask(
        crop_id='w2',
        image_path='/x.jpg',
        item_bbox_norm=(0.1, 0.1, 0.5, 0.5),
        region_status='pending_detection',
        class_name='',
        group='',
        class_token=class_state_token(read),
    )
    task.update_doc = {F.status: 'detected', 'class_id': 7, 'class_source': 'vlm'}
    await _bulk_update(fake, [task])
    assert fake.docs(CURATION_ITEMS_INDEX)['w2']['class_id'] == 7
