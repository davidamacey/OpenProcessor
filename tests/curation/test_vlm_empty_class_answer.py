"""An empty VLM class answer is not ``vlm_unmatched``.

Live evidence (2026-09-24): 2,927 of 3,102 ``vlm_unmatched`` items carried
``vlm_raw_class=''`` -- the VLM reply had no class at all, yet every writer
recorded it as "the VLM answered a label not in the registry", replacing a
proposal's proposal source / a classifier label's source with
``vlm_unmatched`` and dropping the item out of every later VLM pass.

Now an empty answer (no class, ``null``, ``-1``, out-of-range index, or no
parseable entry) leaves the class fields untouched and records the attempt
(``vlm_class_attempted_at`` + ``vlm_class_empty_reason``). ``vlm_unmatched``
is kept for a real, non-empty label with ``vlm_raw_class`` preserved. Any
class write that does happen snapshots the pre-write class state
(``record_class_snapshot``, restorable).
"""

from __future__ import annotations

import io
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock

import pytest
from PIL import Image

from curation.query_fakes import QueryFakeOpenSearch
from scripts.curation.worker.verify import _combined_class_update
from src.clients.curation_opensearch import ClassRegistry
from src.config import get_curation_config
from src.services.curation.history import CLASS_STATE_FIELDS
from src.services.curation.vlm_class_attempt import (
    VLM_CLASS_ATTEMPTED_AT_FIELD as ATTEMPTED,
    VLM_CLASS_EMPTY_REASON_FIELD as REASON,
)
from src.services.labeling.vlm_labeler import VlmClassPrediction, VlmCombinedReply


if TYPE_CHECKING:
    from pathlib import Path


ITEMS = get_curation_config().items_index
CLASS_KEYS = {
    'class_id',
    'class_name',
    'class_source',
    'label_source',
    'class_validated',
    'cluster_id',
    'vlm_confidence',
    'vlm_raw_class',
    'vlm_raw_label',
}


# ---------------------------------------------------------------- combined reply


class TestCombinedClassUpdate:
    NAMES = ['widget', 'gadget']

    @pytest.mark.parametrize(
        ('class_id', 'reason'),
        [(None, 'no_answer'), (-1, 'no_match'), (2, 'invalid_index'), (99, 'invalid_index')],
    )
    def test_empty_answer_leaves_class_fields_and_records_attempt(
        self, class_id: int | None, reason: str
    ) -> None:
        reply = VlmCombinedReply(img_id='c', class_id=class_id, class_confidence='low')
        update = _combined_class_update(reply, self.NAMES, now='2026-09-24T12:00:00+00:00')
        assert not CLASS_KEYS & update.keys()
        assert update[ATTEMPTED] == '2026-09-24T12:00:00+00:00'
        assert update[REASON] == reason
        # The region-side markers are unaffected.
        assert update['vlm_verify_completed_at'] == '2026-09-24T12:00:00+00:00'

    def test_named_label_outside_registry_is_unmatched_with_raw_label(self) -> None:
        reply = VlmCombinedReply(
            img_id='c', class_id=None, class_raw='zeppelin', class_confidence='medium'
        )
        update = _combined_class_update(reply, self.NAMES, now='t')
        assert update['class_source'] == 'vlm_unmatched'
        assert update['vlm_raw_class'] == 'zeppelin'
        assert update['vlm_raw_label'] == 'zeppelin'
        assert update['vlm_confidence'] == 'medium'
        assert update[REASON] is None
        assert update[ATTEMPTED] == 't'

    def test_resolved_class_clears_empty_reason(self) -> None:
        reply = VlmCombinedReply(img_id='c', class_id=1, class_confidence='high')
        update = _combined_class_update(
            reply, self.NAMES, now='t', name_to_id={'widget': 10, 'gadget': 11}
        )
        assert (update['class_id'], update['class_source']) == (11, 'vlm')
        assert update[REASON] is None
        assert update[ATTEMPTED] == 't'

    def test_classification_not_requested_records_no_attempt(self) -> None:
        reply = VlmCombinedReply(img_id='c', class_id=None)
        update = _combined_class_update(reply, None, now='t')
        assert ATTEMPTED not in update
        assert REASON not in update


@pytest.mark.asyncio
async def test_worker_combined_class_write_snapshots_restorable_history() -> None:
    """The combined write onto a proposal (no class_id yet) still snapshots
    the full pre-write class state."""
    from curation.occ_fakes import make_bulk_response, make_bulk_update_item, make_mget_response
    from scripts.curation.worker.bulk_writer import _bulk_update
    from scripts.curation.worker.state import _ItemTask
    from src.config import get_region_fields
    from src.services.curation.class_write_guard import class_state_token

    F = get_region_fields()
    t = _ItemTask(
        crop_id='crop-1',
        image_path='/dev/null/never-read',
        item_bbox_norm=(0.1, 0.1, 0.5, 0.5),
        region_status='pending',
        class_name='',
        group='',
    )
    t.update_doc = {
        'class_id': 9,
        'class_name': 'camaro',
        'class_source': 'vlm',
        'label_source': 'vlm',
        'class_validated': False,
        F.status: 'detected',
    }
    source = {
        'class_source': 'det_proposal',
        'label_source': 'det_proposal',
        'class_detector': 'det',
        'class_labeler': 'ingest',
        'class_validated': False,
        F.status: 'pending',
    }
    t.class_token = class_state_token(source)
    opensearch = AsyncMock()
    opensearch.mget = AsyncMock(return_value=make_mget_response({'crop-1': source}))
    opensearch.bulk = AsyncMock(
        return_value=make_bulk_response([make_bulk_update_item('crop-1', status=200)])
    )
    await _bulk_update(opensearch, [t])
    assert opensearch.bulk.await_args is not None
    written = opensearch.bulk.await_args.kwargs['body'][1]['doc']
    entry = written['class_id_history'][-1]
    assert set(CLASS_STATE_FIELDS) <= entry.keys()
    assert entry['restorable'] is True
    assert (entry['class_source'], entry['class_detector']) == ('det_proposal', 'det')
    assert entry['writer'] == 'region_worker'


# ---------------------------------------------------------------- label_batch route


def _item(crop_id: str, **extra: Any) -> dict[str, Any]:
    return {
        'crop_id': crop_id,
        'image_path': f'/data/{crop_id}.jpg',
        'bbox_norm': [0.1, 0.1, 0.5, 0.5],
        'class_id': 3,
        'class_name': 'widget',
        'class_source': 'item_model',
        'label_source': 'item_model',
        'class_detector': 'item',
        'class_labeler': 'ingest',
        'class_validated': False,
        'confidence': 0.4,
        **extra,
    }


def _proposal(crop_id: str) -> dict[str, Any]:
    return _item(
        crop_id,
        class_id=None,
        class_name=None,
        class_source='det_proposal',
        label_source='det_proposal',
        class_detector='det',
    )


async def _run_label_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, preds: dict[str, VlmClassPrediction]
) -> dict[str, dict[str, Any]]:
    import src.routers.curation.vlm as vlm_mod
    from src.routers.curation.vlm import VlmLabelBatchRequest

    buf = io.BytesIO()
    Image.new('RGB', (32, 32), (1, 2, 3)).save(buf, format='JPEG')
    for cid in preds:
        (tmp_path / f'{cid}.jpg').write_bytes(buf.getvalue())
    monkeypatch.setattr(
        vlm_mod, 'get_curation_config', lambda: SimpleNamespace(crop_cache_dir=tmp_path)
    )
    reg = ClassRegistry(path=tmp_path / 'class_registry.json')
    reg.add_class('widget')
    reg.add_class('gadget')
    monkeypatch.setattr(vlm_mod, 'get_class_registry', lambda: reg)
    monkeypatch.setattr(vlm_mod, '_default_pack_name', AsyncMock(return_value=None))
    docs = {cid: (_proposal(cid) if cid.startswith('p') else _item(cid)) for cid in preds}
    fake = QueryFakeOpenSearch({ITEMS: docs})

    class _Labeler:
        async def label_or_propose_batch(self, crops: list[Any], _names: list[str]) -> list[Any]:
            return [preds[c.img_id] for c in crops]

    monkeypatch.setattr(vlm_mod, '_get_vlm_labeler', lambda *_a, **_k: _Labeler())
    await vlm_mod.vlm_label_batch(VlmLabelBatchRequest(crop_ids=list(preds)), fake)
    return fake.docs(ITEMS)


def _pred(cid: str, name: str, **kw: Any) -> VlmClassPrediction:
    return VlmClassPrediction(img_id=cid, class_name=name, confidence=kw.pop('conf', 'low'), **kw)


@pytest.mark.asyncio
async def test_label_batch_empty_answers_keep_class_and_record_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    preds = {
        'p_empty': _pred('p_empty', ''),
        'c_empty': _pred('c_empty', ''),
        'c_garbled': _pred('c_garbled', '', failure='unparseable'),
        'c_new_no_name': _pred('c_new_no_name', '__new__'),
        'c_down': _pred('c_down', '', failure='request_failed'),
    }
    docs = await _run_label_batch(tmp_path, monkeypatch, preds)

    assert (docs['p_empty']['class_source'], docs['p_empty']['label_source']) == (
        'det_proposal',
        'det_proposal',
    )
    assert docs['p_empty'].get('class_id') is None
    for cid in ('c_empty', 'c_garbled', 'c_new_no_name'):
        doc = docs[cid]
        assert (doc['class_id'], doc['class_source'], doc['label_source']) == (
            3,
            'item_model',
            'item_model',
        )
        assert doc.get('vlm_raw_class') is None
        assert doc.get('vlm_confidence') is None
        assert doc.get(ATTEMPTED)
        assert not doc.get('class_id_history')
    assert docs['p_empty'][REASON] == 'no_answer'
    assert docs['c_empty'][REASON] == 'no_answer'
    assert docs['c_garbled'][REASON] == 'unparseable'
    assert docs['c_new_no_name'][REASON] == 'no_match'
    # A call that never completed is not an answer: nothing is written.
    assert docs['c_down'].get(ATTEMPTED) is None
    assert docs['c_down']['class_source'] == 'item_model'


@pytest.mark.asyncio
async def test_label_batch_real_answers_snapshot_history_and_clear_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    preds = {
        'c_label': _pred('c_label', 'gadget', conf='high'),
        'p_unmatched': _pred('p_unmatched', 'zeppelin', conf='medium'),
    }
    docs = await _run_label_batch(tmp_path, monkeypatch, preds)

    labeled = docs['c_label']
    assert (labeled['class_name'], labeled['class_source']) == ('gadget', 'vlm')
    assert labeled[REASON] is None
    assert labeled[ATTEMPTED]
    entry = labeled['class_id_history'][-1]
    assert entry['restorable'] is True
    assert entry['writer'] == 'vlm_label_batch'
    assert (entry['class_id'], entry['class_source'], entry['class_detector']) == (
        3,
        'item_model',
        'item',
    )

    unmatched = docs['p_unmatched']
    assert unmatched['class_source'] == 'vlm_unmatched'
    assert unmatched['vlm_raw_class'] == 'zeppelin'
    assert unmatched[REASON] is None
    # Even a class-source-only write (no class_id) is snapshotted.
    snap = unmatched['class_id_history'][-1]
    assert (snap['class_source'], snap['restorable']) == ('det_proposal', True)


# ---------------------------------------------------------------- auto-label VLM stage


@pytest.mark.asyncio
async def test_pipeline_vlm_stage_empty_answer_keeps_class(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from curation.test_pipeline import _FakeClassEntry, _FakeRegistry
    from src.routers.curation import pipeline, pipeline_health
    from src.services.curation import image_serving
    from src.services.curation.autolabel import selection

    fake = QueryFakeOpenSearch({ITEMS: {'e': _item('e'), 'g': _item('g'), 'x': _item('x')}})
    answers = {
        'e': VlmClassPrediction(
            img_id='e', class_name='', confidence='low', raw_response=']', failure='unparseable'
        ),
        'g': VlmClassPrediction(img_id='g', class_name='sportscar', confidence='high'),
        'x': VlmClassPrediction(img_id='x', class_name='', confidence='low', raw_response='{}'),
    }

    class _Labeler:
        model = 'fake-vlm'
        _pack = None

        async def label_or_propose_batch(self, crops: list[Any], *_a: Any, **_k: Any) -> list[Any]:
            return [answers[c.img_id] for c in crops]

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
    for cid, reason in (('e', 'unparseable'), ('x', 'no_answer')):
        assert (docs[cid]['class_id'], docs[cid]['class_source']) == (3, 'item_model')
        assert docs[cid][REASON] == reason
        assert docs[cid].get('vlm_raw_class') is None
    assert (docs['g']['class_id'], docs['g']['class_source']) == (7, 'vlm')
    assert docs['g'][REASON] is None
    assert docs['g']['class_id_history'][-1]['restorable'] is True


# ---------------------------------------------------------------- selectors


def _recent_empty(now: datetime, hours_ago: float) -> dict[str, Any]:
    return {
        ATTEMPTED: (now - timedelta(hours=hours_ago)).isoformat(),
        REASON: 'no_answer',
    }


@pytest.mark.asyncio
async def test_selectors_skip_items_with_a_recent_empty_answer() -> None:
    from scripts.curation.vlm_worker import _build_pending_query
    from src.services.curation.autolabel.selection import vlm_selection_query

    now = datetime.now(UTC)
    docs = {
        'fresh_empty': {**_proposal('fresh_empty'), **_recent_empty(now, 1)},
        'stale_empty': {**_proposal('stale_empty'), **_recent_empty(now, 48)},
        'answered': {**_proposal('answered'), ATTEMPTED: now.isoformat(), REASON: None},
        'never': _proposal('never'),
    }
    for d in docs.values():
        d['pe_embedding'] = [0.0]
    fake = QueryFakeOpenSearch({ITEMS: docs})

    worker = await fake.search(index=ITEMS, body={'size': 50, 'query': _build_pending_query(0.8)})
    sweep = await fake.search(
        index=ITEMS,
        body={
            'size': 50,
            'query': vlm_selection_query(
                class_id=None, cluster_id=None, classifier_confidence_skip_vlm=0.8, now=now
            ),
        },
    )
    expected = {'stale_empty', 'answered', 'never'}
    assert {h['_id'] for h in worker['hits']['hits']} == expected
    assert {h['_id'] for h in sweep['hits']['hits']} == expected


def test_all_review_tab_surfaces_empty_answers() -> None:
    from src.services.curation.review_queries import build_tab_query

    must, _must_not, _reason = build_tab_query('all', include_test=False, text=None, max_rank=None)
    should = must[0]['bool']['should']
    assert {'exists': {'field': REASON}} in should


# ---------------------------------------------------------------- storage + wire


def test_fields_are_mapped_and_guarded() -> None:
    from src.clients.curation_opensearch import _items_body
    from src.clients.occ import CLASS_WRITE_FIELDS

    props = _items_body()['mappings']['properties']
    assert props[ATTEMPTED] == {'type': 'date'}
    assert props[REASON] == {'type': 'keyword'}
    assert {ATTEMPTED, REASON} <= CLASS_WRITE_FIELDS


@pytest.mark.asyncio
async def test_mapping_migration_adds_the_fields() -> None:
    from src.clients.curation_opensearch import ensure_items_vlm_raw_label_fields

    client = SimpleNamespace(indices=SimpleNamespace(put_mapping=AsyncMock(return_value={})))
    await ensure_items_vlm_raw_label_fields(client)
    body = client.indices.put_mapping.await_args.kwargs['body']
    assert {ATTEMPTED, REASON} <= body['properties'].keys()


def test_wire_item_carries_the_attempt() -> None:
    from src.services.curation.wire import serialize_item

    item = serialize_item(
        {'crop_id': 'c', ATTEMPTED: '2026-09-24T12:00:00+00:00', REASON: 'no_match'}
    )
    assert item[ATTEMPTED] == '2026-09-24T12:00:00+00:00'
    assert item[REASON] == 'no_match'
    empty = serialize_item({'crop_id': 'd'})
    assert empty[ATTEMPTED] is None
    assert empty[REASON] is None
