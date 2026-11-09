"""Labeler Undo round trips on the items index.

- ``DELETE /crops/{id}/label`` must put an item back exactly as it was
  before the most recent human label write (single label, batch label,
  move) — class, provenance, validation and cluster placement — not just
  null the provenance while leaving the human's class in place.
- ``POST /crops/batch_exclude`` -> ``batch_unexclude`` must give a validated item its
  validation back and leave its cluster placement alone.

Every write runs through the real router functions against the
query-evaluating in-memory OpenSearch.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from _class_registry_fake import InMemoryRegistry

from curation.query_fakes import QueryFakeOpenSearch
from curation.queued_label import call_raw, call_route
from curation.serving_partition import serve_clusters
from src.config.curation_accessors import base_curation_config
from src.routers.curation import label_undo
from src.services.curation.item_doc import DetectedItem, build_item_doc


if TYPE_CHECKING:
    from pathlib import Path


pytestmark = pytest.mark.usefixtures('label_ops_state')


ITEMS = base_curation_config().items_index
CLUSTER_UID = '6f1c1d3e-0a7e-4c55-9d5e-2f0f8f2a9b11'

# The class / label / provenance / cluster state an undo must restore.
CLASS_STATE_FIELDS = (
    'class_id',
    'class_name',
    'class_source',
    'label_source',
    'confidence',
    'class_detector',
    'class_detector_version',
    'class_labeler',
    'class_labeled_at',
    'class_validated',
    'cluster_partition_id',
    'cluster_uid',
    'cluster_state',
    'cluster_score',
    'cluster_margin',
    'cluster_assigned_by',
)


pytest_plugins = ['_records_world', 'curation.jobs_a.optin_queue']


def _ingest_doc(crop_id: str, **item_kw: Any) -> dict[str, Any]:
    item = DetectedItem(bbox_pixel=(10.0, 10.0, 50.0, 50.0), score=0.42, **item_kw)
    return build_item_doc(
        crop_id=crop_id,
        image_id=f'img-{crop_id}',
        image_path=f'/data/{crop_id}.jpg',
        source='test',
        request_id='req-1',
        bbox_norm=[0.1, 0.1, 0.5, 0.5],
        item=item,
        now='2026-09-01T00:00:00+00:00',
        crop_area_norm=0.16,
        crop_rank_in_image=1,
        blur_full_var=None,
        blur_lap_var=None,
        blur_lap_ratio=None,
    )


def _proposal_doc(crop_id: str) -> dict[str, Any]:
    """An unlabeled ingest proposal sitting in a cluster of the serving partition."""
    doc = _ingest_doc(
        crop_id,
        class_source='detector_proposal',
        proposal_name='thing',
        class_detector='detector',
        class_detector_version='3',
    )
    return {**doc, 'cluster_uid': CLUSTER_UID, 'cluster_state': 'assigned'}


def _class_state(doc: dict[str, Any]) -> dict[str, Any]:
    state = {f: doc.get(f) for f in CLASS_STATE_FIELDS}
    state['class_validated'] = bool(state['class_validated'])
    return state


_NAMES = ('widget', 'gadget', 'gizmo')


@pytest.fixture
def registry_ids(tmp_path: Path) -> tuple[InMemoryRegistry, dict[str, int]]:
    reg = InMemoryRegistry()
    return reg, {name: reg.add(name) for name in _NAMES}


@pytest.fixture
def registry(registry_ids: tuple[InMemoryRegistry, dict[str, int]]) -> InMemoryRegistry:
    return registry_ids[0]


@pytest.fixture
def ids(registry_ids: tuple[InMemoryRegistry, dict[str, int]]) -> dict[str, int]:
    return registry_ids[1]


@pytest.fixture
def crops(monkeypatch: pytest.MonkeyPatch, registry: InMemoryRegistry):
    from types import SimpleNamespace

    from src.routers.curation import crops_exclude, crops_write

    monkeypatch.setattr(crops_write, 'get_class_registry', lambda: registry)
    return SimpleNamespace(
        label_crop=crops_write.label_crop,
        batch_label_crops=crops_write.batch_label_crops,
        move_crops=crops_write.move_crops,
        batch_exclude_crops=crops_exclude.batch_exclude_crops,
        batch_unexclude_crops=crops_exclude.batch_unexclude_crops,
    )


async def _label(crops, fake, registry, crop_id: str, class_id: int) -> None:
    from src.routers.curation._common_models import CropLabelRequest

    await crops.label_crop(crop_id, CropLabelRequest(class_id=class_id), fake, registry)


# =============================================================================
# Bug A — DELETE /crops/{id}/label restores the pre-label state
# =============================================================================


@pytest.mark.asyncio
async def test_undo_restores_ingest_proposal_field_for_field(crops, registry, ids) -> None:
    fake = QueryFakeOpenSearch({ITEMS: {'c1': _proposal_doc('c1')}})
    before = _class_state(fake.docs(ITEMS)['c1'])
    assert before['class_id'] is None

    await _label(crops, fake, registry, 'c1', ids['gizmo'])
    labeled = fake.docs(ITEMS)['c1']
    assert labeled['class_id'] == ids['gizmo']
    assert labeled['class_validated'] is True

    await label_undo.unlabel_crop('c1', fake)
    assert _class_state(fake.docs(ITEMS)['c1']) == before


@pytest.mark.asyncio
async def test_undo_restores_prior_vlm_label(crops, registry, ids) -> None:
    doc = _ingest_doc('c2', class_source='detector_proposal', class_detector='detector')
    doc.update(
        class_id=ids['widget'],
        class_name='widget',
        class_source='vlm',
        label_source='vlm',
        class_detector='vlm',
        class_detector_version='v2',
        class_labeler='vlm_label_batch',
        class_labeled_at='2026-09-02T00:00:00+00:00',
        class_validated=False,
    )
    fake = QueryFakeOpenSearch({ITEMS: {'c2': doc}})
    before = _class_state(fake.docs(ITEMS)['c2'])

    await _label(crops, fake, registry, 'c2', ids['gadget'])
    await label_undo.unlabel_crop('c2', fake)

    after = fake.docs(ITEMS)['c2']
    assert _class_state(after) == before
    assert after['class_source'] == 'vlm'
    assert after['class_id'] == ids['widget']


@pytest.mark.asyncio
@pytest.mark.db
@pytest.mark.usefixtures('world', 'fake_queue')
async def test_undo_reverses_batch_label_and_move(crops, registry, ids) -> None:
    from src.routers.curation._common_models import CropBatchLabelRequest, CropMoveRequest

    fake = QueryFakeOpenSearch({ITEMS: {'b': _proposal_doc('b'), 'm': _proposal_doc('m')}})
    before = {k: _class_state(v) for k, v in fake.docs(ITEMS).items()}
    _, (uid, _other) = await serve_clusters()

    await call_route(
        crops.batch_label_crops, CropBatchLabelRequest(crop_ids=['b'], class_id=ids['gadget']), fake
    )
    await call_route(crops.move_crops, CropMoveRequest(crop_ids=['m'], cluster_uid=uid), fake)
    assert fake.docs(ITEMS)['m']['cluster_uid'] == str(uid)

    await label_undo.unlabel_crop('b', fake)
    await label_undo.unlabel_crop('m', fake)
    assert {k: _class_state(v) for k, v in fake.docs(ITEMS).items()} == before


@pytest.mark.asyncio
async def test_successive_undos_step_back_through_human_labels(crops, registry, ids) -> None:
    fake = QueryFakeOpenSearch({ITEMS: {'c3': _proposal_doc('c3')}})
    original = _class_state(fake.docs(ITEMS)['c3'])

    await _label(crops, fake, registry, 'c3', ids['widget'])
    first_label = _class_state(fake.docs(ITEMS)['c3'])
    await _label(crops, fake, registry, 'c3', ids['gadget'])

    await label_undo.unlabel_crop('c3', fake)
    restored = _class_state(fake.docs(ITEMS)['c3'])
    # Back to the first human label: still validated, in its class (cluster placement is part of
    # CLASS_STATE_FIELDS, so the equality covers it).
    assert restored == first_label
    assert restored['class_id'] == ids['widget']
    assert restored['class_validated'] is True

    await label_undo.unlabel_crop('c3', fake)
    assert _class_state(fake.docs(ITEMS)['c3']) == original

    # Nothing left to undo: unlabeled, no invented provenance.
    await label_undo.unlabel_crop('c3', fake)
    final = fake.docs(ITEMS)['c3']
    assert final['class_id'] is None
    assert final['class_name'] is None
    assert final['class_validated'] is False
    for f in ('class_source', 'class_detector', 'class_labeler', 'class_labeled_at'):
        assert final[f] is None


@pytest.mark.asyncio
async def test_undo_without_history_clears_class(crops) -> None:
    """A human label with no recorded history (written before snapshots
    existed) can't be restored to anything; it must not stay validated
    under the human's class."""
    doc = _proposal_doc('legacy')
    doc.update(
        class_id=4,
        class_name='gizmo',
        class_source='human',
        label_source='human',
        class_validated=True,
        cluster_uid='00000000-0000-0000-0000-000000000004',
    )
    fake = QueryFakeOpenSearch({ITEMS: {'legacy': doc}})
    await label_undo.unlabel_crop('legacy', fake)
    after = fake.docs(ITEMS)['legacy']
    assert after['class_id'] is None
    assert after['class_validated'] is False
    assert after['class_source'] is None
    assert after['cluster_uid'] is None


# =============================================================================
# Bug B — exclude -> unexclude restores the validation and never moves the item
# =============================================================================


async def _exclude(crops, fake, crop_ids: list[str]) -> dict[str, Any]:
    from src.routers.curation._common_models import CropExcludeRequest

    return await crops.batch_exclude_crops(CropExcludeRequest(crop_ids=crop_ids), fake)


async def _unexclude(crops, fake, crop_ids: list[str]) -> dict[str, Any]:
    from src.routers.curation._common_models import CropUnexcludeRequest

    return await crops.batch_unexclude_crops(CropUnexcludeRequest(crop_ids=crop_ids), fake)


@pytest.mark.asyncio
async def test_exclusion_restores_validation_and_never_moves_the_item(crops, registry, ids) -> None:
    fake = QueryFakeOpenSearch({ITEMS: {'v': _proposal_doc('v'), 'u': _proposal_doc('u')}})
    await _label(crops, fake, registry, 'v', ids['gadget'])
    placed = {k: v for k, v in fake.docs(ITEMS)['v'].items() if k.startswith('cluster_')}

    assert await _exclude(crops, fake, ['v', 'u']) == {
        'excluded': 2,
        'updated_ids': ['v', 'u'],
        'errors': 0,
    }
    for doc in fake.docs(ITEMS).values():
        assert doc['class_excluded'] is True
        assert doc['class_validated'] is False
        assert doc['cluster_uid'] == CLUSTER_UID, 'the cluster readers filter class_excluded'

    assert await _unexclude(crops, fake, ['v', 'u']) == {
        'unexcluded': 2,
        'updated_ids': ['v', 'u'],
        'errors': 0,
    }
    v, u = fake.docs(ITEMS)['v'], fake.docs(ITEMS)['u']
    assert v['class_excluded'] is False
    assert v['class_validated'] is True
    assert {k: x for k, x in v.items() if k.startswith('cluster_')} == placed
    assert u['class_validated'] is False
    assert u['cluster_uid'] == CLUSTER_UID


# =============================================================================
# A move is placement in the serving partition, not a class label
# =============================================================================


@pytest.mark.asyncio
@pytest.mark.db
@pytest.mark.usefixtures('world', 'fake_queue')
async def test_move_sets_the_partition_placement_not_the_class(crops, registry, ids) -> None:
    from src.routers.curation._common_models import CropMoveRequest

    partition, (uid, _other) = await serve_clusters()
    fake = QueryFakeOpenSearch({ITEMS: {'m': _proposal_doc('m'), 'peer': _proposal_doc('peer')}})
    before = _class_state(fake.docs(ITEMS)['m'])

    out = await call_route(crops.move_crops, CropMoveRequest(crop_ids=['m'], cluster_uid=uid), fake)
    assert out['updated'] == 1
    m = fake.docs(ITEMS)['m']
    assert (m['cluster_partition_id'], m['cluster_uid']) == (str(partition), str(uid))
    assert (m['cluster_state'], m['cluster_assigned_by']) == ('assigned', 'edit')
    assert m['cluster_score'] is None
    assert m.get('class_id') is None
    assert m['class_validated'] is False
    assert m['class_source'] == 'detector_proposal'
    assert 'cluster_uid' not in fake.docs(ITEMS)['peer']

    await label_undo.unlabel_crop('m', fake)
    assert _class_state(fake.docs(ITEMS)['m']) == before


@pytest.mark.asyncio
@pytest.mark.db
@pytest.mark.usefixtures('world', 'fake_queue')
async def test_move_clears_a_human_class(crops, registry, ids) -> None:
    """A human-validated crop moved into a cluster is no longer that class; keeping it
    validated would leak the old class into export."""
    from src.routers.curation._common_models import CropMoveRequest

    _, (uid, _other) = await serve_clusters()
    fake = QueryFakeOpenSearch({ITEMS: {'h': _proposal_doc('h'), 'peer': _proposal_doc('peer')}})
    await _label(crops, fake, registry, 'h', ids['widget'])
    await call_route(crops.move_crops, CropMoveRequest(crop_ids=['h'], cluster_uid=uid), fake)
    h = fake.docs(ITEMS)['h']
    assert h['cluster_uid'] == str(uid)
    assert h['class_id'] is None
    assert h['class_validated'] is False


@pytest.mark.asyncio
@pytest.mark.db
@pytest.mark.usefixtures('world', 'fake_queue')
@pytest.mark.parametrize('serving', [True, False])
async def test_move_rejects_a_cluster_the_serving_partition_does_not_hold(
    crops, registry, serving
) -> None:
    """A uid that is not a cluster of the serving partition (unknown, or nothing is serving)
    is not a move target; writing it would invent a phantom cluster."""
    import uuid

    from fastapi import HTTPException

    from src.routers.curation._common_models import CropMoveRequest

    if serving:
        await serve_clusters()
    fake = QueryFakeOpenSearch({ITEMS: {'m': _proposal_doc('m')}})
    before = dict(fake.docs(ITEMS)['m'])
    target = uuid.uuid4()
    with pytest.raises(HTTPException) as exc:
        await call_raw(crops.move_crops, CropMoveRequest(crop_ids=['m'], cluster_uid=target), fake)
    assert exc.value.status_code == 400
    assert str(target) in str(exc.value.detail)
    assert fake.docs(ITEMS)['m'] == before


@pytest.mark.asyncio
async def test_re_exclude_keeps_the_recorded_pre_exclusion_state(crops, registry, ids) -> None:
    fake = QueryFakeOpenSearch({ITEMS: {'v': _proposal_doc('v')}})
    await _label(crops, fake, registry, 'v', ids['widget'])
    await _exclude(crops, fake, ['v'])
    await _exclude(crops, fake, ['v'])
    await _unexclude(crops, fake, ['v'])
    v = fake.docs(ITEMS)['v']
    assert v['class_validated'] is True


@pytest.mark.asyncio
async def test_unexclude_leaves_non_excluded_items_untouched(crops, registry, ids) -> None:
    fake = QueryFakeOpenSearch({ITEMS: {'v': _proposal_doc('v')}})
    await _label(crops, fake, registry, 'v', ids['widget'])
    before = dict(fake.docs(ITEMS)['v'])
    # Nothing was restored, so nothing is reported (and no id offered for undo).
    assert await _unexclude(crops, fake, ['v']) == {
        'unexcluded': 0,
        'updated_ids': [],
        'errors': 0,
    }
    assert fake.docs(ITEMS)['v'] == before


@pytest.mark.asyncio
async def test_unexclude_legacy_exclusion_of_human_label(crops) -> None:
    """Excluded before the pre-exclusion state was recorded: the exclude
    already wiped class_validated, but a human-sourced class can only have
    been a validated human label."""
    doc = _proposal_doc('old')
    doc.update(
        class_id=7,
        class_name='gizmo',
        class_source='human',
        class_validated=False,
        class_excluded=True,
        excluded_reason='ignore',
    )
    fake = QueryFakeOpenSearch({ITEMS: {'old': doc}})
    await _unexclude(crops, fake, ['old'])
    after = fake.docs(ITEMS)['old']
    assert after['class_validated'] is True


@pytest.mark.asyncio
async def test_undo_label_on_excluded_item_keeps_it_excluded(crops, registry, ids) -> None:
    fake = QueryFakeOpenSearch({ITEMS: {'x': _proposal_doc('x')}})
    await _label(crops, fake, registry, 'x', ids['widget'])
    await _label(crops, fake, registry, 'x', ids['gadget'])
    await _exclude(crops, fake, ['x'])

    await label_undo.unlabel_crop('x', fake)
    x = fake.docs(ITEMS)['x']
    assert x['class_excluded'] is True
    assert x['class_validated'] is False
    assert x['class_id'] == ids['widget']

    await _unexclude(crops, fake, ['x'])
    x = fake.docs(ITEMS)['x']
    assert x['class_validated'] is True


# =============================================================================
# Undo routes: single-crop undo and undo_batch
# =============================================================================


@pytest.fixture
def client_for():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from src.routers.curation import _raw_opensearch_dep, router as curation_router

    def _make(fake: QueryFakeOpenSearch) -> TestClient:
        app = FastAPI()
        from _curation_app import mount_curation_routers

        mount_curation_routers(app, curation_router)
        app.dependency_overrides[_raw_opensearch_dep] = lambda: fake
        return TestClient(app)

    return _make


@pytest.mark.asyncio
async def test_undo_route_restores_and_returns_the_wire_item(
    crops, registry, ids, client_for
) -> None:
    fake = QueryFakeOpenSearch({ITEMS: {'c1': _proposal_doc('c1')}})
    before = _class_state(fake.docs(ITEMS)['c1'])
    await _label(crops, fake, registry, 'c1', ids['widget'])
    first = _class_state(fake.docs(ITEMS)['c1'])
    await _label(crops, fake, registry, 'c1', ids['gadget'])
    client = client_for(fake)

    r = client.post('/curation/projects/default/crops/c1/label/undo')
    assert r.status_code == 200, r.text
    body = r.json()
    assert body['crop_id'] == 'c1'
    assert body['class_id'] == ids['widget']
    # The earlier validated human label comes back as it was.
    assert _class_state(fake.docs(ITEMS)['c1']) == first

    r = client.post('/curation/projects/default/crops/c1/label/undo')
    assert r.status_code == 200, r.text
    assert r.json()['class_id'] is None
    assert _class_state(fake.docs(ITEMS)['c1']) == before

    r = client.post('/curation/projects/default/crops/c1/label/undo')
    assert r.status_code == 409
    assert _class_state(fake.docs(ITEMS)['c1']) == before


def test_undo_route_unknown_crop_is_404(client_for) -> None:
    client = client_for(QueryFakeOpenSearch({ITEMS: {}}))
    assert client.post('/curation/projects/default/crops/nope/label/undo').status_code == 404


@pytest.mark.asyncio
@pytest.mark.db
@pytest.mark.usefixtures('world', 'fake_queue')
async def test_undo_batch_reverses_a_batch_label(crops, ids, client_for) -> None:
    from src.routers.curation._common_models import CropBatchLabelRequest

    fake = QueryFakeOpenSearch({ITEMS: {k: _proposal_doc(k) for k in ('b1', 'b2', 'untouched')}})
    before = {k: _class_state(v) for k, v in fake.docs(ITEMS).items()}
    await call_route(
        crops.batch_label_crops,
        CropBatchLabelRequest(crop_ids=['b1', 'b2'], class_id=ids['gizmo']),
        fake,
    )
    client = client_for(fake)

    r = client.post(
        '/curation/projects/default/crops/label/undo_batch',
        json={'crop_ids': ['b1', 'b2', 'untouched', 'gone']},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body['undone'] == 2
    assert sorted(i['crop_id'] for i in body['items']) == ['b1', 'b2']
    assert body['nothing_to_undo'] == ['untouched']
    assert body['not_found'] == ['gone']
    assert body['conflicts'] == []
    assert {k: _class_state(v) for k, v in fake.docs(ITEMS).items()} == before

    r = client.post(
        '/curation/projects/default/crops/label/undo_batch', json={'crop_ids': ['b1', 'untouched']}
    )
    assert r.status_code == 409
