"""Labeler Undo round trips on the items index.

- ``DELETE /crops/{id}/label`` must put an item back exactly as it was
  before the most recent human label write (single label, batch label,
  move) — class, provenance, validation and cluster placement — not just
  null the provenance while leaving the human's class in place.
- ``POST /crops/batch_exclude`` -> ``batch_unexclude`` must return a
  validated item to its class cluster (``cluster_id == class_id``)
  instead of dropping it into the residual pool.

Every write runs through the real router functions against the
query-evaluating in-memory OpenSearch.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from curation.query_fakes import QueryFakeOpenSearch
from src.clients.curation_opensearch import ClassRegistry
from src.config import get_curation_config
from src.services.curation.item_doc import DetectedItem, build_item_doc


if TYPE_CHECKING:
    from pathlib import Path


ITEMS = get_curation_config().items_index

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
    'cluster_id',
    'cluster_subid',
)


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
    """An unlabeled ingest proposal sitting in a residual candidate cluster."""
    doc = _ingest_doc(
        crop_id,
        class_source='detector_proposal',
        proposal_name='thing',
        class_detector='detector',
        class_detector_version='3',
        cluster_id=10003,
        cluster_distance=0.2,
    )
    doc['cluster_subid'] = '10003b'
    return doc


def _class_state(doc: dict[str, Any]) -> dict[str, Any]:
    state = {f: doc.get(f) for f in CLASS_STATE_FIELDS}
    state['class_validated'] = bool(state['class_validated'])
    return state


_NAMES = ('widget', 'gadget', 'gizmo')


@pytest.fixture
def registry_ids(tmp_path: Path) -> tuple[ClassRegistry, dict[str, int]]:
    reg = ClassRegistry(path=tmp_path / 'class_registry.json')
    return reg, {name: reg.add_class(name) for name in _NAMES}


@pytest.fixture
def registry(registry_ids: tuple[ClassRegistry, dict[str, int]]) -> ClassRegistry:
    return registry_ids[0]


@pytest.fixture
def ids(registry_ids: tuple[ClassRegistry, dict[str, int]]) -> dict[str, int]:
    return registry_ids[1]


@pytest.fixture
def crops(monkeypatch: pytest.MonkeyPatch, registry: ClassRegistry):
    from src.routers.curation import crops as crops_mod

    monkeypatch.setattr(crops_mod, 'get_class_registry', lambda: registry)
    return crops_mod


async def _label(crops, fake, registry, crop_id: str, class_id: int) -> None:
    from src.routers.curation._common import CropLabelRequest

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

    await crops.unlabel_crop('c1', fake)
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
        cluster_id=ids['widget'],
    )
    fake = QueryFakeOpenSearch({ITEMS: {'c2': doc}})
    before = _class_state(fake.docs(ITEMS)['c2'])

    await _label(crops, fake, registry, 'c2', ids['gadget'])
    await crops.unlabel_crop('c2', fake)

    after = fake.docs(ITEMS)['c2']
    assert _class_state(after) == before
    assert after['class_source'] == 'vlm'
    assert after['class_id'] == ids['widget']


@pytest.mark.asyncio
async def test_undo_reverses_batch_label_and_move(crops, registry, ids) -> None:
    from src.routers.curation._common import CropBatchLabelRequest, CropMoveRequest

    fake = QueryFakeOpenSearch({ITEMS: {'b': _proposal_doc('b'), 'm': _proposal_doc('m')}})
    before = {k: _class_state(v) for k, v in fake.docs(ITEMS).items()}

    await crops.batch_label_crops(
        CropBatchLabelRequest(crop_ids=['b'], class_id=ids['gadget']), fake
    )
    await crops.move_crops(CropMoveRequest(crop_ids=['m'], cluster_id=ids['widget']), fake)
    assert fake.docs(ITEMS)['m']['class_source'] == 'human_move'

    await crops.unlabel_crop('b', fake)
    await crops.unlabel_crop('m', fake)
    assert {k: _class_state(v) for k, v in fake.docs(ITEMS).items()} == before


@pytest.mark.asyncio
async def test_successive_undos_step_back_through_human_labels(crops, registry, ids) -> None:
    fake = QueryFakeOpenSearch({ITEMS: {'c3': _proposal_doc('c3')}})
    original = _class_state(fake.docs(ITEMS)['c3'])

    await _label(crops, fake, registry, 'c3', ids['widget'])
    first_label = _class_state(fake.docs(ITEMS)['c3'])
    await _label(crops, fake, registry, 'c3', ids['gadget'])

    await crops.unlabel_crop('c3', fake)
    restored = _class_state(fake.docs(ITEMS)['c3'])
    # Back to the first human label: still validated, in its class cluster.
    assert restored == first_label
    assert restored['cluster_id'] == ids['widget']

    await crops.unlabel_crop('c3', fake)
    assert _class_state(fake.docs(ITEMS)['c3']) == original

    # Nothing left to undo: unlabeled, no invented provenance.
    await crops.unlabel_crop('c3', fake)
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
        cluster_id=4,
    )
    fake = QueryFakeOpenSearch({ITEMS: {'legacy': doc}})
    await crops.unlabel_crop('legacy', fake)
    after = fake.docs(ITEMS)['legacy']
    assert after['class_id'] is None
    assert after['class_validated'] is False
    assert after['class_source'] is None
    assert after['cluster_id'] is None


# =============================================================================
# Bug B — exclude -> unexclude keeps validated items in their class cluster
# =============================================================================


async def _exclude(crops, fake, crop_ids: list[str]) -> dict[str, Any]:
    from src.routers.curation._common import CropExcludeRequest

    return await crops.batch_exclude_crops(CropExcludeRequest(crop_ids=crop_ids), fake)


async def _unexclude(crops, fake, crop_ids: list[str]) -> dict[str, Any]:
    from src.routers.curation._common import CropUnexcludeRequest

    return await crops.batch_unexclude_crops(CropUnexcludeRequest(crop_ids=crop_ids), fake)


@pytest.mark.asyncio
async def test_unexclude_returns_validated_item_to_its_class_cluster(crops, registry, ids) -> None:
    fake = QueryFakeOpenSearch({ITEMS: {'v': _proposal_doc('v'), 'u': _proposal_doc('u')}})
    await _label(crops, fake, registry, 'v', ids['gadget'])
    fake.docs(ITEMS)['v']['cluster_subid'] = f'{ids["gadget"]}a'

    assert await _exclude(crops, fake, ['v', 'u']) == {'excluded': 2, 'errors': 0}
    for doc in fake.docs(ITEMS).values():
        assert doc['class_excluded'] is True
        assert doc['cluster_id'] == -2
        assert doc['class_validated'] is False

    assert await _unexclude(crops, fake, ['v', 'u']) == {'unexcluded': 2, 'errors': 0}
    v, u = fake.docs(ITEMS)['v'], fake.docs(ITEMS)['u']
    assert v['class_excluded'] is False
    assert v['class_validated'] is True
    assert v['cluster_id'] == v['class_id'] == ids['gadget']
    assert v['cluster_subid'] == f'{ids["gadget"]}a'
    # Unvalidated items drop to the residual pool for a fresh assignment.
    assert u['class_validated'] is False
    assert u['cluster_id'] is None
    assert u['cluster_subid'] is None


@pytest.mark.asyncio
async def test_re_exclude_keeps_the_recorded_pre_exclusion_state(crops, registry, ids) -> None:
    fake = QueryFakeOpenSearch({ITEMS: {'v': _proposal_doc('v')}})
    await _label(crops, fake, registry, 'v', ids['widget'])
    await _exclude(crops, fake, ['v'])
    await _exclude(crops, fake, ['v'])
    await _unexclude(crops, fake, ['v'])
    v = fake.docs(ITEMS)['v']
    assert v['class_validated'] is True
    assert v['cluster_id'] == ids['widget']


@pytest.mark.asyncio
async def test_unexclude_leaves_non_excluded_items_untouched(crops, registry, ids) -> None:
    fake = QueryFakeOpenSearch({ITEMS: {'v': _proposal_doc('v')}})
    await _label(crops, fake, registry, 'v', ids['widget'])
    before = dict(fake.docs(ITEMS)['v'])
    assert await _unexclude(crops, fake, ['v']) == {'unexcluded': 1, 'errors': 0}
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
        cluster_id=-2,
        cluster_subid=None,
    )
    fake = QueryFakeOpenSearch({ITEMS: {'old': doc}})
    await _unexclude(crops, fake, ['old'])
    after = fake.docs(ITEMS)['old']
    assert after['class_validated'] is True
    assert after['cluster_id'] == 7


@pytest.mark.asyncio
async def test_undo_label_on_excluded_item_keeps_it_excluded(crops, registry, ids) -> None:
    fake = QueryFakeOpenSearch({ITEMS: {'x': _proposal_doc('x')}})
    await _label(crops, fake, registry, 'x', ids['widget'])
    await _label(crops, fake, registry, 'x', ids['gadget'])
    await _exclude(crops, fake, ['x'])

    await crops.unlabel_crop('x', fake)
    x = fake.docs(ITEMS)['x']
    assert x['class_excluded'] is True
    assert x['cluster_id'] == -2
    assert x['class_validated'] is False
    assert x['class_id'] == ids['widget']

    await _unexclude(crops, fake, ['x'])
    x = fake.docs(ITEMS)['x']
    assert x['class_validated'] is True
    assert x['cluster_id'] == ids['widget']
