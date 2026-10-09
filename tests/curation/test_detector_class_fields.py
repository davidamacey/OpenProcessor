"""The detector's own class stays queryable: ``detector_class_name`` (the
normalized label), ``detector_confidence`` (its raw score) and, only when that
label resolved to a registry class, ``detector_class_id``.

Written by ingest from the detection itself. No other writer (VLM, human,
classifier) touches them, so "detector said X, VLM said Y" survives any later
relabel.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from curation.occ_fakes import FakeIngestOpenSearch
from curation.test_class_resolution_by_name import _Entry, _Registry
from curation.test_ingest_service import FakePEEncoder, FakeTritonPool, _jpeg_bytes
from src.config import CurationConfig, DetectionProfile
from src.services.curation.class_label import ItemLabel, class_label_update
from src.services.curation.ingest import CurationIngestService
from src.services.curation.ingest_policy import DetectFilter, IngestPolicy


if TYPE_CHECKING:
    from pathlib import Path

DETECTIONS = [(0.05, 0.05, 0.6, 0.6, 0.9, 1), (0.1, 0.6, 0.4, 0.9, 0.8, 0)]
FIELDS = ('detector_class_name', 'detector_class_id', 'detector_confidence')


def _service(
    tmp_path: Path, *, policy: IngestPolicy, assigns_class: bool, registry: _Registry
) -> tuple[CurationIngestService, FakeIngestOpenSearch]:
    labels = tmp_path / 'labels.txt'
    labels.write_text('gadget\nTraffic Light\n')
    os_fake = FakeIngestOpenSearch()
    profile = DetectionProfile(
        name='item',
        detector_model='proposer',
        assigns_class=assigns_class,
        labels_path=str(labels),
        confidence_floor=0.85,
        input_size=320,
        batch_limit=8,
    )
    svc = CurationIngestService(
        opensearch=os_fake,
        triton_pool=FakeTritonPool(DETECTIONS),
        registry=registry,
        profile=profile,
        pe_encoder=FakePEEncoder(),
        config=CurationConfig(),
        policy=policy,
    )
    return svc, os_fake


def _by_label(os_fake: FakeIngestOpenSearch) -> dict[str, dict[str, Any]]:
    return {d['detector_class_name']: d for d in os_fake.items.values()}


@pytest.mark.asyncio
async def test_a_proposal_keeps_the_detector_label_score_and_no_registry_id(tmp_path: Path) -> None:
    svc, os_fake = _service(
        tmp_path, policy=IngestPolicy(), assigns_class=False, registry=_Registry(_Entry(7, 'x'))
    )
    await svc.ingest_one(_jpeg_bytes(), '/tmp/a.jpg')
    docs = _by_label(os_fake)
    assert set(docs) == {'gadget', 'traffic_light'}  # normalized: the registry slug form
    light = docs['traffic_light']
    assert light['detector_confidence'] == pytest.approx(0.9)
    assert light['proposal_name'] == 'Traffic Light'  # the raw label stays as lineage
    # A proposer's class ids are not registry ids: never stored as one.
    assert 'detector_class_id' not in light
    assert 'class_id' not in light


@pytest.mark.asyncio
async def test_by_name_resolution_records_the_registry_id_it_resolved_to(tmp_path: Path) -> None:
    svc, os_fake = _service(
        tmp_path,
        policy=IngestPolicy(detect=DetectFilter(class_resolution='by_name')),
        assigns_class=False,
        registry=_Registry(_Entry(4, 'traffic_light')),
    )
    await svc.ingest_one(_jpeg_bytes(), '/tmp/a.jpg')
    docs = _by_label(os_fake)
    assert docs['traffic_light']['detector_class_id'] == 4
    assert docs['traffic_light']['class_id'] == 4
    assert 'detector_class_id' not in docs['gadget']  # no registry class of that name
    assert docs['gadget']['detector_confidence'] == pytest.approx(0.8)


@pytest.mark.asyncio
async def test_a_class_assigning_detector_below_its_floor_still_records_what_it_said(
    tmp_path: Path,
) -> None:
    registry = _Registry(_Entry(0, 'gadget'), _Entry(1, 'traffic_light'))
    svc, os_fake = _service(tmp_path, policy=IngestPolicy(), assigns_class=True, registry=registry)
    await svc.ingest_one(_jpeg_bytes(), '/tmp/a.jpg')
    docs = _by_label(os_fake)
    # 0.9 >= floor 0.85: labeled. 0.8 < floor: left an unlabeled low-confidence proposal.
    assert (docs['traffic_light']['class_id'], docs['traffic_light']['detector_class_id']) == (1, 1)
    assert 'class_id' not in docs['gadget']
    assert docs['gadget']['detector_class_id'] == 0
    assert docs['gadget']['class_source'] == 'item_low_conf'


def _stored_item() -> dict[str, Any]:
    return {
        'crop_id': 'c1',
        'class_id': 3,
        'class_name': 'widget',
        'class_source': 'item_model',
        'label_source': 'item_model',
        'class_validated': False,
        'confidence': 0.9,
        'pe_embedding': [0.1],
        'detector_class_name': 'gadget',
        'detector_class_id': 5,
        'detector_confidence': 0.42,
    }


@pytest.mark.asyncio
async def test_a_vlm_relabel_leaves_the_detector_fields_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import curation.test_vlm_empty_class_answer as vlm_test

    original = vlm_test._item
    monkeypatch.setattr(
        vlm_test,
        '_item',
        lambda cid, **extra: original(
            cid,
            detector_class_name='gadget',
            detector_class_id=5,
            detector_confidence=0.42,
            **extra,
        ),
    )
    docs = await vlm_test._run_label_batch(
        tmp_path, monkeypatch, {'c_label': vlm_test._pred('c_label', 'gadget', conf='high')}
    )
    doc = docs['c_label']
    assert (doc['class_name'], doc['class_source']) == ('gadget', 'vlm')  # the VLM did write
    assert (doc['detector_class_name'], doc['detector_class_id'], doc['detector_confidence']) == (
        'gadget',
        5,
        0.42,
    )


def test_a_human_label_write_does_not_carry_detector_fields() -> None:
    label = ItemLabel.human(class_id=3, class_name='widget', label_source='human', writer='test')
    update = class_label_update(_stored_item(), label)
    assert not set(FIELDS) & update.keys()
    assert update['class_source'] == 'human'  # and it did write the label


@pytest.mark.asyncio
async def test_re_ingest_keeps_a_human_label_and_does_not_drop_detector_fields(
    tmp_path: Path,
) -> None:
    svc, os_fake = _service(
        tmp_path, policy=IngestPolicy(), assigns_class=False, registry=_Registry(_Entry(7, 'x'))
    )
    await svc.ingest_one(_jpeg_bytes(), '/tmp/a.jpg')
    crop_id, doc = next((k, v) for k, v in os_fake.items.items() if v['proposal_name'] == 'gadget')
    os_fake.items[crop_id] = {
        **doc,
        'class_id': 7,
        'class_name': 'widget',
        'class_source': 'human',
        'label_source': 'human',
        'class_validated': True,
    }
    os_fake.images.clear()  # defeat the duplicate-image shortcut
    await svc.ingest_one(_jpeg_bytes(), '/tmp/a.jpg')
    after = os_fake.items[crop_id]
    assert (after['class_name'], after['class_source']) == ('widget', 'human')
    assert (after['detector_class_name'], after['detector_confidence']) == (
        'gadget',
        pytest.approx(0.8),
    )


def test_the_fields_are_mapped_and_served() -> None:
    from src.clients.curation_opensearch.bodies_core import _items_body
    from src.services.curation.wire import serialize_item

    props = _items_body()['mappings']['properties']
    assert props['detector_class_name'] == {'type': 'keyword'}
    assert props['detector_class_id'] == {'type': 'integer'}
    assert props['detector_confidence'] == {'type': 'float'}
    served = serialize_item(_stored_item())
    assert (
        served['detector_class_name'],
        served['detector_class_id'],
        served['detector_confidence'],
    ) == ('gadget', 5, 0.42)
    bare = serialize_item({'crop_id': 'd'})
    assert [bare[f] for f in FIELDS] == [None, None, None]


@pytest.mark.asyncio
async def test_mapping_migration_adds_the_fields_to_existing_indexes() -> None:
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from src.clients.curation_opensearch.items_extra import ensure_items_detector_fields

    client = SimpleNamespace(indices=SimpleNamespace(put_mapping=AsyncMock(return_value={})))
    result = await ensure_items_detector_fields(client)
    assert set(result['fields_added']) == set(FIELDS)
