"""`detect.class_resolution = by_name`: a detection gets the registry class named
like its own label, written like a classifier's label; nothing else changes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from curation.occ_fakes import FakeIngestOpenSearch
from curation.test_ingest_service import FakePEEncoder, FakeTritonPool, _jpeg_bytes
from src.config import CurationConfig, DetectionProfile
from src.services.curation.autolabel.selection import vlm_selection_query
from src.services.curation.ingest import CurationIngestService
from src.services.curation.ingest_class_sources import is_classifier_class_source
from src.services.curation.ingest_policy import (
    DetectFilter,
    IngestPolicy,
    assign_classes_by_name,
    registry_name_index,
)
from src.services.curation.item_doc import DetectedItem


@dataclass
class _Entry:
    class_id: int
    class_name: str
    deprecated: bool = False
    group: str | None = None


@dataclass
class _File:
    classes: list[_Entry] = field(default_factory=list)


class _Registry:
    def __init__(self, *entries: _Entry) -> None:
        self._file = _File(list(entries))

    def load(self) -> _File:
        return self._file

    def get(self, class_id: int) -> _Entry | None:
        return next((c for c in self._file.classes if c.class_id == class_id), None)


def _item(name: str | None, **kw: Any) -> DetectedItem:
    return DetectedItem(
        bbox_pixel=(0, 0, 10, 10),
        score=0.9,
        class_source='item_proposal',
        proposal_name=name,
        **kw,
    )


def test_a_proposal_gets_the_class_named_like_it_and_keeps_its_label() -> None:
    reg = registry_name_index(_Registry(_Entry(4, 'traffic_light'), _Entry(5, 'person')))
    items = [_item('traffic light'), _item('Person'), _item('dog'), _item(None)]
    assert assign_classes_by_name(items, reg, 'item') == 2
    assert [(i.class_id, i.class_name) for i in items] == [
        (4, 'traffic_light'),
        (5, 'person'),
        (None, None),
        (None, None),
    ]
    assert items[0].class_source == 'item_model'
    assert items[0].proposal_name == 'traffic light'  # lineage kept
    assert items[2].class_source == 'item_proposal'


def test_a_deprecated_or_region_class_never_resolves() -> None:
    reg = registry_name_index(
        _Registry(_Entry(1, 'car', deprecated=True), _Entry(2, 'wheel', group='region'))
    )
    items = [_item('car'), _item('wheel')]
    assert assign_classes_by_name(items, reg, 'item') == 0


def test_an_already_classed_or_labeled_item_is_left_alone() -> None:
    reg = registry_name_index(_Registry(_Entry(1, 'car')))
    classed = _item('car', class_id=9, class_name='other')
    assert assign_classes_by_name([classed], reg, 'item') == 0
    assert classed.class_id == 9


def test_the_resolved_source_is_recognised_as_a_classifier_label_the_vlm_skips() -> None:
    assert is_classifier_class_source('item_model')
    query = vlm_selection_query(class_id=None, cluster_id=None, classifier_confidence_skip_vlm=0.8)
    skipped = [
        c
        for c in query['bool']['must_not']
        if 'bool' in c and 'terms' in str(c) and 'class_source' in str(c)
    ]
    assert skipped
    assert 'item_model' in str(skipped)


def _service(policy: IngestPolicy, tmp_path: Any) -> tuple[CurationIngestService, Any]:
    labels = tmp_path / 'labels.txt'
    labels.write_text('gadget\nwidget\n')
    os_fake = FakeIngestOpenSearch()
    profile = DetectionProfile(
        name='item',
        detector_model='proposer',
        assigns_class=False,
        labels_path=str(labels),
        input_size=320,
        batch_limit=8,
    )
    svc = CurationIngestService(
        opensearch=os_fake,
        triton_pool=FakeTritonPool([(0.05, 0.05, 0.6, 0.6, 0.9, 1), (0.1, 0.6, 0.4, 0.9, 0.8, 0)]),
        registry=_Registry(_Entry(7, 'widget')),
        profile=profile,
        pe_encoder=FakePEEncoder(),
        config=CurationConfig(),
        policy=policy,
    )
    return svc, os_fake


@pytest.mark.asyncio
async def test_ingest_resolves_by_name_only_when_the_project_asks(tmp_path: Any) -> None:
    by_name = IngestPolicy(detect=DetectFilter(class_resolution='by_name'))
    svc, os_fake = _service(by_name, tmp_path)
    await svc.ingest_one(_jpeg_bytes(), '/tmp/a.jpg')
    by_label = {d['proposal_name']: d for d in os_fake.items.values()}
    widget, gadget = by_label['widget'], by_label['gadget']
    assert (widget['class_id'], widget['class_name'], widget['class_source']) == (
        7,
        'widget',
        'item_model',
    )
    assert widget['cluster_id'] == 7  # a class-labeled item sits in its class cluster
    assert widget['class_validated'] is False  # a machine label, never a human one
    assert 'class_id' not in gadget
    assert gadget['class_source'] == 'item_proposal'

    default_svc, default_fake = _service(IngestPolicy(), tmp_path)
    await default_svc.ingest_one(_jpeg_bytes(seed=1), '/tmp/b.jpg')
    assert all('class_id' not in d for d in default_fake.items.values())
