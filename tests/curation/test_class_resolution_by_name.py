"""`detect.class_resolution = by_name` (the default): a detection gets the registry class named
like its own label, written as the detector's own machine label (not a confident one); a class the
registry lacks is created by name; nothing else changes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import pytest
from _embedding_registry_fake import DEFAULT_WRITE_FIELDS

from curation.ingest_fakes import FakePEEncoder, FakeTritonPool, _jpeg_bytes
from curation.occ_fakes import FakeIngestOpenSearch
from src.config import CurationConfig, DetectionProfile
from src.domain.ingest_class_sources import (
    DETECTOR_CLASS_SOURCE,
    confident_class_sources,
    is_classifier_class_source,
)
from src.services.curation.autolabel.selection import vlm_selection_query
from src.services.curation.ingest import CurationIngestService
from src.services.curation.ingest_policy import (
    DetectFilter,
    IngestPolicy,
    assign_classes_by_name,
    registry_name_index,
)
from src.services.curation.item_doc import DetectedItem


@pytest.fixture(autouse=True)
def _labels_inside_the_model_repository(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('OP_TRITON_MODEL_REPO', str(tmp_path))


@dataclass
class _Entry:
    class_id: int
    class_name: str
    deprecated: bool = False
    group: str | None = None


class _Registry:
    """A registry document snapshot: ``classes`` and ``get``."""

    def __init__(self, *entries: _Entry) -> None:
        self.classes = list(entries)

    def get(self, class_id: int) -> _Entry | None:
        return next((c for c in self.classes if c.class_id == class_id), None)


def _item(name: str | None, **kw: Any) -> DetectedItem:
    return DetectedItem(
        bbox_pixel=(0, 0, 10, 10),
        score=0.9,
        class_source='item_proposal',
        proposal_name=name,
        **kw,
    )


def test_a_proposal_gets_the_class_named_like_it_and_keeps_its_label() -> None:
    reg = registry_name_index(
        cast('Any', _Registry(_Entry(4, 'traffic_light'), _Entry(5, 'person')))
    )
    items = [_item('traffic light'), _item('Person'), _item('dog'), _item(None)]
    assert assign_classes_by_name(items, reg) == 2
    assert [(i.class_id, i.class_name) for i in items] == [
        (4, 'traffic_light'),
        (5, 'person'),
        (None, None),
        (None, None),
    ]
    assert items[0].class_source == DETECTOR_CLASS_SOURCE
    assert items[0].proposal_name == 'traffic light'  # lineage kept
    assert items[2].class_source == 'item_proposal'


def test_a_deprecated_or_region_class_never_resolves() -> None:
    reg = registry_name_index(
        cast(
            'Any', _Registry(_Entry(1, 'car', deprecated=True), _Entry(2, 'wheel', group='region'))
        )
    )
    items = [_item('car'), _item('wheel')]
    assert assign_classes_by_name(items, reg) == 0


def test_a_detection_below_the_primary_floor_stays_a_low_confidence_proposal() -> None:
    reg = registry_name_index(cast('Any', _Registry(_Entry(1, 'car'))))
    weak = _item('car')
    weak.class_source = 'item_low_conf'
    assert assign_classes_by_name([weak], reg, 'item_low_conf') == 0
    assert (weak.class_id, weak.class_source) == (None, 'item_low_conf')


def test_an_already_classed_or_labeled_item_is_left_alone() -> None:
    reg = registry_name_index(cast('Any', _Registry(_Entry(1, 'car'))))
    classed = _item('car', class_id=9, class_name='other')
    assert assign_classes_by_name([classed], reg) == 0
    assert classed.class_id == 9


def test_the_resolved_source_is_a_guess_the_vlm_and_clustering_still_see() -> None:
    """The detector's class is not a classifier label: the VLM may still label the item and it
    stays in the clustering pool."""
    assert not is_classifier_class_source(DETECTOR_CLASS_SOURCE)
    assert DETECTOR_CLASS_SOURCE not in confident_class_sources()
    query = vlm_selection_query(
        embedding=DEFAULT_WRITE_FIELDS,
        class_id=None,
        cluster_uid=None,
        classifier_confidence_skip_vlm=0.8,
    )
    assert DETECTOR_CLASS_SOURCE not in str(query['bool']['must_not'])


class _Ensurer:
    """Stands in for the registry write: remembers what was asked and hands out the next id."""

    def __init__(self) -> None:
        self.asked: list[tuple[str, str]] = []

    async def __call__(self, name: str, label: str) -> tuple[int, str]:
        self.asked.append((name, label))
        return 100 + len(self.asked), name


def _service(
    policy: IngestPolicy,
    tmp_path: Any,
    *,
    ensurer: _Ensurer | None = None,
    registry: _Registry | None = None,
) -> tuple[CurationIngestService, Any]:
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
        registry=cast('Any', registry or _Registry(_Entry(7, 'widget'))),
        profile=profile,
        pe_encoder=FakePEEncoder(),
        config=CurationConfig(),
        policy=policy,
        class_ensurer=ensurer,
    )
    return svc, os_fake


@pytest.mark.asyncio
async def test_ingest_gives_the_detector_class_by_default(tmp_path: Any) -> None:
    svc, os_fake = _service(IngestPolicy(), tmp_path)
    await svc.ingest_one(_jpeg_bytes(), '/tmp/a.jpg')
    by_label = {d['proposal_name']: d for d in os_fake.items.values()}
    widget, gadget = by_label['widget'], by_label['gadget']
    assert (widget['class_id'], widget['class_name'], widget['class_source']) == (
        7,
        'widget',
        DETECTOR_CLASS_SOURCE,
    )
    assert 'cluster_id' not in widget  # a class label places nothing in a cluster
    assert widget['class_validated'] is False  # a machine label, never a human one
    # No ensurer wired and no registry class of that name: left a proposal, not guessed.
    assert 'class_id' not in gadget
    assert gadget['class_source'] == 'item_proposal'


@pytest.mark.asyncio
async def test_a_project_can_keep_detections_as_proposals(tmp_path: Any) -> None:
    ensurer = _Ensurer()
    keep = IngestPolicy(detect=DetectFilter(class_resolution='proposal'))
    svc, os_fake = _service(keep, tmp_path, ensurer=ensurer)
    await svc.ingest_one(_jpeg_bytes(), '/tmp/a.jpg')
    assert all('class_id' not in d for d in os_fake.items.values())
    assert ensurer.asked == []


@pytest.mark.asyncio
async def test_ingest_creates_the_missing_class_by_name_once(tmp_path: Any) -> None:
    ensurer = _Ensurer()
    svc, os_fake = _service(IngestPolicy(), tmp_path, ensurer=ensurer)
    await svc.ingest_one(_jpeg_bytes(), '/tmp/a.jpg')
    await svc.ingest_one(_jpeg_bytes(seed=1), '/tmp/b.jpg')
    # 'widget' is in the registry already; 'gadget' is not, and is created once per service.
    assert ensurer.asked == [('gadget', 'gadget')]
    gadgets = [d for d in os_fake.items.values() if d['proposal_name'] == 'gadget']
    assert len(gadgets) == 2
    assert {(d['class_id'], d['class_name'], d['class_source']) for d in gadgets} == {
        (101, 'gadget', DETECTOR_CLASS_SOURCE)
    }
    assert all(d['detector_class_id'] == 101 for d in gadgets)


@pytest.mark.asyncio
async def test_a_deprecated_or_region_class_name_is_not_created_again(tmp_path: Any) -> None:
    ensurer = _Ensurer()
    registry = _Registry(_Entry(7, 'widget'), _Entry(3, 'gadget', deprecated=True))
    svc, os_fake = _service(IngestPolicy(), tmp_path, ensurer=ensurer, registry=registry)
    await svc.ingest_one(_jpeg_bytes(), '/tmp/a.jpg')
    assert ensurer.asked == []
    gadget = next(d for d in os_fake.items.values() if d['proposal_name'] == 'gadget')
    assert 'class_id' not in gadget
