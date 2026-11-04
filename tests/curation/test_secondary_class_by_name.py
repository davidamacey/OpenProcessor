"""The secondary detector maps a model output index to a registry class by NAME.

A registry with a gap (ids 0, 1, 3 for model indices 0, 1, 2) must give the class named like the
model output. An index with no name, or a name with no active registry class, leaves the
primary's class unchanged.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from curation.class_by_name_common import GAPPED, IN_ORDER, primary_detector, write_model_labels
from curation.test_class_resolution_by_name import _Entry, _Registry
from src.config import DetectionProfile
from src.services.curation.item_doc import DetectedItem


if TYPE_CHECKING:
    from pathlib import Path

    from src.services.curation.ingest_detect import WholeImageDetector


@pytest.fixture
def labels(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    return write_model_labels(tmp_path, monkeypatch)


def _secondary_result(
    detector: WholeImageDetector, monkeypatch: pytest.MonkeyPatch, model_idx: int
) -> DetectedItem:
    monkeypatch.setattr(
        'src.services.detection.ensemble_nms.apply_ensemble_nms',
        lambda *_a, **_k: [
            [{'box': (10.0, 10.0, 50.0, 50.0), 'score': 0.8, 'class_id': model_idx}]
        ],
    )
    item = DetectedItem(
        bbox_pixel=(10.0, 10.0, 50.0, 50.0),
        score=0.5,
        class_id=9,
        class_name='from_primary',
        class_source='item_model',
    )
    detector.resolve_with_secondary([item], np.zeros((4, 8), dtype=np.float32), 1.0, (0.0, 0.0))
    return item


def _with_secondary(labels: str, registry: _Registry) -> WholeImageDetector:
    sec = DetectionProfile(name='secondary', detector_model='sec', labels_path=labels)
    detector = primary_detector(labels, registry)
    detector.secondary_profile = sec
    return detector


def test_secondary_gives_the_registry_class_named_like_the_model_output(
    labels: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    item = _secondary_result(_with_secondary(labels, _Registry(*GAPPED)), monkeypatch, 2)

    assert (item.class_id, item.class_name) == (3, 'delta')
    assert item.class_source == 'secondary_model'
    assert item.score == pytest.approx(0.8)


def test_secondary_in_model_order_gives_the_same_ids_as_before(
    labels: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    item = _secondary_result(_with_secondary(labels, _Registry(*IN_ORDER)), monkeypatch, 2)

    assert (item.class_id, item.class_name) == (2, 'delta')


def test_secondary_name_without_a_registry_class_leaves_the_primary_class(
    labels: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    item = _secondary_result(_with_secondary(labels, _Registry(_Entry(0, 'alpha'))), monkeypatch, 2)

    assert (item.class_id, item.class_name, item.class_source) == (9, 'from_primary', 'item_model')


def test_secondary_index_without_a_name_leaves_the_primary_class(
    labels: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    item = _secondary_result(_with_secondary(labels, _Registry(*GAPPED)), monkeypatch, 40)

    assert item.class_id == 9
