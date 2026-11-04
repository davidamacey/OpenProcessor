"""The primary (assigns_class) detector maps a model output index to a registry class by NAME.

A registry seeded out of ONNX order (ids 0, 1, 3 for model indices 0, 1, 2) must give the class
named like the model output, never the entry that happens to share the index. A model name with
no active registry class stays an unlabeled proposal.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from curation.class_by_name_common import GAPPED, IN_ORDER, primary_detector, write_model_labels
from curation.test_class_resolution_by_name import _Entry, _Registry


if TYPE_CHECKING:
    from pathlib import Path

    from src.services.curation.ingest_detect import WholeImageDetector
    from src.services.curation.item_doc import DetectedItem


@pytest.fixture
def labels(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    return write_model_labels(tmp_path, monkeypatch)


def _decode(detector: WholeImageDetector, model_idx: int, score: float = 0.99) -> DetectedItem:
    (item,) = detector.decode_primary_row(
        np.array([1]),
        np.array([[0.1, 0.1, 0.5, 0.5]], dtype=np.float32),
        np.array([score], dtype=np.float32),
        np.array([model_idx], dtype=np.float32),
        1.0,
        (0.0, 0.0),
        640,
    )
    return item


def test_primary_gives_the_registry_class_named_like_the_model_output(labels: str) -> None:
    item = _decode(primary_detector(labels, _Registry(*GAPPED)), 2)

    assert (item.class_id, item.class_name) == (3, 'delta')
    assert item.detector_class_id == 3
    assert item.class_source == 'item_model'


def test_primary_in_model_order_gives_the_same_ids_as_before(labels: str) -> None:
    detector = primary_detector(labels, _Registry(*IN_ORDER))

    assert [(i.class_id, i.class_name) for i in (_decode(detector, k) for k in range(3))] == [
        (0, 'alpha'),
        (1, 'bravo'),
        (2, 'delta'),
    ]


def test_primary_name_without_a_registry_class_stays_an_unlabeled_proposal(labels: str) -> None:
    item = _decode(primary_detector(labels, _Registry(_Entry(0, 'alpha'))), 2)

    assert item.class_id is None
    assert item.class_name is None
    assert item.detector_class_id is None
    assert item.class_source == 'item_low_conf'
    assert item.proposal_name == 'delta'


def test_primary_ignores_a_deprecated_class_with_the_name(labels: str) -> None:
    registry = _Registry(_Entry(0, 'alpha'), _Entry(3, 'delta', deprecated=True))

    assert _decode(primary_detector(labels, registry), 2).class_id is None
