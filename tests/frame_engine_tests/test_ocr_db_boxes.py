"""``frame_engine/ocr/db_boxes.py`` against PaddleX release/3.3's ``DBPostProcess`` and
``SortQuadBoxes`` (``db_reference.npz``: synthetic probability maps, and the boxes, scores and
order the reference made of them, produced by running the reference's own ``DBPostProcess`` and
``SortQuadBoxes`` code, ``processors.py`` sha256 7b714e1a...a5de)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from frame_engine.ocr.db_boxes import DbParams, boxes_from_map, sort_reading_order


FIXTURE = np.load(Path(__file__).with_name('db_reference.npz'))


@pytest.mark.parametrize('dilated', [False, True])
@pytest.mark.parametrize('k', range(4))
def test_boxes_and_scores_equal_the_reference(k: int, dilated: bool) -> None:
    pred = FIXTURE[f'm{k}'].astype(np.float32)
    boxes, scores = boxes_from_map(pred, 800, 480, DbParams(use_dilation=dilated))
    d = int(dilated)
    assert np.array_equal(boxes, FIXTURE[f'm{k}_d{d}_boxes'])
    assert np.allclose(scores, FIXTURE[f'm{k}_d{d}_scores'], atol=1e-6)


def test_the_fixture_has_boxes_to_compare() -> None:
    assert sum(len(FIXTURE[f'm{k}_d0_boxes']) for k in range(4)) >= 8


def test_an_empty_map_gives_no_boxes() -> None:
    boxes, scores = boxes_from_map(np.zeros((64, 64), np.float32), 640, 640, DbParams())
    assert boxes.shape == (0, 4, 2)
    assert scores.shape == (0,)


def test_reading_order_equals_the_reference() -> None:
    order = sort_reading_order(FIXTURE['sort_quads'])
    assert np.array_equal(order, FIXTURE['sort_order'])
