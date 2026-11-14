"""``frame_engine/ocr/batching.py``: PaddleOCR's recogniser batching recipe.

The oracle is the loop of ``PaddleOCR/tools/infer/predict_rec.py`` ``TextRecognizer.__call__``
(release/2.9): sort the crops by
width over height, take ``rec_batch_num`` at a time, and pad each batch to the widest crop in it
(at least 320 pixels, the model's ``rec_image_shape`` ``[3, 48, 320]``).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from frame_engine.ocr.batching import REC_HEIGHT, Batch, plan_batches, resized_width


def _reference(ratios: list[float], batch_num: int, max_width: int) -> list[tuple[list[int], int]]:
    order = np.argsort(np.array(ratios))
    out = []
    for beg in range(0, len(ratios), batch_num):
        idx = [int(order[j]) for j in range(beg, min(len(ratios), beg + batch_num))]
        max_wh_ratio = 320 / 48
        for i in idx:
            max_wh_ratio = max(max_wh_ratio, ratios[i])
        out.append((idx, min(int(48 * max_wh_ratio), max_width)))
    return out


@pytest.mark.parametrize('batch_num', [1, 4, 16, 64])
def test_batches_equal_the_reference_loop(batch_num: int) -> None:
    rng = np.random.default_rng(batch_num)
    ratios = list(np.exp(rng.normal(1.4, 1.0, 100)))  # distinct: the reference sort is not stable
    got = plan_batches(ratios, batch_num, max_width=2048)
    assert [(b.indices, b.width) for b in got] == _reference(ratios, batch_num, 2048)


def test_every_crop_is_in_exactly_one_batch_and_batches_are_bounded() -> None:
    ratios = [0.5, 12.0, 3.0, 3.0, 80.0, 1.0, 6.7]
    batches = plan_batches(ratios, 3, max_width=2048)
    assert sorted(i for b in batches for i in b.indices) == list(range(len(ratios)))
    assert all(len(b.indices) <= 3 for b in batches)
    assert all(320 <= b.width <= 2048 for b in batches)


def test_a_wide_crop_is_squeezed_into_the_cap_not_dropped() -> None:
    [batch] = plan_batches([100.0], 16, max_width=2048)
    assert batch.width == 2048
    assert resized_width(100.0, batch.width) == 2048  # squeezed to the batch width


def test_resized_width_keeps_the_aspect_ratio_inside_the_batch_width() -> None:
    assert resized_width(2.0, 320) == math.ceil(REC_HEIGHT * 2.0)
    assert resized_width(9.0, 320) == 320
    assert isinstance(Batch([0], 320), Batch)
