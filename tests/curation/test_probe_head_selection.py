"""The probe reads a real class posterior from end-to-end (NMS-free) models.

A YOLO26 checkpoint's default forward returns post-selection rows
``(300, 6)`` = box, confidence, class id. Read as ``(4 + nc, anchors)`` those
rows become a fake 296-way "posterior". The probe must switch such a model to
its one-to-many head, whose output is the per-class score tensor, and must
refuse any tensor whose height is not ``4 + nc``.
"""

from __future__ import annotations

import pytest

from src.services.curation import probe_predictions


torch = pytest.importorskip('torch')


class _Inner:
    def __init__(self, end2end: bool | None) -> None:
        if end2end is not None:
            self.end2end = end2end


class _Model:
    def __init__(self, end2end: bool | None, nc: int = 5) -> None:
        self.model = _Inner(end2end)
        self.names = {i: f'class_{i}' for i in range(nc)}


def test_end2end_model_is_switched_to_its_one_to_many_head() -> None:
    model = _Model(end2end=True)

    probe_predictions._use_class_score_head(model)

    assert model.model.end2end is False


def test_model_without_an_end2end_head_is_left_alone() -> None:
    model = _Model(end2end=None)

    probe_predictions._use_class_score_head(model)

    assert not hasattr(model.model, 'end2end')


def test_post_selection_rows_are_not_read_as_a_posterior() -> None:
    model = _Model(end2end=False, nc=5)
    rows = torch.rand(300, 6)

    result = probe_predictions._summarize_prediction_raw([probe_predictions._RawPreds(rows)], model)

    assert result == (None, 0.0, 0.0, 0.0)


def test_class_score_tensor_yields_a_posterior_over_the_model_classes() -> None:
    model = _Model(end2end=False, nc=5)
    scores = torch.zeros(4 + 5, 10)
    scores[4:, 7] = torch.tensor([0.1, 0.1, 0.9, 0.1, 0.1])

    name, confidence, entropy, margin = probe_predictions._summarize_prediction_raw(
        [probe_predictions._RawPreds(scores)], model
    )

    assert name == 'class_2'
    assert 0.0 < confidence < 1.0
    assert entropy > 0.0
    assert margin > 0.0


def test_confident_class_scores_give_a_confident_posterior() -> None:
    """Per-class sigmoid scores are normalized by their sum; a softmax over
    values already in [0, 1] flattens every posterior toward uniform."""
    model = _Model(end2end=False, nc=5)
    scores = torch.zeros(4 + 5, 3)
    scores[4:, 1] = torch.tensor([0.9, 0.05, 0.05, 0.05, 0.05])

    name, confidence, _entropy, margin = probe_predictions._summarize_prediction_raw(
        [probe_predictions._RawPreds(scores)], model
    )

    assert name == 'class_0'
    assert confidence == pytest.approx(0.9 / 1.1, abs=1e-6)
    assert margin == pytest.approx((0.9 - 0.05) / 1.1, abs=1e-6)
