"""Detector adapters declare their box frame; ``inverse_letterbox_coords``
takes it instead of guessing from ``boxes.max() > 1.0`` (audit GPU-14)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from src.clients.model_adapters import End2EndNMSAdapter, FusedDetAdapter
from src.utils.affine import inverse_letterbox_coords


# 640x640 image, no padding, scale 1: letterbox frame equals the image frame.
_ARGS = {'orig_shape': (640, 640), 'scale': 1.0, 'padding': (0.0, 0.0)}


def test_overshooting_normalized_box_does_not_flip_the_whole_image_to_pixels() -> None:
    boxes = np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.5, 641 / 640, 0.9]])
    out = inverse_letterbox_coords(boxes, frame='normalized', **_ARGS)
    np.testing.assert_allclose(out[0], [0.1, 0.2, 0.3, 0.4], atol=1e-6)
    assert out[1, 2] == pytest.approx(1.0)


def test_normalized_boxes_at_or_below_one_are_unchanged() -> None:
    boxes = np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.5, 1.0, 0.9]])
    out = inverse_letterbox_coords(boxes, frame='normalized', **_ARGS)
    np.testing.assert_allclose(out, boxes, atol=1e-6)


def test_pixel_boxes_are_divided_by_the_image_size_even_when_all_are_small() -> None:
    # Every value <= 1.0 px: the old guess read these as normalized.
    boxes = np.array([[0.5, 0.5, 1.0, 1.0]])
    out = inverse_letterbox_coords(boxes, frame='pixels', **_ARGS)
    np.testing.assert_allclose(out, boxes / 640, atol=1e-7)


def test_unknown_frame_fails_closed() -> None:
    with pytest.raises(ValueError, match='box frame'):
        inverse_letterbox_coords(
            np.zeros((1, 4)),
            frame='letterbox',  # type: ignore[arg-type]
            **_ARGS,
        )


class _Response:
    def __init__(self, tensors: dict[str, np.ndarray]) -> None:
        self._t = tensors

    def as_numpy(self, name: str) -> np.ndarray:
        return self._t[name]


def test_end2end_adapter_declares_normalized() -> None:
    resp = _Response(
        {
            'num_dets': np.array([[1]]),
            'det_boxes': np.zeros((1, 300, 4), dtype=np.float32),
            'det_scores': np.zeros((1, 300), dtype=np.float32),
            'det_classes': np.zeros((1, 300), dtype=np.int32),
        }
    )
    assert End2EndNMSAdapter().parse(resp, 1)[0]['box_frame'] == 'normalized'


def test_fused_adapter_declares_pixels() -> None:
    fused = np.zeros((1, 300, 6), dtype=np.float32)
    fused[0, 0] = [10, 20, 30, 40, 0.9, 2]
    resp = SimpleNamespace(as_numpy=lambda _n: fused)
    assert FusedDetAdapter('output0').parse(resp, 1)[0]['box_frame'] == 'pixels'
