"""The incumbent comparison reads the fused ``output0`` tensor and maps its letterbox-pixel
boxes back to the image's pixels."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import incumbent_compare
import numpy as np
import pytest


if TYPE_CHECKING:
    from pathlib import Path


class _Response:
    def __init__(self, outputs: list[dict[str, Any]]) -> None:
        self._outputs = outputs

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return {'outputs': self._outputs}


def _image(tmp_path: Path, w: int, h: int) -> Path:
    import cv2

    path = tmp_path / 'img.jpg'
    cv2.imwrite(str(path), np.full((h, w, 3), 90, dtype=np.uint8))
    return path


def test_the_request_asks_for_output0_and_boxes_return_in_image_pixels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import requests

    rows = np.zeros((300, 6), dtype=np.float32)
    # Letterbox pixels of a 1000x500 image in a 640 frame: scale 0.64, 160 px of vertical pad.
    rows[0] = (64, 200, 320, 360, 0.9, 2)
    rows[1] = (10, 10, 50, 50, 0.0, 1)  # padding: score 0
    sent: list[dict[str, Any]] = []

    def post(_url: str, json: dict[str, Any], timeout: float) -> _Response:
        sent.append(json)
        return _Response(
            [{'name': 'output0', 'shape': [1, 300, 6], 'data': rows.flatten().tolist()}]
        )

    monkeypatch.setattr(requests, 'post', post)

    (dets,) = incumbent_compare.run_incumbent_inference([_image(tmp_path, 1000, 500)], 'incumbent')

    assert sent[0]['outputs'] == [{'name': 'output0'}]
    assert len(dets) == 1
    assert dets[0][:4] == pytest.approx((100.0, 62.5, 500.0, 312.5))
    assert dets[0][4:] == (pytest.approx(0.9), 2)


def test_a_model_without_output0_yields_no_detections_for_the_image(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import requests

    monkeypatch.setattr(
        requests, 'post', lambda *_a, **_k: _Response([{'name': 'det_boxes', 'data': []}])
    )

    assert incumbent_compare.run_incumbent_inference([_image(tmp_path, 640, 640)], 'old') == [[]]
