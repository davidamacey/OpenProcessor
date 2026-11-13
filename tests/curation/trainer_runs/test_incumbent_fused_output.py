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
    def __init__(
        self, outputs: list[dict[str, Any]], document: dict[str, Any] | None = None
    ) -> None:
        self._outputs = outputs
        self._document = document

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self._document if self._document is not None else {'outputs': self._outputs}


def _metadata(datatype: str) -> _Response:
    return _Response(
        [], {'name': 'incumbent', 'inputs': [{'name': 'images', 'datatype': datatype}]}
    )


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
    monkeypatch.setattr(requests, 'get', lambda *_a, **_k: _metadata('FP32'))

    (dets,) = incumbent_compare.run_incumbent_inference([_image(tmp_path, 1000, 500)], 'incumbent')

    assert sent[0]['outputs'] == [{'name': 'output0'}]
    assert len(dets) == 1
    assert dets[0][:4] == pytest.approx((100.0, 62.5, 500.0, 312.5))
    assert dets[0][4] == pytest.approx(0.9)
    assert dets[0][5] == 2


def test_a_uint8_incumbent_is_sent_uint8_pixels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import requests

    sent: list[dict[str, Any]] = []

    def post(_url: str, json: dict[str, Any], timeout: float) -> _Response:
        sent.append(json)
        return _Response([{'name': 'output0', 'data': [0.0] * 6}])

    monkeypatch.setattr(requests, 'post', post)
    monkeypatch.setattr(requests, 'get', lambda *_a, **_k: _metadata('UINT8'))

    incumbent_compare.run_incumbent_inference([_image(tmp_path, 640, 640)], 'incumbent')

    tensor = sent[0]['inputs'][0]
    assert tensor['datatype'] == 'UINT8'
    assert all(isinstance(v, int) and 0 <= v <= 255 for v in tensor['data'][:50])


def test_an_input_dtype_the_comparison_cannot_feed_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import requests

    monkeypatch.setattr(requests, 'get', lambda *_a, **_k: _metadata('INT64'))

    with pytest.raises(incumbent_compare.IncumbentInferenceError, match='INT64'):
        incumbent_compare.run_incumbent_inference([_image(tmp_path, 640, 640)], 'incumbent')


def test_an_inference_error_fails_the_comparison_instead_of_an_empty_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import requests

    def post(*_a: Any, **_k: Any) -> _Response:
        raise requests.ConnectionError('triton down')

    monkeypatch.setattr(requests, 'post', post)
    monkeypatch.setattr(requests, 'get', lambda *_a, **_k: _metadata('FP32'))

    with pytest.raises(incumbent_compare.IncumbentInferenceError):
        incumbent_compare.run_incumbent_inference([_image(tmp_path, 640, 640)], 'incumbent')


def test_a_failed_incumbent_sweep_propagates_to_the_trainer_eval_handler(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*_a: Any, **_k: Any) -> list[Any]:
        raise incumbent_compare.IncumbentInferenceError('x')

    image = _image(tmp_path, 64, 64)
    monkeypatch.setattr(incumbent_compare, 'resolve_incumbent_model_name', lambda: 'incumbent')
    monkeypatch.setattr(incumbent_compare, 'list_test_images', lambda _p: ([image], [image]))
    monkeypatch.setattr(incumbent_compare, 'run_incumbent_inference', boom)

    with pytest.raises(incumbent_compare.IncumbentInferenceError):
        incumbent_compare.build_compare_block(
            save_dir=tmp_path,
            candidate_per_class=[],
            candidate_summary={},
            data_yaml_path=tmp_path / 'data.yaml',
        )


def test_a_model_without_output0_fails_the_comparison(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import requests

    monkeypatch.setattr(
        requests, 'post', lambda *_a, **_k: _Response([{'name': 'det_boxes', 'data': []}])
    )
    monkeypatch.setattr(requests, 'get', lambda *_a, **_k: _metadata('FP32'))

    with pytest.raises(incumbent_compare.IncumbentInferenceError, match='output0'):
        incumbent_compare.run_incumbent_inference([_image(tmp_path, 640, 640)], 'old')
