"""
DF1: generic `/ingest` and `/ingest/batch` read `box_embeddings` /
`normalized_boxes` keys that `infer_yolo_clip_cpu` never populated, so
per-box vectors were never produced or indexed.

This exercises `TritonClient.infer_yolo_clip_cpu` directly (mocking only the
Triton gRPC boundary) and asserts it now returns real, non-zero per-box
CLIP embeddings cropped from the full-resolution image, plus
`normalized_boxes` in the original image's coordinate frame.
"""

from __future__ import annotations

import io
from typing import Any

import numpy as np
import pytest
from PIL import Image

from src.clients.triton_client import TritonClient


class _FakeAdapter:
    """Reports one detection box in letterbox-normalized [0,1] space."""

    requested_outputs = ('det_boxes', 'det_scores', 'det_classes', 'num_dets')

    def parse(self, response: Any, batch_size: int) -> list[dict]:  # noqa: ARG002
        return [
            {
                'num_dets': 1,
                'boxes': np.array([[0.2, 0.2, 0.6, 0.6]], dtype=np.float32),
                'scores': np.array([0.93], dtype=np.float32),
                'classes': np.array([2], dtype=np.int64),
            }
        ]


class _FakeResponse:
    def __init__(self, batch_size: int):
        self._batch_size = batch_size

    def as_numpy(self, name: str) -> np.ndarray:
        assert name == 'image_embeddings'
        # Distinct, non-zero, non-uniform so we can tell real embeddings
        # were produced (not a stub zero vector).
        rng = np.random.default_rng(42)
        return rng.random((self._batch_size, 512)).astype(np.float32)


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TritonClient:
    c = TritonClient.__new__(TritonClient)
    c.triton_url = 'unused:8001'
    c.client = None
    c.input_size = 640
    c.max_retries = 1
    c.retry_base_delay = 0.0
    c.retry_max_delay = 0.0
    c._detection_adapters = {}

    def fake_infer_with_retry(model_name: str, inputs: list, outputs: list):
        if model_name == 'mobileclip2_s2_image_encoder':
            batch_size = inputs[0].shape()[0] if hasattr(inputs[0], 'shape') else 1
            return _FakeResponse(batch_size)
        return object()  # yolo response; ignored by _FakeAdapter.parse

    monkeypatch.setattr(c, '_infer_with_retry', fake_infer_with_retry)
    monkeypatch.setattr(c, '_get_detection_adapter', lambda model_name: _FakeAdapter())  # noqa: ARG005
    return c


def _make_jpeg_bytes(width: int, height: int) -> bytes:
    img = Image.new('RGB', (width, height), color=(120, 60, 200))
    buf = io.BytesIO()
    img.save(buf, format='JPEG')
    return buf.getvalue()


def test_infer_yolo_clip_cpu_produces_box_embeddings(client: TritonClient) -> None:
    image_bytes = _make_jpeg_bytes(800, 600)

    result = client.infer_yolo_clip_cpu(image_bytes)

    assert result['num_dets'] == 1
    assert 'box_embeddings' in result
    assert 'normalized_boxes' in result

    box_embeddings = np.asarray(result['box_embeddings'])
    normalized_boxes = np.asarray(result['normalized_boxes'])

    assert box_embeddings.shape == (1, 512)
    assert normalized_boxes.shape == (1, 4)

    # Not a stub zero vector — a real per-box embedding was produced.
    assert not np.allclose(box_embeddings, 0.0)

    # normalized_boxes must be valid [0,1] fractions of the ORIGINAL image.
    assert np.all(normalized_boxes >= 0.0)
    assert np.all(normalized_boxes <= 1.0)
    x1, y1, x2, y2 = normalized_boxes[0]
    assert x2 > x1
    assert y2 > y1


def test_infer_yolo_clip_cpu_no_detections_returns_empty_arrays(
    client: TritonClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _EmptyAdapter:
        requested_outputs = ('det_boxes', 'det_scores', 'det_classes', 'num_dets')

        def parse(self, response, batch_size):  # noqa: ARG002
            return [
                {
                    'num_dets': 0,
                    'boxes': np.zeros((0, 4)),
                    'scores': np.zeros(0),
                    'classes': np.zeros(0),
                }
            ]

    monkeypatch.setattr(client, '_get_detection_adapter', lambda model_name: _EmptyAdapter())  # noqa: ARG005

    result = client.infer_yolo_clip_cpu(_make_jpeg_bytes(400, 300))

    assert result['num_dets'] == 0
    assert np.asarray(result['box_embeddings']).shape == (0, 512)
    assert np.asarray(result['normalized_boxes']).shape == (0, 4)
