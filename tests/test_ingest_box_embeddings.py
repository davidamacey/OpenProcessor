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


class _MultiSmallBoxAdapter:
    """Several small, differently-sized boxes on a 640x480 (COCO-shaped)
    image -- reproduces F-28: distant/small detection crops whose
    center_crop_cpu resize-then-crop math previously undershot 256px on
    some boxes but not others, so np.stack across the batch raised
    'all input arrays must have the same shape'."""

    requested_outputs = ('det_boxes', 'det_scores', 'det_classes', 'num_dets')

    # XYXY in [0,1] letterbox-normalized space, chosen so the boxes crop out
    # to a range of small absolute pixel sizes on a 640x480 image (tens of
    # px on a side -- the regime that hit the float-truncation bug).
    _BOXES = np.array(
        [
            [0.10, 0.10, 0.14, 0.16],
            [0.30, 0.30, 0.36, 0.34],
            [0.50, 0.20, 0.58, 0.30],
            [0.70, 0.60, 0.80, 0.78],
            [0.05, 0.80, 0.09, 0.86],
        ],
        dtype=np.float32,
    )

    def parse(self, response: Any, batch_size: int) -> list[dict]:  # noqa: ARG002
        n = len(self._BOXES)
        return [
            {
                'num_dets': n,
                'boxes': self._BOXES,
                'scores': np.full(n, 0.9, dtype=np.float32),
                'classes': np.zeros(n, dtype=np.int64),
            }
        ]


def test_infer_yolo_clip_cpu_stacks_many_small_boxes_on_a_coco_image(
    client: TritonClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """F-28: reproduces the reported failure on an ordinary 640x480 image
    with several small detections -- previously raised ValueError from
    np.stack when one box's center-cropped tensor came out smaller than
    256x256 due to float truncation in the resize-then-crop math."""
    monkeypatch.setattr(
        client,
        '_get_detection_adapter',
        lambda model_name: _MultiSmallBoxAdapter(),  # noqa: ARG005
    )

    result = client.infer_yolo_clip_cpu(_make_jpeg_bytes(640, 480))

    n = len(_MultiSmallBoxAdapter._BOXES)
    assert result['num_dets'] == n
    box_embeddings = np.asarray(result['box_embeddings'])
    normalized_boxes = np.asarray(result['normalized_boxes'])
    assert box_embeddings.shape == (n, 512)
    assert normalized_boxes.shape == (n, 4)


@pytest.mark.parametrize(
    ('mode', 'size'),
    [
        ('L', (640, 480)),  # grayscale
        ('RGBA', (640, 480)),  # alpha channel
        ('CMYK', (640, 480)),  # CMYK (JPEG supports it)
        ('RGB', (8, 6)),  # tiny image, smaller than the 256px crop target
    ],
)
def test_infer_yolo_clip_cpu_handles_every_image_mode(
    client: TritonClient, mode: str, size: tuple[int, int]
) -> None:
    """F-28: grayscale / alpha / CMYK / tiny inputs must decode to a plain
    RGB array and produce a well-formed box embedding, not raise."""
    img = Image.new(mode, size, color=200 if mode == 'L' else (200, 60, 60, 255)[: len(mode)])
    buf = io.BytesIO()
    # RGBA can't be saved as JPEG (no alpha channel support); PNG round-trips
    # it losslessly and still exercises the same decode->convert('RGB') path.
    img.save(buf, format='PNG' if mode == 'RGBA' else 'JPEG')

    result = client.infer_yolo_clip_cpu(buf.getvalue())

    assert result['num_dets'] == 1
    box_embeddings = np.asarray(result['box_embeddings'])
    normalized_boxes = np.asarray(result['normalized_boxes'])
    assert box_embeddings.shape == (1, 512)
    assert normalized_boxes.shape == (1, 4)


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
