"""``/ocr`` client frames: detector input geometry (GPU-1) and one oriented decode (GPU-3a).

The detector map is read back by the BLS (``DBPostProcess``) as ``src / pred.shape``, so the map
must cover the whole image: stretching each axis to a multiple of 32 keeps that exact, zero
padding does not.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.clients import triton_ocr


BLS_DIR = Path(__file__).resolve().parents[1] / 'models' / 'ocr_pipeline' / '1'
TEXT_BGR = (0, 0, 255)


class _Response:
    def as_numpy(self, name: str) -> np.ndarray:
        return {
            'num_texts': np.array([0], dtype=np.int32),
            'text_boxes': np.zeros((128, 8), np.float32),
            'text_boxes_normalized': np.zeros((128, 4), np.float32),
            'texts': np.array([b''] * 128, dtype=object),
            'text_scores': np.zeros(128, np.float32),
            'rec_scores': np.zeros(128, np.float32),
        }[name]


def _png(arr: np.ndarray) -> bytes:
    ok, buf = cv2.imencode('.png', arr)
    assert ok
    return buf.tobytes()


def _sent_inputs(image: bytes | np.ndarray) -> dict[str, np.ndarray]:
    sent: dict[str, np.ndarray] = {}

    def infer(_model: str, inputs: list, _outputs: list) -> _Response:
        dtypes = {'UINT8': np.uint8, 'INT32': np.int32}
        for inp in inputs:
            sent[inp.name()] = np.frombuffer(
                inp._get_content(), dtype=dtypes[inp.datatype()]
            ).reshape(inp.shape())
        return _Response()

    data = image if isinstance(image, bytes) else _png(image)
    triton_ocr.infer_ocr(infer, data, lambda _n: 'UINT8')
    return sent


def _text_band_image(w: int, h: int) -> np.ndarray:
    img = np.full((h, w, 3), 255, np.uint8)
    img[200:221, 100:540] = TEXT_BGR
    return img


def _box_centre_y(image: np.ndarray) -> float:
    """Run the detector input through DBPostProcess as the BLS does, return the box centre y."""
    sys.path.insert(0, str(BLS_DIR))
    try:
        from ocr_postprocess import DBPostProcess
    finally:
        sys.path.remove(str(BLS_DIR))
    det = _sent_inputs(image)['ocr_images']  # CHW BGR
    pred = ((det[2] > 200) & (det[1] < 50)).astype(np.float32)
    boxes, _ = DBPostProcess(unclip_ratio=1.6)(pred, image.shape[:2])
    assert len(boxes) == 1
    return float(boxes[0][:, 1].mean())


class TestDetectorFrame:
    def test_padded_size_centre_matches_the_text(self) -> None:
        # 427 is not a multiple of 32 once resized: padding shifted the box ~19 px up.
        assert _box_centre_y(_text_band_image(640, 427)) == pytest.approx(210, abs=1)

    def test_detector_input_covers_the_whole_image_in_32_multiples(self) -> None:
        det = _sent_inputs(_text_band_image(640, 427))['ocr_images']
        assert det.shape[1] % 32 == 0
        assert det.shape[2] % 32 == 0
        assert det[:, -1, :].any()
        assert det[:, :, -1].any()

    def test_unpadded_size_is_exact(self) -> None:
        # Control: 640x480 needs no padding, so the centre was exact before the fix too.
        assert _box_centre_y(_text_band_image(640, 480)) == pytest.approx(210, abs=1)
