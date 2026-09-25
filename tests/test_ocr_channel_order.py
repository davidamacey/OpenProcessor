"""
DF4: the detection worker's OCR preprocessing (`cascade_detect.py`) built
the `original_image` BLS input tensor straight from a PIL (RGB) decode with
no channel swap, while the `ocr_pipeline` BLS (`models*/ocr_pipeline/1/model.py`)
reads that tensor as HWC and perspective-warps recognition crops from it with
no channel correction — its recognition preprocessing (`_resize_norm_img*`)
documents its input as BGR and never converts. Every OTHER OCR caller
(`/ocr/*`, `/analyze`, generic ingest — all via `TritonClient.infer_ocr`)
decodes with `cv2.imdecode` (BGR) and sends that array unchanged, so they
were already correct; only the worker was channel-swapped.

This test drives `PaddleOcrTextRecognizer._preprocess` directly (pure CPU,
no Triton call) and asserts the `original_image` tensor it returns is BGR
— i.e. matches what `cv2.imdecode` would have produced for the same pixel
data, not the raw PIL RGB array.
"""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from src.services.detection.cascade_detect import PaddleOcrTextRecognizer


@pytest.fixture
def recognizer() -> PaddleOcrTextRecognizer:
    # triton_pool/profile are unused by _preprocess (pure CPU tensor prep).
    return PaddleOcrTextRecognizer.__new__(PaddleOcrTextRecognizer)


def test_original_image_tensor_is_bgr(recognizer: PaddleOcrTextRecognizer) -> None:
    # A distinctly-colored solid image: R=10, G=20, B=200 (clearly blue-ish).
    r, g, b = 10, 20, 200
    img = Image.new('RGB', (64, 48), color=(r, g, b))

    _ocr_in, orig_in, orig_shape = recognizer._preprocess(img)

    assert orig_shape.tolist() == [48, 64]  # [H, W]
    assert orig_in.shape == (3, 48, 64)  # C, H, W

    # De-normalize back to uint8 pixel values per channel.
    channel_means = (orig_in.reshape(3, -1).mean(axis=1) * 255.0).round()

    # BGR order: channel 0 = Blue (200), channel 1 = Green (20), channel 2 = Red (10).
    assert channel_means[0] == pytest.approx(b, abs=1)
    assert channel_means[1] == pytest.approx(g, abs=1)
    assert channel_means[2] == pytest.approx(r, abs=1)

    # And NOT RGB order (the pre-fix bug): channel 0 must not be Red.
    assert channel_means[0] != pytest.approx(r, abs=1)


def test_original_image_tensor_matches_cv2_imdecode_convention(
    recognizer: PaddleOcrTextRecognizer,
) -> None:
    """Cross-check against the OTHER OCR callers' convention: decode the
    same JPEG bytes with cv2.imdecode (BGR, what /ocr/*, /analyze, and
    generic ingest all send) and compare channel means to the worker's
    tensor — they must agree once both are read as HWC BGR uint8.
    """
    import cv2

    color = (30, 90, 210)  # PIL RGB fill color
    pil_img = Image.new('RGB', (32, 32), color=color)
    buf = io.BytesIO()
    pil_img.save(buf, format='JPEG', quality=95)
    jpeg_bytes = buf.getvalue()

    # What /ocr/*, /analyze, and generic ingest actually send (BGR, unchanged).
    cv2_bgr = cv2.imdecode(np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR)
    cv2_channel_means = cv2_bgr.reshape(-1, 3).mean(axis=0)  # already BGR order

    # Re-decode the same bytes with PIL (as the worker does) and preprocess.
    reloaded = Image.open(io.BytesIO(jpeg_bytes)).convert('RGB')
    _ocr_in, orig_in, _orig_shape = recognizer._preprocess(reloaded)
    worker_channel_means = orig_in.reshape(3, -1).mean(axis=1) * 255.0

    assert np.allclose(worker_channel_means, cv2_channel_means, atol=2.0)
