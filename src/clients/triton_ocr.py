"""OCR through Triton's ``ocr_pipeline`` BLS model (PP-OCRv5 detection and recognition).

:func:`infer_ocr` is what :meth:`src.clients.triton_client.TritonClient.infer_ocr` delegates to;
``infer`` is the client's retrying ``(model_name, inputs, outputs)`` call.
"""

import logging
from collections.abc import Callable
from typing import Any

import numpy as np
from tritonclient.grpc import InferInput, InferRequestedOutput

from src.config.settings import TritonModelConfig


logger = logging.getLogger(__name__)


def infer_ocr(infer: Callable[[str, list, list], Any], image_bytes: bytes) -> dict[str, Any]:
    """
    OCR inference: PP-OCRv5 text detection and recognition via BLS pipeline.

    Args:
        image_bytes: Raw JPEG/PNG bytes

    Returns:
        Dict with num_texts, texts, text_boxes, text_boxes_normalized,
        text_scores, rec_scores
    """
    import cv2

    img_array = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
    if img_array is None:
        return {
            'num_texts': 0,
            'texts': [],
            'text_boxes': np.array([]),
            'text_boxes_normalized': np.array([]),
            'text_scores': np.array([]),
            'rec_scores': np.array([]),
        }

    # Handle different channel formats
    if len(img_array.shape) == 2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
    elif img_array.shape[2] == 4:
        alpha = img_array[:, :, 3:4] / 255.0
        rgb = img_array[:, :, :3]
        white_bg = np.ones_like(rgb) * 255
        img_array = (rgb * alpha + white_bg * (1 - alpha)).astype(np.uint8)

    orig_h, orig_w = img_array.shape[:2]

    # PP-OCR approach: scale to fit within limit, pad to 32-boundary
    ocr_limit_side = 960
    ocr_min_side = 640

    max_side = max(orig_h, orig_w)
    if max_side > ocr_limit_side:
        ratio = ocr_limit_side / max_side
    elif max_side < ocr_min_side:
        ratio = ocr_min_side / max_side
    else:
        ratio = 1.0

    resize_h = max(32, int(orig_h * ratio))
    resize_w = max(32, int(orig_w * ratio))

    resized = cv2.resize(img_array, (resize_w, resize_h), interpolation=cv2.INTER_LINEAR)

    # Pad to 32-boundary
    pad_h = (32 - resize_h % 32) % 32
    pad_w = (32 - resize_w % 32) % 32
    if pad_h > 0 or pad_w > 0:
        padded = np.zeros((resize_h + pad_h, resize_w + pad_w, 3), dtype=np.uint8)
        padded[:resize_h, :resize_w, :] = resized
        resized = padded

    # Normalize: (x / 127.5) - 1 for PP-OCR
    ocr_input = resized.astype(np.float32) / 127.5 - 1.0
    ocr_input = ocr_input.transpose(2, 0, 1)

    # Original image for cropping
    orig_normalized = img_array.astype(np.float32) / 255.0
    orig_normalized = orig_normalized.transpose(2, 0, 1)

    inputs = [
        InferInput('ocr_images', list(ocr_input.shape), 'FP32'),
        InferInput('original_image', list(orig_normalized.shape), 'FP32'),
        InferInput('orig_shape', [2], 'INT32'),
    ]

    inputs[0].set_data_from_numpy(ocr_input)
    inputs[1].set_data_from_numpy(orig_normalized)
    inputs[2].set_data_from_numpy(np.array([orig_h, orig_w], dtype=np.int32))

    outputs = [
        InferRequestedOutput('num_texts'),
        InferRequestedOutput('text_boxes'),
        InferRequestedOutput('text_boxes_normalized'),
        InferRequestedOutput('texts'),
        InferRequestedOutput('text_scores'),
        InferRequestedOutput('rec_scores'),
    ]

    response = infer(TritonModelConfig.OCR_PIPELINE_MODEL, inputs, outputs)

    num_texts_raw = response.as_numpy('num_texts')
    logger.info(f'OCR response: num_texts_raw shape={num_texts_raw.shape}, value={num_texts_raw}')
    num_texts = int(num_texts_raw[0])
    text_boxes = response.as_numpy('text_boxes')[:num_texts]
    text_boxes_norm = response.as_numpy('text_boxes_normalized')[:num_texts]
    text_scores = response.as_numpy('text_scores')[:num_texts]
    rec_scores_raw = response.as_numpy('rec_scores')[:num_texts]

    texts_raw = response.as_numpy('texts')[:num_texts]
    texts = []
    for t in texts_raw:
        if isinstance(t, bytes):
            texts.append(t.decode('utf-8', errors='ignore'))
        elif isinstance(t, np.bytes_):
            texts.append(str(t, 'utf-8', errors='ignore'))
        else:
            texts.append(str(t))

    # The BLS (models/ocr_pipeline/1/model.py) emits -1.0 for a crop
    # whose recognition inference itself failed -- distinguishable from
    # a real CTC confidence, which is always in [0, 1]. -1.0 must never
    # leave this method: every caller gets None + an explicit reason
    # instead, so a served field can never read as a fake low score.
    rec_scores: list[float | None] = []
    rec_errors: list[str | None] = []
    for s in rec_scores_raw:
        if float(s) < 0.0:
            rec_scores.append(None)
            rec_errors.append('recognition_failed')
        else:
            rec_scores.append(float(s))
            rec_errors.append(None)

    return {
        'num_texts': num_texts,
        'texts': texts,
        'text_boxes': text_boxes,
        'text_boxes_normalized': text_boxes_norm,
        'text_scores': text_scores,
        'rec_scores': rec_scores,
        'rec_errors': rec_errors,
        'image_width': orig_w,
        'image_height': orig_h,
    }
