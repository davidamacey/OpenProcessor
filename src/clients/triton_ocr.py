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
from src.utils.image_processing import decode_ocr_image
from src.utils.stage_timing import stage_timer


logger = logging.getLogger(__name__)

OCR_IMAGE_INPUTS = ('ocr_images', 'original_image')


def require_uint8_ocr_inputs(model_name: str, datatype_of: Callable[[str], str]) -> None:
    """Refuse an OCR pipeline model whose image inputs are not UINT8.

    ``datatype_of`` maps an input name to the datatype the loaded model declares. A pipeline
    still declaring FP32 is an installed model repository the upgrade did not replace; there is
    no FP32 fallback, so the operation fails closed with the repair command.

    Raises:
        ValueError: an image input is not ``UINT8``.
    """
    for name in OCR_IMAGE_INPUTS:
        datatype = datatype_of(name)
        if datatype != 'UINT8':
            msg = (
                f"model '{model_name}' declares input '{name}' as {datatype or '<none>'!r}, "
                'but OCR sends UINT8 pixels. The installed model repository is out of date: '
                'run `./openprocessor models repair`.'
            )
            raise ValueError(msg)


def _bgr_frame(img: np.ndarray) -> np.ndarray:
    """A decoded OCR image as 3-channel uint8 BGR (a grey decode gains its channels)."""
    import cv2

    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def infer_ocr(
    infer: Callable[[str, list, list], Any],
    image: bytes | np.ndarray,
    datatype_of: Callable[[str], str],
) -> dict[str, Any]:
    """
    OCR inference: PP-OCRv5 text detection and recognition via BLS pipeline.

    Args:
        image: an already decoded, EXIF-oriented BGR array (the frame the caller reports
            sizes in), or raw image bytes, which are decoded here by ``decode_ocr_image``
        datatype_of: input name -> datatype the loaded ``ocr_pipeline`` declares

    Returns:
        Dict with num_texts, texts, text_boxes, text_boxes_normalized,
        text_scores, rec_scores
    """
    import cv2

    require_uint8_ocr_inputs(TritonModelConfig.OCR_PIPELINE_MODEL, datatype_of)
    if isinstance(image, np.ndarray):
        img_array = image
    else:
        try:
            img_array = decode_ocr_image(image)
        except ValueError:
            return {
                'num_texts': 0,
                'texts': [],
                'text_boxes': np.array([]),
                'text_boxes_normalized': np.array([]),
                'text_scores': np.array([]),
                'rec_scores': np.array([]),
            }

    img_array = _bgr_frame(img_array)

    orig_h, orig_w = img_array.shape[:2]

    # PP-OCR approach: scale to fit within the limit, then round each side to 32
    ocr_limit_side = 960
    ocr_min_side = 640

    max_side = max(orig_h, orig_w)
    if max_side > ocr_limit_side:
        ratio = ocr_limit_side / max_side
    elif max_side < ocr_min_side:
        ratio = ocr_min_side / max_side
    else:
        ratio = 1.0

    # Each axis is stretched to its own multiple of 32 (PaddleOCR DetResizeForTest), never
    # zero-padded: the BLS maps boxes back as ``orig / map size``, which only holds when the
    # map covers the whole image.
    resize_h = max(32, round(orig_h * ratio / 32) * 32)
    resize_w = max(32, round(orig_w * ratio / 32) * 32)

    resized = cv2.resize(img_array, (resize_w, resize_h), interpolation=cv2.INTER_LINEAR)

    # The BLS normalizes ((x / 127.5) - 1 for the detector) and crops from the raw pixels.
    with stage_timer('triton_request', nbytes=resized.nbytes + img_array.nbytes):
        ocr_input = np.ascontiguousarray(resized.transpose(2, 0, 1))
        orig_chw = np.ascontiguousarray(img_array.transpose(2, 0, 1))
        inputs = [
            InferInput('ocr_images', list(ocr_input.shape), 'UINT8'),
            InferInput('original_image', list(orig_chw.shape), 'UINT8'),
            InferInput('orig_shape', [2], 'INT32'),
        ]
        inputs[0].set_data_from_numpy(ocr_input)
        inputs[1].set_data_from_numpy(orig_chw)
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
