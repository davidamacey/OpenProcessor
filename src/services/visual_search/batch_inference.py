"""Per-image inference for a batch: YOLO + MobileCLIP, faces and OCR, run in
parallel threads against the sync Triton clients."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any


logger = logging.getLogger(__name__)


def infer_one(
    img_bytes: bytes,
    *,
    client: Any,
    face_client: Any,
    ocr_service: Any,
    enable_detection: bool,
    enable_clip: bool,
    enable_faces: bool,
    enable_ocr: bool,
) -> dict:
    """Run selective inference based on enabled pipelines."""
    try:
        result: dict = {
            'num_dets': 0,
            'num_faces': 0,
            'num_texts': 0,
            'global_embedding': [],
        }

        # Run YOLO + MobileCLIP if detection or CLIP is enabled
        if enable_detection or enable_clip:
            yolo_result = client.infer_yolo_clip_cpu(img_bytes)
            if enable_detection:
                result.update(
                    {
                        'num_dets': yolo_result.get('num_dets', 0),
                        'boxes': yolo_result.get('boxes', []),
                        'scores': yolo_result.get('scores', []),
                        'classes': yolo_result.get('classes', []),
                        'normalized_boxes': yolo_result.get('normalized_boxes', []),
                        'box_embeddings': yolo_result.get('box_embeddings', []),
                    }
                )
            if enable_clip:
                result['global_embedding'] = yolo_result.get('image_embedding', [])
            # Always get orig_shape for image dimensions
            result['orig_shape'] = yolo_result.get('orig_shape')

        # Run face detection + embedding extraction if faces is enabled
        if enable_faces:
            face_result = face_client.recognize(img_bytes, confidence=0.5)
            result.update(
                {
                    'num_faces': face_result.get('num_faces', 0),
                    'face_boxes': face_result.get('face_boxes', []),
                    'face_landmarks': face_result.get('face_landmarks', []),
                    'face_scores': face_result.get('face_scores', []),
                    'face_embeddings': face_result.get('face_embeddings', []),
                    'face_quality': face_result.get('face_quality', []),
                }
            )

        # Run OCR text extraction if enabled (DF2: this used to be
        # skipped entirely, so num_texts stayed 0 and the OCR
        # indexing step never had anything to index).
        if enable_ocr:
            ocr_result = ocr_service.extract_text(img_bytes, filter_by_score=True)
            if ocr_result.get('status') == 'success':
                result.update(
                    {
                        'num_texts': ocr_result.get('num_texts', 0),
                        'texts': ocr_result.get('texts', []),
                        'text_boxes': ocr_result.get('boxes', []),
                        'text_boxes_normalized': ocr_result.get('boxes_normalized', []),
                        'text_det_scores': ocr_result.get('det_scores', []),
                        'text_rec_scores': ocr_result.get('rec_scores', []),
                    }
                )

        return result
    except Exception as e:
        logger.error(f'Unified pipeline inference failed: {e}', exc_info=True)
        return {'error': str(e), 'num_dets': 0, 'num_faces': 0, 'num_texts': 0}


def infer_batch(images_bytes: list[bytes], **pipelines: Any) -> list[dict]:
    """Run the unified pipeline on a batch of images in parallel."""
    # Limit concurrent Triton inference to prevent gRPC "too_many_pings" errors
    # Single shared gRPC connection can handle ~8 concurrent streams reliably
    inference_workers = min(8, len(images_bytes))
    with ThreadPoolExecutor(max_workers=inference_workers) as executor:
        return list(executor.map(lambda img: infer_one(img, **pipelines), images_bytes))
