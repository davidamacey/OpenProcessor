"""Shared building blocks of the combined analysis pipeline (single and batch)."""

from __future__ import annotations

import io
import logging
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Any

from src.config.settings import TritonModelConfig
from src.schemas.analyze import (
    AnalyzeResponse,
    BatchAnalyzeResult,
    DetectionResult,
    FaceResult,
    OcrResult,
)
from src.schemas.detection import ImageMetadata
from src.services.inference import InferenceService
from src.services.ocr_service import get_ocr_service
from src.utils.affine import format_detections_from_triton


if TYPE_CHECKING:
    import numpy as np


logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_inference_service() -> InferenceService:
    """Get or create InferenceService singleton (cached)."""
    return InferenceService()


def ocr_failure(ocr_result: dict[str, Any] | None) -> str | None:
    """The message of an OCR result that failed (``None`` when it ran, even
    with no text, or was not requested): a failed OCR must not look like an
    image with no text."""
    if ocr_result and ocr_result.get('status') == 'error':
        return str(ocr_result.get('error') or 'OCR failed')
    return None


def detections_from_yolo(
    yolo_clip_result: dict[str, Any], confidence: float
) -> list[DetectionResult]:
    """The YOLO detections at or above ``confidence``, in the response shape."""
    detections: list[DetectionResult] = []
    if yolo_clip_result.get('num_dets', 0) > 0:
        formatted = format_detections_from_triton(
            yolo_clip_result,
            input_size=yolo_clip_result['input_size'],
            model_name=TritonModelConfig.YOLO_MODEL,
        )
        detections.extend(
            DetectionResult(
                box=[det['x1'], det['y1'], det['x2'], det['y2']],
                confidence=det['confidence'],
                class_id=det['class'],
                class_name=det.get('class_name'),
            )
            for det in formatted
            if det['confidence'] >= confidence
        )
    return detections


def image_size(yolo_clip_result: dict[str, Any]) -> tuple[int, int]:
    """``(width, height)`` of the original image, ``(0, 0)`` when unknown."""
    orig_shape = yolo_clip_result.get('orig_shape', (0, 0))
    height = orig_shape[0] if orig_shape else 0
    width = orig_shape[1] if orig_shape else 0
    return width, height


def faces_from_result(
    faces_result: dict[str, Any], include_face_embeddings: bool
) -> list[FaceResult]:
    """The detected faces in the response shape, with embeddings on request."""
    faces = []
    for i, face in enumerate(faces_result.get('faces', [])):
        face_emb = None
        if include_face_embeddings and i < len(faces_result.get('embeddings', [])):
            face_emb = faces_result['embeddings'][i]
            if hasattr(face_emb, 'tolist'):
                face_emb = face_emb.tolist()

        faces.append(
            FaceResult(
                box=face['box'],
                landmarks=face['landmarks'],
                score=face['score'],
                quality=face.get('quality'),
                embedding=face_emb,
            )
        )
    return faces


def ocr_from_result(ocr_result: dict[str, Any] | None) -> OcrResult | None:
    """The OCR text in the response shape; ``None`` unless OCR succeeded."""
    if ocr_result and ocr_result.get('status') == 'success':
        return OcrResult(
            texts=ocr_result.get('texts', []),
            boxes=ocr_result.get('boxes', []),
            boxes_normalized=ocr_result.get('boxes_normalized', []),
            det_scores=ocr_result.get('det_scores', []),
            rec_scores=ocr_result.get('rec_scores', []),
            rec_errors=ocr_result.get('rec_errors', []),
            full_text=' '.join(ocr_result.get('texts', [])),
            num_texts=ocr_result.get('num_texts', 0),
        )
    return None


def crop_batch_tensor(
    image_bytes: bytes, detections: list[DetectionResult], img_w: int, img_h: int
) -> np.ndarray:
    """Crop and preprocess every detection into one ``[N, 3, 256, 256]`` batch."""
    import numpy as np
    from PIL import Image

    from src.utils.cpu_preprocess import center_crop_cpu

    pil_img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    np_img = np.array(pil_img)

    preprocessed = []
    for det in detections:
        x1, y1, x2, y2 = det.box
        px1, py1 = int(x1 * img_w), int(y1 * img_h)
        px2, py2 = int(x2 * img_w), int(y2 * img_h)
        if px2 > px1 and py2 > py1:
            preprocessed.append(center_crop_cpu(np_img[py1:py2, px1:px2], target_size=256))
        else:
            # Placeholder for invalid boxes (zero tensor)
            preprocessed.append(np.zeros((3, 256, 256), dtype=np.float32))
    return np.stack(preprocessed)


def _run_yolo_and_clip(image_bytes: bytes, timing: dict[str, float]) -> dict[str, Any]:
    t0 = time.perf_counter()
    from src.clients.triton_client import get_sync_triton_client
    from src.config import get_settings

    settings = get_settings()
    client = get_sync_triton_client(settings.triton_url)
    result = client.infer_yolo_clip_cpu(image_bytes)
    timing['yolo_clip_ms'] = (time.perf_counter() - t0) * 1000
    return result


def _run_faces(
    service: InferenceService, image_bytes: bytes, face_confidence: float, timing: dict[str, float]
) -> dict[str, Any]:
    t0 = time.perf_counter()
    result = service.detect_faces(image_bytes, confidence=face_confidence)
    timing['faces_ms'] = (time.perf_counter() - t0) * 1000
    return result


def _run_ocr(
    ocr_service: Any, image_bytes: bytes, enable_ocr: bool, timing: dict[str, float]
) -> dict[str, Any] | None:
    if not enable_ocr:
        return None
    t0 = time.perf_counter()
    result = ocr_service.extract_text(image_bytes, filter_by_score=True)
    timing['ocr_ms'] = (time.perf_counter() - t0) * 1000
    typed: dict[str, Any] | None = result
    return typed


def _run_crop_embeddings(
    image_bytes: bytes,
    detections: list[DetectionResult],
    img_w: int,
    img_h: int,
    timing: dict[str, float],
) -> np.ndarray | None:
    """Batch-encode all detection crops via single Triton call."""
    if not detections or img_w <= 0 or img_h <= 0:
        return None
    t0 = time.perf_counter()
    from src.clients.triton_client import get_sync_triton_client
    from src.config import get_settings

    batch_tensor = crop_batch_tensor(image_bytes, detections, img_w, img_h)

    # Single batched Triton inference call
    settings = get_settings()
    client = get_sync_triton_client(settings.triton_url)
    embeddings = client.infer_mobileclip_batch(batch_tensor)
    timing['crop_embeddings_ms'] = (time.perf_counter() - t0) * 1000
    return embeddings  # [N, 512]


def _attach_crop_embeddings(
    detections: list[DetectionResult], crop_future: Future[np.ndarray | None] | None
) -> None:
    """Set each detection's embedding from the finished crop batch, if there was one."""
    if crop_future is None:
        return
    try:
        crop_embeddings = crop_future.result()
        if crop_embeddings is not None:
            for i, det in enumerate(detections):
                det.embedding = crop_embeddings[i].tolist()
    except Exception:
        logger.warning('Failed to generate batched crop embeddings')


def _global_embedding(
    yolo_clip_result: dict[str, Any], include_embedding: bool
) -> tuple[list[float] | None, float | None]:
    """The whole-image embedding and its L2 norm, when requested and present."""
    import numpy as np

    if include_embedding:
        emb = yolo_clip_result.get('image_embedding')
        if emb is not None:
            emb_array = np.array(emb)
            return emb_array.tolist(), float(np.linalg.norm(emb_array))
    return None, None


def analyze_single_image(
    image_bytes: bytes,
    *,
    include_embedding: bool,
    include_face_embeddings: bool,
    enable_ocr: bool,
    confidence: float,
    face_confidence: float,
) -> AnalyzeResponse:
    """YOLO + CLIP, faces and OCR on one image, overlapped on a thread pool."""
    timing: dict[str, float] = {}
    service = get_inference_service()
    ocr_service = get_ocr_service()

    # Phase 1: Submit YOLO+CLIP, faces, OCR in parallel
    # Phase 2: As soon as YOLO returns boxes, submit batched crop embeddings
    #          (overlaps with faces/OCR still running)
    with ThreadPoolExecutor(max_workers=4) as executor:
        yolo_clip_future = executor.submit(_run_yolo_and_clip, image_bytes, timing)
        faces_future = executor.submit(_run_faces, service, image_bytes, face_confidence, timing)
        ocr_future = executor.submit(_run_ocr, ocr_service, image_bytes, enable_ocr, timing)

        # Wait for YOLO+CLIP first (needed for detection boxes)
        yolo_clip_result = yolo_clip_future.result()

        detections = detections_from_yolo(yolo_clip_result, confidence)
        img_width, img_height = image_size(yolo_clip_result)

        # Submit batched crop embeddings (runs in parallel with faces/OCR)
        crop_future = None
        if include_embedding and detections:
            crop_future = executor.submit(
                _run_crop_embeddings, image_bytes, detections, img_width, img_height, timing
            )

        # Wait for faces and OCR (may already be done)
        faces_result = faces_future.result()
        ocr_result = ocr_future.result()

        _attach_crop_embeddings(detections, crop_future)

    global_embedding, embedding_norm = _global_embedding(yolo_clip_result, include_embedding)

    faces = []
    if faces_result.get('status') != 'error' and faces_result.get('num_faces', 0) > 0:
        faces = faces_from_result(faces_result, include_face_embeddings)

    return AnalyzeResponse(
        status='success',
        image=ImageMetadata(width=img_width, height=img_height) if img_width > 0 else None,
        detections=detections,
        num_detections=len(detections),
        faces=faces,
        num_faces=len(faces),
        global_embedding=global_embedding,
        embedding_norm=embedding_norm,
        ocr=ocr_from_result(ocr_result),
        ocr_error=ocr_failure(ocr_result),
        timing=timing if timing else None,
    )


@dataclass
class BatchTotals:
    """Running summary counts of a batch, added to as each image's pipeline finishes."""

    detections: int = 0
    faces: int = 0
    images_with_text: int = 0


def _embed_detection_crops(
    client: Any,
    image_bytes: bytes,
    detections: list[DetectionResult],
    width: int,
    height: int,
    filename: str,
) -> None:
    """Set each detection's embedding from one batched crop call; a failure only logs."""
    if not (detections and width > 0 and height > 0):
        return
    try:
        batch_tensor = crop_batch_tensor(image_bytes, detections, width, height)
        crop_embeddings = client.infer_mobileclip_batch(batch_tensor)
        for det_i, det in enumerate(detections):
            det.embedding = crop_embeddings[det_i].tolist()
    except Exception:
        logger.warning(f'Failed to generate crop embeddings for {filename}')


def _add_full_results(
    result: BatchAnalyzeResult,
    *,
    client: Any,
    image_bytes: bytes,
    filename: str,
    yolo_clip_result: dict[str, Any],
    faces_result: dict[str, Any],
    ocr_result: dict[str, Any] | None,
    include_embedding: bool,
    include_face_embeddings: bool,
    confidence: float,
) -> None:
    """Fill the detection, face, embedding and OCR lists of a batch image result."""
    width, height = image_size(yolo_clip_result)
    detections = detections_from_yolo(yolo_clip_result, confidence)
    if include_embedding:
        _embed_detection_crops(client, image_bytes, detections, width, height, filename)
    result.detections = detections

    result.faces = (
        faces_from_result(faces_result, include_face_embeddings)
        if faces_result.get('num_faces', 0) > 0
        else []
    )

    result.global_embedding = _global_embedding(yolo_clip_result, include_embedding)[0]

    result.ocr = ocr_from_result(ocr_result)
    result.ocr_error = ocr_failure(ocr_result)


def analyze_batch_item(
    image_bytes: bytes,
    *,
    filename: str,
    idx: int,
    service: InferenceService,
    ocr_service: Any,
    totals: BatchTotals,
    include_embedding: bool,
    include_face_embeddings: bool,
    enable_ocr: bool,
    include_full_results: bool,
    confidence: float,
) -> BatchAnalyzeResult:
    """Run the pipeline on one image of a batch, adding its counts to ``totals``."""
    from src.clients.triton_client import get_sync_triton_client
    from src.config import get_settings

    settings = get_settings()
    client = get_sync_triton_client(settings.triton_url)

    # Run YOLO + CLIP with CPU preprocessing
    yolo_clip_result = client.infer_yolo_clip_cpu(image_bytes)

    # Run face detection
    faces_result = service.detect_faces(image_bytes, confidence=0.5)

    # Run OCR if enabled
    ocr_result = None
    if enable_ocr:
        ocr_result = ocr_service.extract_text(image_bytes, filter_by_score=True)

    num_dets = yolo_clip_result.get('num_dets', 0)
    num_faces = faces_result.get('num_faces', 0)
    has_text = bool(ocr_result and ocr_result.get('num_texts', 0) > 0)

    totals.detections += num_dets
    totals.faces += num_faces
    if has_text:
        totals.images_with_text += 1

    width, height = image_size(yolo_clip_result)
    result = BatchAnalyzeResult(
        filename=filename,
        image_index=idx,
        status='success',
        image=ImageMetadata(width=width, height=height),
        num_detections=num_dets,
        num_faces=num_faces,
        has_text=has_text,
    )
    if include_full_results:
        _add_full_results(
            result,
            client=client,
            image_bytes=image_bytes,
            filename=filename,
            yolo_clip_result=yolo_clip_result,
            faces_result=faces_result,
            ocr_result=ocr_result,
            include_embedding=include_embedding,
            include_face_embeddings=include_face_embeddings,
            confidence=confidence,
        )
    return result
