"""
Combined Analysis Router - All Models in One Request.

Runs YOLO detection, face recognition, CLIP embedding, and OCR
on a single image in one unified request.

Endpoints:
- POST /analyze - All models on single image
- POST /analyze/batch - Batch combined analysis
"""

import logging
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import Annotated

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import ORJSONResponse
from pydantic import BaseModel, Field

from src.schemas.detection import ImageMetadata
from src.services.inference import InferenceService
from src.services.ocr_service import get_ocr_service
from src.utils.affine import format_detections_from_triton


logger = logging.getLogger(__name__)

router = APIRouter(
    prefix='/analyze',
    tags=['Combined Analysis'],
    default_response_class=ORJSONResponse,
)


# =============================================================================
# Response Models
# =============================================================================


class DetectionResult(BaseModel):
    """Single detection from YOLO."""

    box: list[float] = Field(..., description='Bounding box [x1, y1, x2, y2] normalized [0,1]')
    confidence: float = Field(..., description='Detection confidence score')
    class_id: int = Field(..., description='COCO class ID (0-79)')
    class_name: str | None = Field(default=None, description='Human-readable class name')
    embedding: list[float] | None = Field(
        default=None, description='512-dim MobileCLIP crop embedding (if include_embedding=True)'
    )


class FaceResult(BaseModel):
    """Single face detection with embedding."""

    box: list[float] = Field(..., description='Face bounding box [x1, y1, x2, y2] normalized [0,1]')
    landmarks: list[float] = Field(
        ..., description='5-point facial landmarks [lx1,ly1,...,lx5,ly5] normalized [0,1]'
    )
    score: float = Field(..., description='Detection confidence score')
    quality: float | None = Field(default=None, description='Face quality score')
    embedding: list[float] | None = Field(
        default=None, description='512-dim ArcFace embedding (if include_embeddings=True)'
    )


class OcrResult(BaseModel):
    """OCR extraction result."""

    texts: list[str] = Field(default_factory=list, description='Detected text strings')
    boxes: list[list[float]] = Field(
        default_factory=list, description='Quad boxes [x1,y1,x2,y2,x3,y3,x4,y4]'
    )
    boxes_normalized: list[list[float]] = Field(
        default_factory=list, description='Axis-aligned boxes [x1,y1,x2,y2] normalized'
    )
    det_scores: list[float] = Field(default_factory=list, description='Detection confidence scores')
    rec_scores: list[float] = Field(
        default_factory=list, description='Recognition confidence scores'
    )
    full_text: str = Field(default='', description='All text concatenated')
    num_texts: int = Field(default=0, description='Number of text regions')


class AnalyzeResponse(BaseModel):
    """Response for combined analysis endpoint."""

    status: str = Field(default='success', description="'success' or 'error'")
    image: ImageMetadata | None = Field(default=None, description='Original image dimensions')

    # YOLO object detection
    detections: list[DetectionResult] = Field(
        default_factory=list, description='YOLO object detections'
    )
    num_detections: int = Field(default=0, description='Number of objects detected')

    # Face detection/recognition
    faces: list[FaceResult] = Field(default_factory=list, description='Detected faces')
    num_faces: int = Field(default=0, description='Number of faces detected')

    # CLIP embedding
    global_embedding: list[float] | None = Field(
        default=None, description='512-dim MobileCLIP embedding (if include_embedding=True)'
    )
    embedding_norm: float | None = Field(default=None, description='L2 norm of embedding')

    # OCR
    ocr: OcrResult | None = Field(default=None, description='OCR results (if enable_ocr=True)')

    # Timing
    total_time_ms: float | None = Field(default=None, description='Total processing time in ms')

    # Per-component timing
    timing: dict[str, float] | None = Field(
        default=None, description='Per-component timing breakdown in ms'
    )


class BatchAnalyzeResult(BaseModel):
    """Single image result in batch analysis."""

    filename: str = Field(..., description='Original filename')
    image_index: int = Field(..., description='Index in batch')
    status: str = Field(default='success', description="'success' or 'error'")
    error: str | None = Field(default=None, description='Error message if failed')

    image: ImageMetadata | None = Field(default=None, description='Image dimensions')
    num_detections: int = Field(default=0, description='Number of objects detected')
    num_faces: int = Field(default=0, description='Number of faces detected')
    has_text: bool = Field(default=False, description='Whether OCR found text')

    # Full results only if requested
    detections: list[DetectionResult] | None = Field(default=None)
    faces: list[FaceResult] | None = Field(default=None)
    global_embedding: list[float] | None = Field(default=None)
    ocr: OcrResult | None = Field(default=None)


class BatchAnalyzeResponse(BaseModel):
    """Response for batch combined analysis."""

    status: str = Field(default='success', description='Overall status')
    total_images: int = Field(..., description='Total images in request')
    processed_images: int = Field(..., description='Successfully processed images')
    failed_images: int = Field(..., description='Failed images count')

    results: list[BatchAnalyzeResult] = Field(default_factory=list, description='Per-image results')
    failures: list[dict] | None = Field(default=None, description='Details of failed images')

    # Aggregated stats
    total_detections: int = Field(default=0, description='Sum of detections across all images')
    total_faces: int = Field(default=0, description='Sum of faces across all images')
    images_with_text: int = Field(default=0, description='Number of images with OCR text')

    total_time_ms: float | None = Field(
        default=None, description='Total batch processing time in ms'
    )


# =============================================================================
# Service Instance
# =============================================================================


@lru_cache(maxsize=1)
def get_inference_service() -> InferenceService:
    """Get or create InferenceService singleton (cached)."""
    return InferenceService()


# =============================================================================
# Endpoints
# =============================================================================


@router.post('', response_model=AnalyzeResponse)
def analyze_image(
    image: Annotated[UploadFile, File(description='Image file (JPEG/PNG)')],
    include_embedding: Annotated[
        bool, Query(description='Include 512-dim CLIP embedding in response')
    ] = False,
    include_face_embeddings: Annotated[
        bool, Query(description='Include face embeddings in response')
    ] = False,
    enable_ocr: Annotated[bool, Query(description='Run OCR text extraction')] = True,
    confidence: Annotated[
        float, Query(ge=0.0, le=1.0, description='Minimum YOLO detection confidence')
    ] = 0.25,
    face_confidence: Annotated[
        float, Query(ge=0.1, le=0.99, description='Minimum face detection confidence')
    ] = 0.5,
):
    """
    Run all models on a single image: YOLO + faces + CLIP + OCR.

    Unified pipeline for comprehensive image analysis:
    1. YOLO object detection (TensorRT End2End)
    2. YOLO11-face detection + ArcFace embedding extraction
    3. MobileCLIP global image embedding
    4. PP-OCRv5 text detection and recognition

    All models run in parallel where possible for optimal performance.

    Args:
        image: Image file (JPEG/PNG)
        include_embedding: Include 512-dim CLIP embedding in response
        include_face_embeddings: Include ArcFace embeddings per face
        enable_ocr: Run OCR text extraction
        confidence: Minimum YOLO detection confidence (default: 0.25)
        face_confidence: Minimum face detection confidence (default: 0.5)

    Returns:
        Combined analysis results with detections, faces, embedding, and OCR.
    """
    import time

    filename = image.filename or 'uploaded_image'
    timing = {}

    try:
        image_bytes = image.file.read()

        if not image_bytes:
            raise HTTPException(status_code=400, detail='Empty image file')

        service = get_inference_service()
        ocr_service = get_ocr_service()

        # Run all pipelines in parallel using ThreadPoolExecutor
        # Architecture: YOLO+CLIP, faces, OCR start in parallel.
        # As soon as YOLO returns detections, crop embedding tasks are
        # submitted to the same pool — overlapping with faces/OCR.
        import io

        import numpy as np
        from PIL import Image

        def run_yolo_and_clip():
            t0 = time.perf_counter()
            from src.clients.triton_client import get_triton_client
            from src.config import get_settings

            settings = get_settings()
            client = get_triton_client(settings.triton_url)
            result = client.infer_yolo_clip_cpu(image_bytes)
            timing['yolo_clip_ms'] = (time.perf_counter() - t0) * 1000
            return result

        def run_faces():
            t0 = time.perf_counter()
            result = service.detect_faces(image_bytes, confidence=face_confidence)
            timing['faces_ms'] = (time.perf_counter() - t0) * 1000
            return result

        def run_ocr():
            if not enable_ocr:
                return None
            t0 = time.perf_counter()
            result = ocr_service.extract_text(image_bytes, filter_by_score=True)
            timing['ocr_ms'] = (time.perf_counter() - t0) * 1000
            return result

        def run_crop_embeddings(
            det_list: list[DetectionResult],
            img_w: int,
            img_h: int,
        ) -> np.ndarray | None:
            """Batch-encode all detection crops via single Triton call."""
            if not det_list or img_w <= 0 or img_h <= 0:
                return None
            t0 = time.perf_counter()
            from src.clients.triton_client import get_triton_client
            from src.config import get_settings
            from src.services.cpu_preprocess import center_crop_cpu

            pil_img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            np_img = np.array(pil_img)

            # CPU crop + preprocess all detections into a single [N, 3, 256, 256] batch
            preprocessed = []
            for det in det_list:
                x1, y1, x2, y2 = det.box
                px1, py1 = int(x1 * img_w), int(y1 * img_h)
                px2, py2 = int(x2 * img_w), int(y2 * img_h)
                if px2 > px1 and py2 > py1:
                    crop_rgb = np_img[py1:py2, px1:px2]
                    preprocessed.append(center_crop_cpu(crop_rgb, target_size=256))
                else:
                    # Placeholder for invalid boxes (zero tensor)
                    preprocessed.append(np.zeros((3, 256, 256), dtype=np.float32))

            batch_tensor = np.stack(preprocessed)  # [N, 3, 256, 256]

            # Single batched Triton inference call
            settings = get_settings()
            client = get_triton_client(settings.triton_url)
            embeddings = client.infer_mobileclip_batch(batch_tensor)
            timing['crop_embeddings_ms'] = (time.perf_counter() - t0) * 1000
            return embeddings  # [N, 512]

        # Phase 1: Submit YOLO+CLIP, faces, OCR in parallel
        # Phase 2: As soon as YOLO returns boxes, submit batched crop embeddings
        #          (overlaps with faces/OCR still running)
        with ThreadPoolExecutor(max_workers=4) as executor:
            yolo_clip_future = executor.submit(run_yolo_and_clip)
            faces_future = executor.submit(run_faces)
            ocr_future = executor.submit(run_ocr)

            # Wait for YOLO+CLIP first (needed for detection boxes)
            yolo_clip_result = yolo_clip_future.result()

            # Parse detections immediately
            detections: list[DetectionResult] = []
            if yolo_clip_result.get('num_dets', 0) > 0:
                formatted = format_detections_from_triton(yolo_clip_result, input_size=640)
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

            orig_shape = yolo_clip_result.get('orig_shape', (0, 0))
            img_height = orig_shape[0] if orig_shape else 0
            img_width = orig_shape[1] if orig_shape else 0

            # Submit batched crop embeddings (runs in parallel with faces/OCR)
            crop_future = None
            if include_embedding and detections:
                crop_future = executor.submit(
                    run_crop_embeddings, detections, img_width, img_height
                )

            # Wait for faces and OCR (may already be done)
            faces_result = faces_future.result()
            ocr_result = ocr_future.result()

            # Collect batched crop embeddings
            if crop_future is not None:
                try:
                    crop_embeddings = crop_future.result()
                    if crop_embeddings is not None:
                        for i, det in enumerate(detections):
                            det.embedding = crop_embeddings[i].tolist()
                except Exception:
                    logger.warning('Failed to generate batched crop embeddings')

        # Parse global embedding
        global_embedding = None
        embedding_norm = None
        if include_embedding:
            emb = yolo_clip_result.get('image_embedding')
            if emb is not None:
                emb_array = np.array(emb)
                global_embedding = emb_array.tolist()
                embedding_norm = float(np.linalg.norm(emb_array))

        # Parse face results
        faces = []
        if faces_result.get('status') != 'error' and faces_result.get('num_faces', 0) > 0:
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

        # Parse OCR results
        ocr = None
        if ocr_result and ocr_result.get('status') == 'success':
            ocr = OcrResult(
                texts=ocr_result.get('texts', []),
                boxes=ocr_result.get('boxes', []),
                boxes_normalized=ocr_result.get('boxes_normalized', []),
                det_scores=ocr_result.get('det_scores', []),
                rec_scores=ocr_result.get('rec_scores', []),
                full_text=' '.join(ocr_result.get('texts', [])),
                num_texts=ocr_result.get('num_texts', 0),
            )

        return AnalyzeResponse(
            status='success',
            image=ImageMetadata(width=img_width, height=img_height) if img_width > 0 else None,
            detections=detections,
            num_detections=len(detections),
            faces=faces,
            num_faces=len(faces),
            global_embedding=global_embedding,
            embedding_norm=embedding_norm,
            ocr=ocr,
            timing=timing if timing else None,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'Analysis failed for {filename}: {e}')
        raise HTTPException(status_code=500, detail=f'Analysis failed: {e!s}') from e


@router.post('/batch', response_model=BatchAnalyzeResponse)
def analyze_batch(
    images: Annotated[list[UploadFile], File(description='Image files (JPEG/PNG), max 32')],
    include_embedding: Annotated[
        bool, Query(description='Include CLIP embeddings in response')
    ] = False,
    include_face_embeddings: Annotated[
        bool, Query(description='Include face embeddings in response')
    ] = False,
    enable_ocr: Annotated[bool, Query(description='Run OCR text extraction')] = False,
    include_full_results: Annotated[
        bool, Query(description='Include full detection/face lists (can be large)')
    ] = False,
    confidence: Annotated[
        float, Query(ge=0.0, le=1.0, description='Minimum YOLO detection confidence')
    ] = 0.25,
):
    """
    Run all models on multiple images in batch.

    Processes up to 32 images in a single request. By default, returns
    summary statistics. Set include_full_results=True for complete data.

    For large batches (50K+ images), use the ingest endpoint instead.

    Args:
        images: List of image files (max 32)
        include_embedding: Include CLIP embeddings
        include_face_embeddings: Include face embeddings
        enable_ocr: Run OCR (disabled by default for performance)
        include_full_results: Include full detection/face lists
        confidence: Minimum YOLO detection confidence

    Returns:
        Batch analysis results with per-image summaries.
    """
    MAX_BATCH = 32

    if len(images) > MAX_BATCH:
        raise HTTPException(
            status_code=400,
            detail=f'Batch size {len(images)} exceeds maximum {MAX_BATCH}',
        )

    if len(images) == 0:
        return BatchAnalyzeResponse(
            status='success',
            total_images=0,
            processed_images=0,
            failed_images=0,
            results=[],
        )

    results = []
    failures = []
    total_dets = 0
    total_faces = 0
    images_with_text = 0

    service = get_inference_service()
    ocr_service = get_ocr_service()

    for idx, img in enumerate(images):
        filename = img.filename or f'image_{idx}'

        try:
            image_bytes = img.file.read()

            if not image_bytes:
                failures.append(
                    {
                        'filename': filename,
                        'index': idx,
                        'error': 'Empty image file',
                    }
                )
                continue

            # Run unified pipeline
            import numpy as np

            from src.clients.triton_client import get_triton_client
            from src.config import get_settings

            settings = get_settings()
            client = get_triton_client(settings.triton_url)

            # Run YOLO + CLIP with CPU preprocessing
            yolo_clip_result = client.infer_yolo_clip_cpu(image_bytes)

            # Run face detection
            faces_result = service.detect_faces(image_bytes, confidence=0.5)

            # Run OCR if enabled
            ocr_result = None
            if enable_ocr:
                ocr_result = ocr_service.extract_text(image_bytes, filter_by_score=True)

            # Parse results
            num_dets = yolo_clip_result.get('num_dets', 0)
            num_faces = faces_result.get('num_faces', 0)
            has_text = bool(ocr_result and ocr_result.get('num_texts', 0) > 0)

            total_dets += num_dets
            total_faces += num_faces
            if has_text:
                images_with_text += 1

            batch_orig_shape = yolo_clip_result.get('orig_shape', (0, 0))
            result = BatchAnalyzeResult(
                filename=filename,
                image_index=idx,
                status='success',
                image=ImageMetadata(
                    width=batch_orig_shape[1] if batch_orig_shape else 0,
                    height=batch_orig_shape[0] if batch_orig_shape else 0,
                ),
                num_detections=num_dets,
                num_faces=num_faces,
                has_text=has_text,
            )

            # Include full results if requested
            if include_full_results:
                # Detections
                detections: list[DetectionResult] = []
                if num_dets > 0:
                    formatted = format_detections_from_triton(yolo_clip_result, input_size=640)
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

                # Generate CLIP crop embeddings via batched Triton call
                batch_orig_w = batch_orig_shape[1] if batch_orig_shape else 0
                batch_orig_h = batch_orig_shape[0] if batch_orig_shape else 0
                if include_embedding and detections and batch_orig_w > 0 and batch_orig_h > 0:
                    try:
                        import io

                        from PIL import Image

                        from src.services.cpu_preprocess import center_crop_cpu

                        np_img = np.array(Image.open(io.BytesIO(image_bytes)).convert('RGB'))
                        preprocessed = []
                        for det in detections:
                            x1, y1, x2, y2 = det.box
                            px1, py1 = int(x1 * batch_orig_w), int(y1 * batch_orig_h)
                            px2, py2 = int(x2 * batch_orig_w), int(y2 * batch_orig_h)
                            if px2 > px1 and py2 > py1:
                                preprocessed.append(
                                    center_crop_cpu(np_img[py1:py2, px1:px2], target_size=256)
                                )
                            else:
                                preprocessed.append(np.zeros((3, 256, 256), dtype=np.float32))
                        batch_tensor = np.stack(preprocessed)
                        crop_embeddings = client.infer_mobileclip_batch(batch_tensor)
                        for det_i, det in enumerate(detections):
                            det.embedding = crop_embeddings[det_i].tolist()
                    except Exception:
                        logger.warning(f'Failed to generate crop embeddings for {filename}')

                result.detections = detections

                # Faces
                faces = []
                if num_faces > 0:
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
                result.faces = faces

                # Embedding
                if include_embedding:
                    emb = yolo_clip_result.get('image_embedding')
                    if emb is not None:
                        result.global_embedding = np.array(emb).tolist()

                # OCR
                if ocr_result and ocr_result.get('status') == 'success':
                    result.ocr = OcrResult(
                        texts=ocr_result.get('texts', []),
                        boxes=ocr_result.get('boxes', []),
                        boxes_normalized=ocr_result.get('boxes_normalized', []),
                        det_scores=ocr_result.get('det_scores', []),
                        rec_scores=ocr_result.get('rec_scores', []),
                        full_text=' '.join(ocr_result.get('texts', [])),
                        num_texts=ocr_result.get('num_texts', 0),
                    )

            results.append(result)

        except Exception as e:
            logger.warning(f'Analysis failed for {filename}: {e}')
            failures.append(
                {
                    'filename': filename,
                    'index': idx,
                    'error': str(e),
                }
            )

    return BatchAnalyzeResponse(
        status='success' if results else 'all_failed',
        total_images=len(images),
        processed_images=len(results),
        failed_images=len(failures),
        results=results,
        failures=failures if failures else None,
        total_detections=total_dets,
        total_faces=total_faces,
        images_with_text=images_with_text,
    )
