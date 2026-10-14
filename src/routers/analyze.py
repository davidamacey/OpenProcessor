"""
Combined Analysis Router - All Models in One Request.

Runs YOLO detection, face recognition, CLIP embedding, and OCR
on a single image in one unified request.

Endpoints:
- POST /analyze - All models on single image
- POST /analyze/batch - Batch combined analysis
"""

import logging
from typing import Annotated

from fastapi import APIRouter, File, HTTPException, Query, UploadFile

from src.config.settings import TritonModelConfig
from src.schemas.analyze import (
    AnalyzeResponse,
    BatchAnalyzeResponse,
    BatchAnalyzeResult,
    DetectionResult,
    FaceResult,
    OcrResult,
)
from src.schemas.detection import ImageMetadata
from src.services.analyze_pipeline import analyze_single_image, get_inference_service, ocr_failure
from src.services.ocr_service import get_ocr_service
from src.utils.affine import format_detections_from_triton
from src.utils.retry import RetryExhaustedError


logger = logging.getLogger(__name__)

router = APIRouter(
    prefix='/analyze',
    tags=['Combined Analysis'],
)


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
    2. SCRFD face detection + ArcFace embedding extraction
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
    filename = image.filename or 'uploaded_image'

    try:
        image_bytes = image.file.read()

        if not image_bytes:
            raise HTTPException(status_code=400, detail='Empty image file')

        return analyze_single_image(
            image_bytes,
            include_embedding=include_embedding,
            include_face_embeddings=include_face_embeddings,
            enable_ocr=enable_ocr,
            confidence=confidence,
            face_confidence=face_confidence,
        )

    except HTTPException:
        raise
    except RetryExhaustedError:
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

                # Generate CLIP crop embeddings via batched Triton call
                batch_orig_w = batch_orig_shape[1] if batch_orig_shape else 0
                batch_orig_h = batch_orig_shape[0] if batch_orig_shape else 0
                if include_embedding and detections and batch_orig_w > 0 and batch_orig_h > 0:
                    try:
                        import io

                        from PIL import Image

                        from src.utils.cpu_preprocess import center_crop_cpu

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
                        rec_errors=ocr_result.get('rec_errors', []),
                        full_text=' '.join(ocr_result.get('texts', [])),
                        num_texts=ocr_result.get('num_texts', 0),
                    )
                result.ocr_error = ocr_failure(ocr_result)

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
