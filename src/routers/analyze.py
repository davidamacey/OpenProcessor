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

from src.schemas.analyze import AnalyzeResponse, BatchAnalyzeResponse
from src.services.analyze_pipeline import (
    BatchTotals,
    analyze_batch_item,
    analyze_single_image,
    get_inference_service,
)
from src.services.ocr_service import get_ocr_service
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
) -> AnalyzeResponse:
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
) -> BatchAnalyzeResponse:
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
    totals = BatchTotals()

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

            results.append(
                analyze_batch_item(
                    image_bytes,
                    filename=filename,
                    idx=idx,
                    service=service,
                    ocr_service=ocr_service,
                    totals=totals,
                    include_embedding=include_embedding,
                    include_face_embeddings=include_face_embeddings,
                    enable_ocr=enable_ocr,
                    include_full_results=include_full_results,
                    confidence=confidence,
                )
            )

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
        total_detections=totals.detections,
        total_faces=totals.faces,
        images_with_text=totals.images_with_text,
    )
