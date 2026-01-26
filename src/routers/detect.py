"""
Object Detection Router - CPU Preprocessing + Triton TRT Inference.

Provides YOLO object detection:
- CPU preprocessing (letterbox, normalize)
- TensorRT End2End model with GPU NMS
- No DALI dependency

Endpoints:
- POST /detect - Single image detection
- POST /detect/batch - Batch detection (up to 64 images)
"""

import logging
from typing import Annotated

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import ORJSONResponse

from src.schemas.detection import BatchInferenceResult, InferenceResult
from src.services.inference import InferenceService


logger = logging.getLogger(__name__)

router = APIRouter(
    prefix='/detect',
    tags=['Object Detection'],
    default_response_class=ORJSONResponse,
)

# Service instance
inference_service = InferenceService()

# Default model configuration
DEFAULT_MODEL = 'yolov11_small_trt_end2end'
MAX_BATCH_SIZE = 64


@router.post('', response_model=InferenceResult)
def detect_single(
    image: Annotated[UploadFile, File(description='Image file (JPEG/PNG)')],
    model_name: Annotated[
        str, Query(description='Triton model name')
    ] = DEFAULT_MODEL,
    confidence: Annotated[
        float, Query(ge=0.0, le=1.0, description='Minimum confidence threshold')
    ] = 0.25,
):
    """
    Single image object detection using CPU preprocessing + Triton TRT.

    Pipeline:
    - CPU letterbox preprocessing (Ultralytics LetterBox)
    - TensorRT End2End model with compiled GPU NMS
    - Normalized bounding box coordinates [0, 1]

    Args:
        image: Image file (JPEG, PNG)
        model_name: Triton model name (default: yolov11_small_trt_end2end)
        confidence: Minimum detection confidence (default: 0.25)

    Returns:
        Detection results with normalized coordinates and metadata
    """
    filename = image.filename or 'uploaded_image'

    try:
        image_bytes = image.file.read()

        if not image_bytes:
            raise HTTPException(status_code=400, detail='Empty image file')

        # Use CPU preprocessing + TRT End2End inference
        result = inference_service.detect(
            image_bytes=image_bytes,
            model_name=model_name,
        )

        # Filter detections by confidence if threshold differs from model default
        if confidence > 0.25:
            result['detections'] = [
                det for det in result['detections']
                if det['confidence'] >= confidence
            ]
            result['num_detections'] = len(result['detections'])

        return result

    except ValueError as e:
        logger.warning(f'Invalid image {filename}: {e}')
        raise HTTPException(status_code=400, detail=str(e)) from e

    except Exception as e:
        logger.error(f'Detection failed for {filename}: {e}')
        raise HTTPException(status_code=500, detail=f'Detection failed: {e!s}') from e


@router.post('/batch', response_model=BatchInferenceResult)
def detect_batch(
    images: Annotated[
        list[UploadFile], File(description='Image files (JPEG/PNG), max 64')
    ],
    model_name: Annotated[
        str, Query(description='Triton model name')
    ] = DEFAULT_MODEL,
    confidence: Annotated[
        float, Query(ge=0.0, le=1.0, description='Minimum confidence threshold')
    ] = 0.25,
):
    """
    Batch object detection using CPU preprocessing + Triton TRT.

    Processes up to 64 images in a single request for improved throughput.
    Uses batched Triton calls for efficiency.

    Args:
        images: List of image files (max 64)
        model_name: Triton model name (default: yolov11_small_trt_end2end)
        confidence: Minimum detection confidence (default: 0.25)

    Returns:
        Batch results with per-image detections and summary statistics
    """
    if len(images) > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f'Batch size {len(images)} exceeds maximum {MAX_BATCH_SIZE}',
        )

    if len(images) == 0:
        return BatchInferenceResult(
            total_images=0,
            processed_images=0,
            failed_images=0,
            results=[],
            status='success',
        )

    all_results = []
    failed_images = []

    try:
        # Collect image data for batch processing
        images_data = []
        for idx, image in enumerate(images):
            filename = image.filename or f'image_{idx}'
            try:
                image_bytes = image.file.read()
                if not image_bytes:
                    failed_images.append({
                        'filename': filename,
                        'index': idx,
                        'error': 'Empty image file',
                    })
                    continue
                images_data.append((image_bytes, filename, idx))
            except Exception as e:
                logger.warning(f'Failed to read {filename}: {e}')
                failed_images.append({
                    'filename': filename,
                    'index': idx,
                    'error': str(e),
                })

        if not images_data:
            return BatchInferenceResult(
                total_images=len(images),
                processed_images=0,
                failed_images=len(failed_images),
                results=[],
                failures=failed_images,
                status='all_failed',
            )

        # Process each image through CPU preprocessing + TRT End2End
        for image_bytes, filename, idx in images_data:
            try:
                result = inference_service.detect(
                    image_bytes=image_bytes,
                    model_name=model_name,
                )

                # Filter detections by confidence
                detections = result['detections']
                if confidence > 0.25:
                    detections = [
                        det for det in detections
                        if det['confidence'] >= confidence
                    ]

                all_results.append({
                    'filename': filename,
                    'image_index': idx,
                    'detections': detections,
                    'num_detections': len(detections),
                    'status': 'success',
                    'track': 'C',
                    'image': result['image'],
                })

            except Exception as e:
                logger.warning(f'Detection failed for {filename}: {e}')
                failed_images.append({
                    'filename': filename,
                    'index': idx,
                    'error': str(e),
                })

        return BatchInferenceResult(
            total_images=len(images),
            processed_images=len(all_results),
            failed_images=len(failed_images),
            results=all_results,
            failures=failed_images if failed_images else None,
            status='success' if all_results else 'all_failed',
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'Batch detection failed: {e}')
        raise HTTPException(
            status_code=500,
            detail=f'Batch detection failed: {e!s}',
        ) from e
