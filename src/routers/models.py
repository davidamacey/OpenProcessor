"""
Model Management Router.

Provides endpoints for uploading, exporting, and managing YOLO models.
Delegates all business logic to service layer. The routes live in
``models_export`` and ``models_lifecycle``; they are included here, under the one
``/models`` prefix, in the order they were always registered. The test-inference
route stays here, last.

Endpoints:
- POST /models/upload - Upload custom YOLO model (.pt file)
- POST /models/{name}/export - Export uploaded model to TensorRT
- GET /models/{name}/status - Get model status (export/load state)
- POST /models/{name}/load - Load model in Triton
- POST /models/{name}/unload - Unload model from Triton
- DELETE /models/{name} - Delete model files
- GET /models - List all available models
- POST /models/{name}/test - Test inference on uploaded model
- GET /models/export/{task_id} - Get export task status
- GET /models/exports - List all export tasks
"""

import logging

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from src.routers import models_export, models_lifecycle
from src.schemas.model_admin import TestInferenceResponse
from src.services.triton_control import TritonControlService


logger = logging.getLogger(__name__)

router = APIRouter(
    prefix='/models',
    tags=['Model Management'],
)
router.include_router(models_export.router)
router.include_router(models_lifecycle.router)


# =============================================================================
# Test Inference Endpoint
# =============================================================================


@router.post('/{model_name}/test', response_model=TestInferenceResponse)
async def test_model_inference(
    model_name: str,
    file: UploadFile = File(..., description='Test image (JPEG or PNG)'),
    use_end2end: bool = Form(default=True),
) -> TestInferenceResponse:
    """
    Test inference on an uploaded/exported model.

    Runs a single inference to verify the model is working correctly.
    Uses End2End by default for GPU NMS, or standard TensorRT for CPU NMS.

    **Requirements:**
    - Model must be exported and loaded in Triton
    - Image must be JPEG or PNG format

    **Response includes:**
    - Detections with normalized [0,1] bounding boxes
    - Inference time in milliseconds
    - Image dimensions
    """
    import time

    from src.services.inference import InferenceService
    from src.utils.image_processing import decode_image, validate_image

    # Validate image file
    if not file.filename:
        raise HTTPException(status_code=400, detail='Filename required')

    valid_extensions = ('.jpg', '.jpeg', '.png')
    if not file.filename.lower().endswith(valid_extensions):
        raise HTTPException(status_code=400, detail='Image must be JPEG or PNG')

    content = await file.read()
    if len(content) < 100:
        raise HTTPException(status_code=400, detail='Invalid image file')

    # Determine Triton model name
    if use_end2end:
        triton_model_name = f'{model_name}_trt_end2end'
        track = 'C'
    else:
        triton_model_name = f'{model_name}_trt'
        track = 'B'

    # Verify model exists and is loaded
    triton = TritonControlService()
    models = await triton.get_repository_index()
    model_found = False
    model_ready = False

    for m in models:
        if m.get('name') == triton_model_name:
            model_found = True
            model_ready = m.get('state') == 'READY'
            break

    if not model_found:
        raise HTTPException(
            status_code=404,
            detail=f'Model {triton_model_name} not found. Export it first via POST /models/{model_name}/export',
        )

    if not model_ready:
        raise HTTPException(
            status_code=503,
            detail=f'Model {triton_model_name} not ready. Load it first via POST /models/{triton_model_name}/load',
        )

    # Run inference
    try:
        start_time = time.perf_counter()

        # Decode and validate image first
        img = decode_image(content, file.filename)
        validate_image(img, file.filename)

        # Use InferenceService for detection
        inference_service = InferenceService()

        # detect() uses YOLO End2End model by default
        result = inference_service.detect(
            image_bytes=content,
            model_name=triton_model_name,
        )

        inference_time = (time.perf_counter() - start_time) * 1000

        return TestInferenceResponse(
            model_name=triton_model_name,
            track=track,
            backend='triton',
            num_detections=result['num_detections'],
            detections=result['detections'],
            image=result['image'],
            inference_time_ms=round(inference_time, 2),
        )

    except Exception as e:
        logger.exception(f'Test inference failed for {model_name}')
        raise HTTPException(
            status_code=500,
            detail=f'Inference failed: {e!s}',
        ) from e
