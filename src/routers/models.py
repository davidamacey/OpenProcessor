"""
Model Management Router.

Provides endpoints for uploading, exporting, and managing YOLO models.
Delegates all business logic to service layer.

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
import shutil
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from src.schemas.models import (
    ExportFormat,
    ExportStatus,
    ExportTaskResponse,
    ExportTaskStatus,
    ModelDeleteResponse,
    ModelInfo,
    ModelListResponse,
    ModelLoadResponse,
)
from src.services.model_export import (
    create_export_task,
    generate_triton_name,
    get_export_task,
    list_export_tasks,
    run_export,
    save_uploaded_file,
    validate_pytorch_model,
    PYTORCH_MODELS_DIR,
)
from src.services.triton_control import TritonControlService


logger = logging.getLogger(__name__)

router = APIRouter(
    prefix='/models',
    tags=['Model Management'],
)

TRITON_MODELS_DIR = Path('/app/models')

# File size limits
MAX_FILE_SIZE_MB = 500
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024


# =============================================================================
# Response Models
# =============================================================================


class ModelStatusResponse(BaseModel):
    """Response for model status query."""

    name: str = Field(description='Model name')
    exists: bool = Field(description='Whether model files exist')
    pytorch_model: bool = Field(description='Whether .pt file exists in pytorch_models/')
    triton_trt: bool = Field(description='Whether TRT model exists in Triton')
    triton_trt_end2end: bool = Field(description='Whether TRT End2End model exists in Triton')
    triton_status: str | None = Field(default=None, description='Triton model status (READY, LOADING, etc)')
    export_status: ExportStatus | None = Field(default=None, description='Current export status if exporting')
    export_task_id: str | None = Field(default=None, description='Export task ID if exporting')
    num_classes: int | None = Field(default=None, description='Number of classes in model')
    file_size_mb: float | None = Field(default=None, description='PyTorch model file size in MB')


class TestInferenceRequest(BaseModel):
    """Request body for test inference."""

    confidence: float = Field(default=0.25, ge=0.0, le=1.0, description='Confidence threshold')
    use_end2end: bool = Field(default=True, description='Use End2End TRT model (GPU NMS)')


class TestInferenceResponse(BaseModel):
    """Response for test inference."""

    model_name: str
    track: str = Field(description='Inference track used (B or C)')
    backend: str = Field(description='Inference backend (triton)')
    num_detections: int
    detections: list[dict]
    image: dict = Field(description='Image dimensions (height, width)')
    inference_time_ms: float


class ExportRequest(BaseModel):
    """Request body for model export."""

    formats: list[ExportFormat] = Field(
        default=[ExportFormat.TRT_END2END],
        description='Export formats to generate',
    )
    max_batch: int = Field(
        default=32,
        ge=1,
        le=128,
        description='Maximum batch size for dynamic batching',
    )
    normalize_boxes: bool = Field(
        default=True,
        description='Output boxes in [0,1] normalized range',
    )
    auto_load: bool = Field(
        default=True,
        description='Automatically load model into Triton after export',
    )


# =============================================================================
# Upload Endpoint
# =============================================================================


@router.post('/upload', response_model=ExportTaskResponse)
async def upload_model(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description='YOLO11 .pt model file'),
    triton_name: str | None = Form(default=None, description='Custom Triton model name'),
    max_batch: int = Form(default=32, ge=1, le=128, description='Max batch size'),
    formats: list[ExportFormat] = Form(default=[ExportFormat.TRT_END2END]),
    normalize_boxes: bool = Form(default=True),
    auto_load: bool = Form(default=True),
):
    """
    Upload a YOLO11 model and start export to TensorRT.

    The export runs in the background. Use /models/export/{task_id} to check status.

    **File Requirements:**
    - Must be a .pt file (PyTorch checkpoint)
    - Maximum size: 500MB
    - Must be a YOLO detection model (not segmentation, pose, etc.)

    **Export Process:**
    1. File is validated as a valid YOLO detection model
    2. Saved to pytorch_models/ directory
    3. Export task starts in background
    4. Model is automatically loaded into Triton (if auto_load=True)

    Returns task_id for status polling.
    """
    # Validate file extension
    if not file.filename or not file.filename.endswith('.pt'):
        raise HTTPException(status_code=400, detail='File must be a .pt model file')

    # Read file content
    content = await file.read()

    # Validate file size
    if len(content) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f'File too large. Maximum size: {MAX_FILE_SIZE_MB}MB',
        )

    if len(content) < 1000:
        raise HTTPException(status_code=400, detail='File too small to be a valid model')

    # Generate Triton-compatible name
    name = triton_name or generate_triton_name(file.filename)

    try:
        pt_file, model_info = await save_uploaded_file(content, name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    format_strs = [f.value for f in formats]

    # TRT End2End requires ONNX End2End as prerequisite
    if 'trt_end2end' in format_strs and 'onnx_end2end' not in format_strs:
        format_strs.insert(0, 'onnx_end2end')

    task_id = create_export_task(file.filename, name, model_info, format_strs)

    background_tasks.add_task(
        run_export, task_id, pt_file, name, max_batch, format_strs, normalize_boxes, auto_load
    )

    task = get_export_task(task_id)
    return ExportTaskResponse(
        task_id=task_id,
        model_name=task['model_name'],
        triton_name=task['triton_name'],
        status=task['status'],
        message='Export started',
        created_at=task['created_at'],
    )


# =============================================================================
# Export Endpoints
# =============================================================================


@router.post('/{model_name}/export', response_model=ExportTaskResponse)
async def export_model(
    model_name: str,
    request: ExportRequest,
    background_tasks: BackgroundTasks,
):
    """
    Export an uploaded PyTorch model to TensorRT.

    The model must already exist in pytorch_models/ directory (from /upload).
    Use this endpoint to re-export with different settings.

    **Export Formats:**
    - `trt`: Standard TensorRT with CPU NMS
    - `trt_end2end`: TensorRT with GPU NMS (faster)
    - `onnx`: ONNX format (intermediate)
    - `onnx_end2end`: ONNX with NMS (intermediate for trt_end2end)
    - `all`: All formats

    Export runs in background. Returns task_id for status polling.
    """
    # Find the PyTorch model file
    pt_file = PYTORCH_MODELS_DIR / f'{model_name}.pt'

    if not pt_file.exists():
        # Try with common naming patterns
        candidates = list(PYTORCH_MODELS_DIR.glob(f'{model_name}*.pt'))
        if candidates:
            pt_file = candidates[0]
        else:
            raise HTTPException(
                status_code=404,
                detail=f'PyTorch model {model_name} not found. Upload it first via POST /models/upload',
            )

    # Validate the model
    is_valid, error_msg, model_info = validate_pytorch_model(pt_file)
    if not is_valid:
        raise HTTPException(status_code=400, detail=error_msg)

    format_strs = [f.value for f in request.formats]

    # TRT End2End requires ONNX End2End as prerequisite
    if 'trt_end2end' in format_strs and 'onnx_end2end' not in format_strs:
        format_strs.insert(0, 'onnx_end2end')

    task_id = create_export_task(pt_file.name, model_name, model_info, format_strs)

    background_tasks.add_task(
        run_export,
        task_id,
        pt_file,
        model_name,
        request.max_batch,
        format_strs,
        request.normalize_boxes,
        request.auto_load,
    )

    task = get_export_task(task_id)
    return ExportTaskResponse(
        task_id=task_id,
        model_name=task['model_name'],
        triton_name=task['triton_name'],
        status=task['status'],
        message='Export started',
        created_at=task['created_at'],
    )


@router.get('/export/{task_id}', response_model=ExportTaskStatus)
async def get_export_status(task_id: str):
    """Get detailed status of an export task."""
    task = get_export_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f'Export task {task_id} not found')

    return ExportTaskStatus(**task)


@router.get('/exports', response_model=list[ExportTaskStatus])
async def list_exports():
    """List all export tasks."""
    return [ExportTaskStatus(**t) for t in list_export_tasks()]


# =============================================================================
# Model Status Endpoint
# =============================================================================


@router.get('/{model_name}/status', response_model=ModelStatusResponse)
async def get_model_status(model_name: str):
    """
    Get comprehensive status of a model.

    Returns information about:
    - PyTorch model file presence
    - Triton TRT model status
    - Triton TRT End2End model status
    - Current export status (if exporting)
    - Model metadata (classes, file size)
    """
    # Check PyTorch model
    pt_file = PYTORCH_MODELS_DIR / f'{model_name}.pt'
    has_pytorch = pt_file.exists()
    file_size_mb = None
    num_classes = None

    if has_pytorch:
        file_size_mb = round(pt_file.stat().st_size / (1024 * 1024), 2)
        # Try to get class count from labels file
        labels_file = TRITON_MODELS_DIR / f'{model_name}_trt_end2end' / 'labels.txt'
        if not labels_file.exists():
            labels_file = TRITON_MODELS_DIR / f'{model_name}_trt' / 'labels.txt'
        if labels_file.exists():
            num_classes = len(labels_file.read_text().strip().split('\n'))

    # Check Triton models
    triton = TritonControlService()
    models = await triton.get_repository_index()

    triton_trt = False
    triton_trt_end2end = False
    triton_status = None

    for m in models:
        name = m.get('name', '')
        if name == f'{model_name}_trt':
            triton_trt = True
            triton_status = m.get('state', 'UNKNOWN')
        elif name == f'{model_name}_trt_end2end':
            triton_trt_end2end = True
            triton_status = m.get('state', 'UNKNOWN')

    # Check for active export task
    export_status = None
    export_task_id = None
    for task in list_export_tasks():
        if task['triton_name'] == model_name:
            if task['status'] not in [ExportStatus.COMPLETED, ExportStatus.FAILED]:
                export_status = task['status']
                export_task_id = task['task_id']
                break

    exists = has_pytorch or triton_trt or triton_trt_end2end

    return ModelStatusResponse(
        name=model_name,
        exists=exists,
        pytorch_model=has_pytorch,
        triton_trt=triton_trt,
        triton_trt_end2end=triton_trt_end2end,
        triton_status=triton_status,
        export_status=export_status,
        export_task_id=export_task_id,
        num_classes=num_classes,
        file_size_mb=file_size_mb,
    )


# =============================================================================
# Load/Unload Endpoints
# =============================================================================


@router.post('/{model_name}/load', response_model=ModelLoadResponse)
async def load_model(model_name: str):
    """
    Load a model into Triton server.

    The model must exist in the Triton model repository (models/ directory).
    Use this to load a model that was previously unloaded or just exported.

    **Note:** Model names should include the format suffix:
    - `{name}_trt` for standard TRT
    - `{name}_trt_end2end` for End2End TRT
    """
    # Check if model directory exists
    model_dir = TRITON_MODELS_DIR / model_name
    if not model_dir.exists():
        # Try with _trt_end2end suffix
        model_dir = TRITON_MODELS_DIR / f'{model_name}_trt_end2end'
        if model_dir.exists():
            model_name = f'{model_name}_trt_end2end'
        else:
            # Try with _trt suffix
            model_dir = TRITON_MODELS_DIR / f'{model_name}_trt'
            if model_dir.exists():
                model_name = f'{model_name}_trt'
            else:
                raise HTTPException(
                    status_code=404,
                    detail=f'Model {model_name} not found in Triton repository',
                )

    triton = TritonControlService()
    success, message = await triton.load_model(model_name)

    if not success:
        raise HTTPException(status_code=500, detail=message)

    return ModelLoadResponse(model_name=model_name, action='load', success=True, message=message)


@router.post('/{model_name}/unload', response_model=ModelLoadResponse)
async def unload_model(model_name: str):
    """
    Unload a model from Triton server.

    Frees GPU memory by removing the model from Triton.
    The model files remain in the repository and can be reloaded.

    **Note:** Model names should include the format suffix:
    - `{name}_trt` for standard TRT
    - `{name}_trt_end2end` for End2End TRT
    """
    triton = TritonControlService()
    success, message = await triton.unload_model(model_name)

    if not success:
        raise HTTPException(status_code=500, detail=message)

    return ModelLoadResponse(model_name=model_name, action='unload', success=True, message=message)


# =============================================================================
# Delete Endpoint
# =============================================================================


@router.delete('/{model_name}', response_model=ModelDeleteResponse)
async def delete_model(model_name: str):
    """
    Delete a model from the repository and unload from Triton.

    Removes:
    - PyTorch model file (pytorch_models/{name}.pt)
    - TRT model directory (models/{name}_trt/)
    - TRT End2End model directory (models/{name}_trt_end2end/)

    Also unloads the model from Triton if currently loaded.
    """
    triton = TritonControlService()
    deleted_files = []
    unloaded = False

    # Delete PyTorch model
    pt_file = PYTORCH_MODELS_DIR / f'{model_name}.pt'
    if pt_file.exists():
        pt_file.unlink()
        deleted_files.append(f'pytorch_models/{model_name}.pt')
        logger.info(f'Deleted PyTorch model: {pt_file}')

    # Delete Triton model directories
    for suffix in ['_trt', '_trt_end2end']:
        model_dir = TRITON_MODELS_DIR / f'{model_name}{suffix}'
        if model_dir.exists():
            # Unload from Triton first
            success, _ = await triton.unload_model(f'{model_name}{suffix}')
            if success:
                unloaded = True

            # Collect files for response
            for f in model_dir.rglob('*'):
                if f.is_file():
                    deleted_files.append(str(f.relative_to(TRITON_MODELS_DIR.parent)))

            # Delete directory
            shutil.rmtree(model_dir)
            logger.info(f'Deleted model directory: {model_dir}')

    if not deleted_files:
        raise HTTPException(status_code=404, detail=f'Model {model_name} not found')

    return ModelDeleteResponse(
        model_name=model_name,
        deleted_files=deleted_files,
        unloaded_from_triton=unloaded,
        message=f'Model {model_name} deleted successfully',
    )


# =============================================================================
# List Models Endpoint
# =============================================================================


@router.get('/', response_model=ModelListResponse)
async def list_models():
    """
    List all models in Triton repository.

    Returns all models including:
    - Pre-configured models (yolov11_small, etc.)
    - Custom uploaded models
    - Model status (READY, LOADING, UNAVAILABLE)
    - Available versions
    - Class count (if labels.txt exists)
    """
    triton = TritonControlService()
    ready = await triton.server_ready()
    models = await triton.get_repository_index()

    model_list = []
    for m in models:
        name = m.get('name', '')
        state = m.get('state', 'UNKNOWN')

        labels_path = TRITON_MODELS_DIR / name / 'labels.txt'
        has_labels = labels_path.exists()
        num_classes = None
        if has_labels:
            num_classes = len(labels_path.read_text().strip().split('\n'))

        # Get config info if available
        config = await triton.get_model_config(name)
        backend = None
        max_batch_size = None
        input_shape = None

        if config:
            backend = config.get('backend') or config.get('platform', '').replace('_plan', '')
            max_batch_size = config.get('max_batch_size')
            if 'input' in config and config['input']:
                input_config = config['input'][0]
                if 'dims' in input_config:
                    input_shape = input_config['dims']

        model_list.append(
            ModelInfo(
                name=name,
                status=state,
                versions=m.get('versions', []),
                has_labels=has_labels,
                num_classes=num_classes,
                backend=backend,
                max_batch_size=max_batch_size,
                input_shape=input_shape,
            )
        )

    return ModelListResponse(
        models=model_list,
        total=len(model_list),
        triton_status='READY' if ready else 'UNAVAILABLE',
    )


# =============================================================================
# Test Inference Endpoint
# =============================================================================


@router.post('/{model_name}/test', response_model=TestInferenceResponse)
async def test_model_inference(
    model_name: str,
    file: UploadFile = File(..., description='Test image (JPEG or PNG)'),
    confidence: float = Form(default=0.25, ge=0.0, le=1.0),
    use_end2end: bool = Form(default=True),
):
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
        image_shape = img.shape[:2]  # (height, width)

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
            detail=f'Inference failed: {str(e)}',
        ) from e
