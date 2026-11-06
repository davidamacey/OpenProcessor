"""
Model upload and export routes.

- POST /models/upload - Upload custom YOLO model (.pt file)
- POST /models/{name}/export - Export uploaded model to TensorRT
- GET /models/export/{task_id} - Get export task status
- GET /models/exports - List all export tasks

The engine build runs in the model-export worker (the export image), not in the API: these
routes queue a job file for it and answer 503 when no worker is running. See
``src.services.model_export``.
"""

from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from src.core.disk_space import DISK_GUARD
from src.schemas.model_admin import ExportRequest
from src.schemas.models import ExportFormat, ExportTaskResponse, ExportTaskStatus
from src.services.model_export import (
    PYTORCH_MODELS_DIR,
    ExportWorkerUnavailableError,
    check_triton_name,
    create_export_task,
    enqueue_export_job,
    generate_triton_name,
    get_export_task,
    list_export_tasks,
    require_export_worker,
    save_uploaded_file,
    validate_pytorch_model,
)


router = APIRouter()

# File size limits
MAX_FILE_SIZE_MB = 500
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

_WORKER_DOWN: dict[int | str, dict[str, Any]] = {
    503: {'description': 'The model-export worker service is not running'}
}


def _require_worker() -> None:
    try:
        require_export_worker()
    except ExportWorkerUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


# =============================================================================
# Upload Endpoint
# =============================================================================


@router.post(
    '/upload', dependencies=[DISK_GUARD], response_model=ExportTaskResponse, responses=_WORKER_DOWN
)
async def upload_model(
    file: UploadFile = File(..., description='YOLO11 .pt model file'),
    triton_name: str | None = Form(default=None, description='Custom Triton model name'),
    max_batch: int = Form(default=32, ge=1, le=128, description='Max batch size'),
    formats: list[ExportFormat] = Form(default=[ExportFormat.TRT_END2END]),
    normalize_boxes: bool = Form(default=True),
    auto_load: bool = Form(default=True),
) -> ExportTaskResponse:
    """
    Upload a YOLO11 model and start export to TensorRT.

    The export runs in the model-export worker. Use /models/export/{task_id} to check status.
    Answers 503 when that worker service is not running.

    **File Requirements:**
    - Must be a .pt file (PyTorch checkpoint)
    - Maximum size: 500MB
    - Must be a YOLO detection model (not segmentation, pose, etc.)

    **Export Process:**
    1. File is validated as a valid YOLO detection model
    2. Saved to pytorch_models/ directory
    3. The export job is queued for the model-export worker
    4. The worker loads the model into Triton (if auto_load=True)

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

    # Before anything is saved or queued: nothing would ever run the job.
    _require_worker()

    try:
        name = generate_triton_name(file.filename, triton_name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    try:
        pt_file, model_info = await save_uploaded_file(content, name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    format_strs = [f.value for f in formats]

    # TRT End2End requires ONNX End2End as prerequisite
    if 'trt_end2end' in format_strs and 'onnx_end2end' not in format_strs:
        format_strs.insert(0, 'onnx_end2end')

    task_id = create_export_task(file.filename, name, model_info, format_strs)

    enqueue_export_job(
        task_id,
        pt_file=pt_file,
        triton_name=name,
        max_batch=max_batch,
        formats=format_strs,
        normalize_boxes=normalize_boxes,
        auto_load=auto_load,
    )

    task = get_export_task(task_id)
    if task is None:
        msg = f'Failed to retrieve task {task_id} after creation'
        raise RuntimeError(msg)
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


@router.post('/{model_name}/export', response_model=ExportTaskResponse, responses=_WORKER_DOWN)
async def export_model(
    model_name: str,
    request: ExportRequest,
) -> ExportTaskResponse:
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

    The export runs in the model-export worker (503 when it is not running). Returns task_id
    for status polling.
    """
    try:
        check_triton_name(model_name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    _require_worker()

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

    enqueue_export_job(
        task_id,
        pt_file=pt_file,
        triton_name=model_name,
        max_batch=request.max_batch,
        formats=format_strs,
        normalize_boxes=request.normalize_boxes,
        auto_load=request.auto_load,
    )

    task = get_export_task(task_id)
    if task is None:
        msg = f'Failed to retrieve task {task_id} after creation'
        raise RuntimeError(msg)
    return ExportTaskResponse(
        task_id=task_id,
        model_name=task['model_name'],
        triton_name=task['triton_name'],
        status=task['status'],
        message='Export started',
        created_at=task['created_at'],
    )


@router.get('/export/{task_id}', response_model=ExportTaskStatus)
async def get_export_status(task_id: str) -> ExportTaskStatus:
    """Get detailed status of an export task."""
    task = get_export_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f'Export task {task_id} not found')

    return ExportTaskStatus(**task)


@router.get('/exports', response_model=list[ExportTaskStatus])
async def list_exports() -> list[ExportTaskStatus]:
    """List all export tasks."""
    return [ExportTaskStatus(**t) for t in list_export_tasks()]
