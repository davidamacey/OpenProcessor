"""
Model upload and export routes.

- POST /models/upload - Upload custom YOLO model (.pt file)
- POST /models/{name}/export - Export uploaded model to TensorRT
- GET /models/export/{task_id} - Get export task status
- GET /models/exports - List all export tasks

The engine build runs in the model-export worker (the export image), not in the API: these
routes start a ``model_export`` job for it and answer 503 when no worker consumes the
``gpu_export`` queue, 409 while the same model is being built. See
``src.services.model_export_jobs``.
"""

import uuid
from datetime import datetime
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from src.core.disk_space import DISK_GUARD
from src.db.session import async_session_factory
from src.schemas.model_admin import ExportRequest
from src.schemas.models import ExportFormat, ExportStatus, ExportTaskResponse, ExportTaskStatus
from src.services.model_export import (
    PYTORCH_MODELS_DIR,
    ExportWorkerUnavailableError,
    check_triton_name,
    generate_triton_name,
    stage_uploaded_file,
    validate_pytorch_model,
)
from src.services.model_export_jobs import (
    ensure_not_building,
    get_export_task,
    list_export_tasks,
    require_export_consumer,
    start_export,
)


router = APIRouter()

# File size limits
MAX_FILE_SIZE_MB = 500
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

_REFUSED: dict[int | str, dict[str, Any]] = {
    409: {'description': 'An export of this model name is already active'},
    503: {'description': 'The model-export worker service is not running'},
}


async def _require_worker() -> None:
    try:
        await require_export_consumer()
    except ExportWorkerUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def _started(job_id: uuid.UUID, params: dict[str, Any], created_at: datetime) -> ExportTaskResponse:
    return ExportTaskResponse(
        task_id=str(job_id),
        model_name=params['model_name'],
        triton_name=params['triton_name'],
        status=ExportStatus.PENDING,
        message='Export started',
        created_at=created_at,
    )


# =============================================================================
# Upload Endpoint
# =============================================================================


@router.post(
    '/upload', dependencies=[DISK_GUARD], response_model=ExportTaskResponse, responses=_REFUSED
)
async def upload_model(
    file: UploadFile = File(..., description='YOLO11 .pt model file'),
    triton_name: str | None = Form(default=None, description='Custom Triton model name'),
    max_batch: int = Form(default=32, ge=1, le=128, description='Max batch size'),
    formats: list[ExportFormat] = Form(default=[ExportFormat.TRT]),
    auto_load: bool = Form(default=True),
) -> ExportTaskResponse:
    """
    Upload a YOLO11 model and start export to TensorRT.

    The export runs in the model-export worker. Use /models/export/{task_id} to check status.
    Answers 503 when that worker service is not running and 409 while this model is being built.

    **File Requirements:**
    - Must be a .pt file (PyTorch checkpoint)
    - Maximum size: 500MB
    - Must be a YOLO detection model (not segmentation, pose, etc.)

    **Export Process:**
    1. File is validated as a valid YOLO detection model
    2. Saved to pytorch_models/ directory
    3. A model_export job is started for the model-export worker
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
    await _require_worker()

    try:
        name = generate_triton_name(file.filename, triton_name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    # Before the weights are replaced: a running build of this name is reading them.
    async with async_session_factory()() as session:
        await ensure_not_building(session, name)

    try:
        staged = await stage_uploaded_file(content, name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    format_strs = [f.value for f in formats]

    try:
        started = await start_export(
            triton_name=name,
            pt_file=staged.final_path,
            max_batch=max_batch,
            formats=format_strs,
            auto_load=auto_load,
            model_info=staged.model_info,
            new_version=True,
            staged=staged,
        )
    except BaseException:
        staged.discard()
        raise
    return _started(started.job.id, started.job.params, started.job.created_at)


# =============================================================================
# Export Endpoints
# =============================================================================


@router.post('/{model_name}/export', response_model=ExportTaskResponse, responses=_REFUSED)
async def export_model(
    model_name: str,
    request: ExportRequest,
) -> ExportTaskResponse:
    """
    Export an uploaded PyTorch model to TensorRT.

    The model must already exist in pytorch_models/ directory (from /upload).
    Use this endpoint to re-export with different settings.

    **Export Formats:**
    - `trt`: TensorRT engine (the fused ONNX it is built from is kept)
    - `onnx`: the fused ONNX only
    - `all`: same as `trt`

    Every detection family exports to one output, `output0 [300, 6]`: `x1, y1, x2, y2, score,
    class` per row, in letterboxed input pixels.

    The export runs in the model-export worker (503 when it is not running). Returns task_id
    for status polling.
    """
    try:
        check_triton_name(model_name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    await _require_worker()

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

    started = await start_export(
        triton_name=model_name,
        pt_file=pt_file,
        max_batch=request.max_batch,
        formats=format_strs,
        auto_load=request.auto_load,
        model_info=model_info,
        new_version=False,
    )
    return _started(started.job.id, started.job.params, started.job.created_at)


@router.get('/export/{task_id}', response_model=ExportTaskStatus)
async def get_export_status(task_id: str) -> ExportTaskStatus:
    """Get detailed status of an export task."""
    task = await get_export_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f'Export task {task_id} not found')
    return task


@router.get('/exports', response_model=list[ExportTaskStatus])
async def list_exports() -> list[ExportTaskStatus]:
    """List all export tasks."""
    return await list_export_tasks()
