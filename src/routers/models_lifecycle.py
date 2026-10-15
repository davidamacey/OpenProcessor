"""
Model lifecycle routes: status, load, unload, delete and list.

- GET /models/{name}/status - Get model status (export/load state)
- POST /models/{name}/load - Load model in Triton
- POST /models/{name}/unload - Unload model from Triton
- DELETE /models/{name} - Delete model files
- GET /models - List all available models
"""

import logging
import shutil
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query

from src.routers.curation._api_error import api_error
from src.schemas.model_admin import ModelStatusResponse
from src.schemas.models import (
    ExportStatus,
    ModelDeleteResponse,
    ModelInfo,
    ModelListResponse,
    ModelLoadResponse,
)
from src.services.model_export import PYTORCH_MODELS_DIR, list_export_tasks
from src.services.model_repository import promoted_owner, resolve_load_dir, resolve_model_dirs
from src.services.model_unload_guard import UnloadRefusedError, check_unload
from src.services.training.triton_repo import set_explicitly_unloaded
from src.services.triton_control import TritonControlService


logger = logging.getLogger(__name__)

router = APIRouter()

TRITON_MODELS_DIR = Path('/app/models')


# =============================================================================
# Model Status Endpoint
# =============================================================================


@router.get('/{model_name}/status', response_model=ModelStatusResponse)
async def get_model_status(model_name: str) -> ModelStatusResponse:
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
        if task['triton_name'] == model_name and task['status'] not in [
            ExportStatus.COMPLETED,
            ExportStatus.FAILED,
        ]:
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
async def load_model(model_name: str) -> ModelLoadResponse:
    """
    Load a model into Triton server.

    The model must exist in the Triton model repository (models/ directory).
    Use this to load a model that was previously unloaded or just exported.

    **Note:** Model names should include the format suffix:
    - `{name}_trt` for standard TRT
    - `{name}_trt_end2end` for End2End TRT
    """
    model_dir = resolve_load_dir(TRITON_MODELS_DIR, model_name)
    if model_dir is None:
        raise HTTPException(
            status_code=404,
            detail=f'Model {model_name} not found in Triton repository',
        )
    model_name = model_dir.name

    triton = TritonControlService()
    success, message = await triton.load_model(model_name)

    if not success:
        raise HTTPException(status_code=500, detail=message)
    set_explicitly_unloaded(model_dir, False)

    return ModelLoadResponse(model_name=model_name, action='load', success=True, message=message)


@router.post('/{model_name}/unload', response_model=ModelLoadResponse)
async def unload_model(
    model_name: str, force: bool = Query(False, description='Also unload a core pipeline model')
) -> ModelLoadResponse:
    """
    Unload a model from Triton (files stay in the repository, reloadable).

    Model names include the format suffix (`{name}_trt`, `{name}_trt_end2end`).
    Same guard as the delete routes: 403 for the configured detector / OCR
    (even with ``force``), 409 for core models without ``force``.
    """
    try:
        check_unload(model_name, force=force)
    except UnloadRefusedError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc
    triton = TritonControlService()
    success, message = await triton.unload_model(model_name)

    if not success:
        raise HTTPException(status_code=500, detail=message)
    set_explicitly_unloaded(TRITON_MODELS_DIR / model_name, True)

    return ModelLoadResponse(model_name=model_name, action='unload', success=True, message=message)


# =============================================================================
# Delete Endpoint
# =============================================================================


@router.delete('/{model_name}', response_model=ModelDeleteResponse)
async def delete_model(
    model_name: str, force: bool = Query(False, description='Also delete a core pipeline model')
) -> ModelDeleteResponse:
    """
    Delete a model from the repository and unload from Triton.

    Removes:
    - PyTorch model file (pytorch_models/{name}.pt)
    - the directory named ``name`` and the export family it expands to
      (``_trt``, ``_trt_end2end``, the ONNX intermediate ``_end2end``)

    Also unloads it from Triton, through the shared unload guard first (403
    detector / OCR, 409 core without ``force``); nothing is deleted when any is
    refused. A project-promoted model is a 409 ``project_owned_model``.
    """
    triton = TritonControlService()
    deleted_files, unloaded = [], False
    model_dirs = resolve_model_dirs(TRITON_MODELS_DIR, model_name)
    for model_dir in model_dirs:
        owner = promoted_owner(model_dir)
        if owner is not None:
            raise api_error(
                409,
                'project_owned_model',
                f'{model_dir.name!r} was promoted by project {owner!r}; delete it with '
                f'DELETE /curation/projects/{owner}/models/{model_dir.name}, which carries '
                'the in-use and sharing guards.',
                owner_project=owner,
            )
    try:
        for model_dir in model_dirs:
            check_unload(model_dir.name, force=force)
    except UnloadRefusedError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc

    # Delete PyTorch model
    pt_file = PYTORCH_MODELS_DIR / f'{model_name}.pt'
    if pt_file.exists():
        pt_file.unlink()
        deleted_files.append(f'pytorch_models/{model_name}.pt')
        logger.info(f'Deleted PyTorch model: {pt_file}')

    for model_dir in model_dirs:
        # Unload from Triton first
        success, _ = await triton.unload_model(model_dir.name)
        if success:
            unloaded = True

        # Collect files for response
        deleted_files.extend(
            str(f.relative_to(TRITON_MODELS_DIR.parent))
            for f in model_dir.rglob('*')
            if f.is_file()
        )

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
async def list_models() -> ModelListResponse:
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
            if config.get('input'):
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
