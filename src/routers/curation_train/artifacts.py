"""Run lineage manifest and whitelisted artifact routes under ``/train``."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, HTTPException, Path as PathParam
from fastapi.responses import FileResponse, ORJSONResponse

from src.services.training import jobs as train_jobs


router = APIRouter(default_response_class=ORJSONResponse)


@router.get('/manifest/{job_id}')
async def get_manifest(
    job_id: Annotated[str, PathParam(description='Training job_id from {api_prefix}/train/runs')],
) -> ORJSONResponse:
    """Return the run's ``manifest.json`` (lineage envelope).

    404 if the manifest doesn't exist yet — older runs that finished
    before the manifest writer landed simply lack one.
    """
    manifest = await train_jobs.read_manifest(job_id)
    if manifest is None:
        raise HTTPException(status_code=404, detail=f'no manifest for job {job_id!r}')
    return ORJSONResponse(content=manifest)


# =============================================================================
# /artifacts/{job_id}/{name}
# =============================================================================


@router.get('/artifacts/{job_id}/{name}', response_class=FileResponse)
async def get_run_artifact(
    job_id: Annotated[str, PathParam(description='Training job_id from {api_prefix}/train/runs')],
    name: Annotated[
        str,
        PathParam(
            description=(
                'Whitelisted artifact filename '
                '(see src.services.training.jobs.RUN_ARTIFACT_WHITELIST), '
                'e.g. confusion_matrix.png'
            )
        ),
    ],
) -> FileResponse:
    """Serve one whitelisted metrics/plot artifact from a run's directory.

    This is the only sanctioned way to reach these files -- the server
    filesystem path itself never appears on the wire (``eval.
    confusion_matrix_url`` on ``{api_prefix}/train/status*``/``manifest``
    points here instead). 404 alike for an unwhitelisted name, an unknown
    job, or a file that hasn't been written yet — nothing here
    distinguishes those cases to a caller.
    """
    try:
        path = await train_jobs.read_artifact(job_id, name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if path is None:
        raise HTTPException(status_code=404, detail=f'no artifact {name!r} for job {job_id!r}')
    return FileResponse(path, media_type=train_jobs.artifact_media_type(name))
