"""Combine projects into a new one: ``/projects/combine*`` (projects plan
section 6). Global routes (never scoped): a combine acts on several projects
and builds a new one. The work is in :mod:`src.services.projects.combine`."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from src.routers.curation.projects import global_router
from src.services.projects.combine import service
from src.services.projects.combine.models import CombinePreview, CombineRequest, CombineStartRequest
from src.services.projects.guard import make_curation_opensearch


class CombineStartResponse(BaseModel):
    job_id: str
    target: str


class CombineJobResponse(BaseModel):
    job_id: str
    status: str
    phase: str | None = None
    target: str | None = None
    sources: list[str] = []
    done: int = 0
    total: int = 0
    report: dict[str, Any] = {}
    next_steps: list[dict[str, Any]] = []
    error: str | None = None
    started_at: str | None = None
    finished_at: str | None = None


@global_router.post('/projects/combine/preview', response_model=CombinePreview)
async def combine_preview(body: CombineRequest) -> CombinePreview:
    """Dry run: what a combine would do. Writes nothing."""
    client = await make_curation_opensearch()
    result, _analysis = await service.preview(client, body)
    return result


@global_router.post('/projects/combine', response_model=CombineStartResponse, status_code=202)
async def combine_start(body: CombineStartRequest) -> CombineStartResponse:
    client = await make_curation_opensearch()
    return CombineStartResponse(**await service.start(client, body))


@global_router.get('/projects/combine/{job_id}', response_model=CombineJobResponse)
async def combine_status(job_id: str) -> CombineJobResponse:
    return CombineJobResponse(**service.job_state(job_id))


@global_router.post('/projects/combine/{job_id}/cancel', response_model=CombineJobResponse)
async def combine_cancel(job_id: str) -> CombineJobResponse:
    return CombineJobResponse(**service.cancel(job_id))


@global_router.post('/projects/combine/{job_id}/resume', response_model=CombineJobResponse)
async def combine_resume(job_id: str) -> CombineJobResponse:
    client = await make_curation_opensearch()
    return CombineJobResponse(**await service.resume(client, job_id))
