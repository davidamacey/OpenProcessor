"""The queued reprocess job (``detect`` / ``embed`` over many images).

The start route inserts a ``reprocess`` row through ``start_job`` and stores the resolved image
ids in ``job_items`` in the same transaction; a Celery worker runs :func:`run_reprocess`. The
run is resumable: after each slice of images it writes ``{'done', 'results'}`` as the job's
checkpoint, so a delivery that dies (worker stop, the reaper's rule a) continues at the next
slice instead of starting over. It refuses to run when the bound project's indexes are not the
ones the request was planned against.
"""

from __future__ import annotations

import time
from datetime import datetime  # noqa: TC003 - pydantic reads the annotation
from typing import TYPE_CHECKING, Any

from src.config import get_curation_config
from src.db.repos import job_items
from src.services.curation.reprocess_embed_spec import EmbedSpec
from src.services.curation.reprocess_images import process_images
from src.services.curation.reprocess_models import (
    ReprocessJobInfo,
    ReprocessScope,
    ReprocessScopeResult,
)
from src.services.jobs.service import start_job


if TYPE_CHECKING:
    import uuid

    from opensearchpy import AsyncOpenSearch
    from sqlalchemy.ext.asyncio import AsyncSession

    from src.db.repos.jobs import JobRow
    from src.schemas.jobs import JobView
    from src.services.curation.ingest import CurationIngestService
    from src.tasks.runtime.context import JobContext

KIND = 'reprocess'
SLICE_IMAGES = 256
"""Images between two checkpoints (four chunks of ``process_images``)."""
ACTIVE = frozenset({'queued', 'running', 'paused', 'cancelling'})
_POLL_AFTER_S = 2


async def start(
    session: AsyncSession,
    project: uuid.UUID,
    *,
    request: dict[str, Any],
    scopes: list[ReprocessScope],
    image_ids: list[str],
    embed: EmbedSpec | None = None,
) -> JobRow:
    """Queue the job (409 ``job_busy`` while another reprocess of the project is active)."""
    cfg = get_curation_config()
    params = {
        'request': request,
        'scopes': scopes,
        'embed': embed.model_dump() if embed is not None else None,
        'images_total': len(image_ids),
        'items_index': cfg.items_index,
        'images_index': cfg.images_index,
    }

    async def rows(inner: AsyncSession, job: JobRow) -> None:
        await job_items.add_items(inner, job.id, image_ids)

    started = await start_job(
        session, KIND, project, params=params, requested_by='api', with_rows=rows
    )
    return started.job


def _iso(value: datetime | None) -> str | None:
    return None if value is None else value.isoformat()


def job_info(view: JobView) -> ReprocessJobInfo:
    """The family's wire view of a ``reprocess`` job row."""
    params = view.params or {}
    progress = view.progress
    counters = progress.get('counters') or {}
    results = (view.result or {}).get('results') or []
    return ReprocessJobInfo(
        job_id=view.id,
        status=view.status,
        scopes=params.get('scopes') or [],
        images_total=int(params.get('images_total') or progress.get('total') or 0),
        images_done=int(progress.get('done') or 0),
        images_failed=int(counters.get('failed') or 0),
        error=view.error_message,
        started_at=_iso(view.started_at),
        updated_at=_iso(view.updated_at),
        finished_at=_iso(view.finished_at),
        results=[ReprocessScopeResult.model_validate(r) for r in results],
        poll_after_s=_POLL_AFTER_S if view.status in ACTIVE else None,
    )


def _merge(into: ReprocessScopeResult, part: ReprocessScopeResult) -> None:
    for name in ('selected', 'locked_skipped', 'queued', 'not_found', 'failed'):
        setattr(into, name, getattr(into, name) + getattr(part, name))
    for key, value in part.detail.items():
        if isinstance(value, bool) or not isinstance(value, int | float):
            into.detail[key] = value
        else:
            current = into.detail.get(key, 0)
            into.detail[key] = (current if isinstance(current, int | float) else 0) + value


def _failed_images(results: dict[ReprocessScope, ReprocessScopeResult]) -> int:
    return max((r.failed + r.not_found for r in results.values()), default=0)


def _check_planned_for_this_project(params: Any) -> None:
    cfg = get_curation_config()
    if (params['items_index'], params['images_index']) != (cfg.items_index, cfg.images_index):
        raise RuntimeError(
            "the job was planned for other indexes than the bound project's; refusing to run"
        )


async def run_reprocess(
    ctx: JobContext, opensearch: AsyncOpenSearch, service: CurationIngestService
) -> tuple[dict[str, Any], bool]:
    """Execute the job: the existing :func:`process_images`, slice by slice, resumable.

    Returns the result mapping and whether any image failed (the handler wraps it)."""
    params = ctx.params
    _check_planned_for_this_project(params)
    scopes: list[ReprocessScope] = list(params['scopes'])
    embed = EmbedSpec(**params['embed']) if params.get('embed') else None
    total = int(params['images_total'])
    cursor = ctx.resume_from or {}
    done = int(cursor.get('done', 0))
    results = {s: ReprocessScopeResult(scope=s) for s in scopes}
    for stored in cursor.get('results', []):
        part = ReprocessScopeResult.model_validate(stored)
        results[part.scope] = part
    resumed_at = done
    started = time.monotonic()
    cancelled = False
    while done < total:
        if ctx.cancelled():
            cancelled = True
            break
        async with ctx.session() as session:
            ids = await job_items.read_items(session, ctx.job_id, start=done, limit=SLICE_IMAGES)
        if not ids:
            raise RuntimeError(f'job_items holds {done} of the {total} images of this job')
        parts, cancelled = await process_images(
            opensearch,
            service,
            scopes=scopes,
            image_ids=ids,
            embed=embed,
            should_cancel=ctx.cancelled,
        )
        for part in parts:
            _merge(results[part.scope], part)
        if cancelled:
            break
        done += len(ids)
        await ctx.checkpoint({'done': done, 'results': [r.model_dump() for r in results.values()]})
        await ctx.progress(done, total, 'processing', {'failed': _failed_images(results)})
    wall = time.monotonic() - started
    images = done - resumed_at
    failed = any(r.failed or r.not_found for r in results.values())
    return (
        {
            'results': [r.model_dump() for r in results.values()],
            'cancelled': cancelled,
            'perf_summary': {
                'images': images,
                'wall_s': wall,
                'images_per_s': images / wall if wall > 0 else 0.0,
                'decode_routes': {},
                'stage_seconds': {},
                'pe_calls_per_image': 0.0,
                'servers': [] if ctx.server is None else [ctx.server],
            },
        },
        failed,
    )


__all__ = ['ACTIVE', 'KIND', 'SLICE_IMAGES', 'job_info', 'run_reprocess', 'start']
