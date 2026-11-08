"""Combine orchestration: preview, start, status, cancel and resume (projects
plan section 6). The routes are thin over this."""

from __future__ import annotations

import uuid
from typing import Any

from src.core.exceptions import RequestRejectedError
from src.core.logging import get_logger
from src.db.session import async_session_factory
from src.schemas.curation.config_common import ValidationIssue, ValidationReport
from src.services.jobs import service as jobs
from src.services.jobs.status import get_job
from src.services.projects import lifecycle
from src.services.projects.combine import runs
from src.services.projects.combine.execute import load_plan, plan_documents
from src.services.projects.combine.models import (
    CombineIssue,
    CombinePreview,
    CombineRequest,
    CombineStartRequest,
)
from src.services.projects.combine.plan import (
    Analysis,
    analyze,
    build_preview,
    check_target,
    compute_preview_sha,
    resolve_sources,
    source_state,
)
from src.services.projects.registry import get_project_registry
from src.services.rejections import rejection


logger = get_logger(__name__)

RESUMABLE_STATUSES = frozenset({'interrupted', 'paused'})

# The target is a new project: its own keymap and classes are not cloned.
_CLONE_AXES = ['settings_defaults', 'activations', 'prompt_packs']

_SLUG_ERRORS = frozenset({'slug_taken', 'slug_retired', 'slug_invalid'})


def target_space_keys() -> frozenset[str]:
    """The space keys a new target will have: its default encoder's (every project is created
    with the default spaces)."""
    from src.domain.embedding_spaces import space_key
    from src.services.projects.store import DEFAULT_ENCODER

    return frozenset({space_key(*DEFAULT_ENCODER)})


def report_of(issues: list[CombineIssue]) -> ValidationReport:
    return ValidationReport(
        ok=not issues,
        errors=[
            ValidationIssue(
                code=i.code,
                severity='error',
                field=i.project,
                message=i.message or i.code,
                detail=i.detail,
            )
            for i in issues
        ],
    )


async def preview(client: Any, request: CombineRequest) -> tuple[CombinePreview, Analysis | None]:
    """The dry run. Nothing is written; the analysis is returned too so
    ``start`` can persist exactly what was previewed."""
    records, errors = await resolve_sources(request)
    target_errors, warnings = await check_target(client, request)
    errors = [*errors, *target_errors]
    slug_available = not any(e.code in _SLUG_ERRORS for e in errors)
    if len(records) != len(request.sources):
        return _unreadable(request, errors, warnings, slug_available), None
    analysis = await analyze(client, request, records, target_keys=target_space_keys())
    states = [await source_state(client, r) for r in records]
    result = build_preview(
        analysis,
        errors=[*errors, *analysis.errors],
        warnings=[*warnings, *analysis.warnings],
        preview_sha=compute_preview_sha(request, states),
        slug_available=slug_available,
    )
    return result, analysis


def _unreadable(
    request: CombineRequest,
    errors: list[CombineIssue],
    warnings: list[CombineIssue],
    slug_available: bool,
) -> CombinePreview:
    return CombinePreview(
        ok=False,
        errors=errors,
        warnings=warnings,
        preview_sha='',
        suggested_mapping={},
        sources=[],
        target={'slug': request.target.slug, 'slug_available': slug_available},
        dedup={},
        bytes={'to_link': 0, 'to_copy': 0},
    )


async def start(client: Any, request: CombineStartRequest) -> dict[str, str]:
    """Validate, create the ``building`` target, persist the plan and queue the job.

    Returns ``{job_id, target}``; the work runs in a worker (``combine``, scope: the target).
    """
    plain = CombineRequest.model_validate(request.model_dump(exclude={'expected_preview_sha'}))
    result, analysis = await preview(client, plain)
    if not result.ok or analysis is None:
        raise rejection(
            422,
            'combine_invalid',
            'the combine request has errors',
            report=report_of(result.errors),
        )
    if result.preview_sha != request.expected_preview_sha:
        raise rejection(
            409,
            'preview_stale',
            'a source changed since the preview; preview again',
            project=request.target.slug,
        )
    params, document = plan_documents(analysis, result.preview_sha)
    slugs = [r.slug for r in analysis.records]
    target, _warnings = await lifecycle.create_project(
        client,
        slug=request.target.slug,
        display_name=request.target.display_name,
        description=request.target.description,
        clone_settings_from=request.settings_from,
        clone_axes=_CLONE_AXES if request.settings_from else None,
        origin={'kind': 'combine', 'sources': slugs},
        activate=False,
    )

    async def with_rows(session: Any, job: Any) -> None:
        await runs.insert_rows(
            session,
            job,
            target=target,
            sources=analysis.records,
            holdout=request.holdout,
            document=document,
        )

    try:
        async with async_session_factory()() as session:
            started = await jobs.start_job(
                session, 'combine', target.id, params=params, with_rows=with_rows
            )
    except BaseException:
        await lifecycle.finish_building(client, target, ok=False)
        raise
    return {'job_id': str(started.job.id), 'target': target.slug}


def _parse(job_id: str) -> uuid.UUID:
    """A malformed id names no job: answered before any database work."""
    try:
        return uuid.UUID(job_id)
    except ValueError:
        raise rejection(404, 'combine_not_found', f"no combine job '{job_id}'") from None


async def _run_of(session: Any, job_id: str) -> runs.CombineRunRow:
    run = await runs.find(session, _parse(job_id))
    if run is None:
        raise rejection(404, 'combine_not_found', f"no combine job '{job_id}'")
    return run


async def job_state(job_id: str) -> dict[str, Any]:
    _parse(job_id)
    async with async_session_factory()() as session:
        run = await _run_of(session, job_id)
        view = await get_job(session, run.job_id, project=run.target_project_id)
    progress = view.progress
    finished = view.result or {}
    return {
        'job_id': job_id,
        'status': view.status,
        'phase': progress.get('stage'),
        'target': view.project,
        'sources': run.sources,
        'done': int(progress.get('done') or 0),
        'total': int(progress.get('total') or 0),
        'report': finished.get('report') or progress.get('counters') or {},
        'next_steps': finished.get('next_steps') or [],
        'error': view.error_message,
        'started_at': view.started_at.isoformat() if view.started_at else None,
        'finished_at': view.finished_at.isoformat() if view.finished_at else None,
    }


async def cancel(job_id: str) -> dict[str, Any]:
    _parse(job_id)
    async with async_session_factory()() as session:
        run = await _run_of(session, job_id)
        try:
            await jobs.cancel_job(session, run.job_id, run.target_project_id, requested_by='api')
        except RequestRejectedError as exc:
            if isinstance(exc.detail, dict) and exc.detail.get('error') == 'job_finished':
                raise rejection(409, 'combine_not_resumable', 'the job is not running') from exc
            raise
    return await job_state(job_id)


async def resume(client: Any, job_id: str) -> dict[str, Any]:
    """Resume an interrupted combine. The sources go through the same :func:`resolve_sources`
    gate as a start and must still match the preview; the target must still be ``building``."""
    state = await job_state(job_id)
    if state['status'] not in RESUMABLE_STATUSES:
        raise rejection(
            409,
            'combine_not_resumable',
            f'the job is {state["status"]}; only an interrupted combine resumes',
        )
    async with async_session_factory()() as session:
        run = await _run_of(session, job_id)
        view = await get_job(session, run.job_id, project=run.target_project_id)
    plan = load_plan(view.params or {}, run.document)
    sources, errors = await resolve_sources(plan.request, ignore_job=job_id)
    if errors:
        raise rejection(
            409,
            'combine_not_resumable',
            'a source can no longer be combined',
            report=report_of(errors),
        )
    states = [await source_state(client, r) for r in sources]
    if compute_preview_sha(plan.request, states) != (view.params or {}).get('preview_sha'):
        raise rejection(
            409,
            'preview_stale',
            'a source changed since the preview this combine was started from; start a new combine',
        )
    target = await get_project_registry().lookup(state['target'])
    if target is None or target.status != 'building':
        raise rejection(409, 'combine_not_resumable', 'the target project is no longer building')
    async with async_session_factory()() as session:
        await jobs.resume_job(session, run.job_id, run.target_project_id)
    return await job_state(job_id)


__all__ = ['cancel', 'job_state', 'preview', 'resume', 'start']
