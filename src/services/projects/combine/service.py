"""Combine orchestration: preview, start, status, cancel and resume (projects
plan section 6). The routes are thin over this."""

from __future__ import annotations

import asyncio
import shutil
from typing import TYPE_CHECKING, Any

from src.config.curation import get_curation_config
from src.core.logging import get_logger
from src.routers.curation._config_common_models import ValidationIssue, ValidationReport, api_error
from src.services.curation.dataset_import.store import now_iso
from src.services.curation.job_lock import exclusive_start_lock
from src.services.projects import lifecycle
from src.services.projects.combine import store as job_store
from src.services.projects.combine.execute import load_plan, persist_plan, run_combine
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
from src.services.projects.registry import get_record_with_seq


if TYPE_CHECKING:
    from src.config.projects import ProjectRecord
    from src.services.curation.dataset_import.store import ImportStore

logger = get_logger(__name__)

# Strong references so a fire-and-forget job is never collected mid-run.
_TASKS: dict[str, asyncio.Task[None]] = {}

# The target is a new project: its own keymap and classes are not cloned.
_CLONE_AXES = ['settings_defaults', 'activations', 'prompt_packs']

_SLUG_ERRORS = frozenset({'slug_taken', 'slug_retired', 'slug_invalid'})


def target_embedding_dim() -> int:
    """The vector size the target (the process-wide encoder) uses."""
    return get_curation_config().encoder_embedding_dim


def report_of(issues: list[CombineIssue]) -> ValidationReport:
    return ValidationReport(
        ok=not issues,
        errors=[
            ValidationIssue(
                code=i.code,  # type: ignore[arg-type]
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
    analysis = await analyze(client, request, records, target_dim=target_embedding_dim())
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
    """Validate, create the ``building`` target, persist the plan and run the
    job in the background. Returns ``{job_id, target}``."""
    plain = CombineRequest.model_validate(request.model_dump(exclude={'expected_preview_sha'}))
    result, analysis = await preview(client, plain)
    if not result.ok or analysis is None:
        raise api_error(
            422,
            'combine_invalid',
            'the combine request has errors',
            report=report_of(result.errors),
        )
    if result.preview_sha != request.expected_preview_sha:
        raise api_error(
            409,
            'preview_stale',
            'a source changed since the preview; preview again',
            project=request.target.slug,
        )
    job_id = job_store.new_job_id()
    store = job_store.create_job(job_id)
    persist_plan(store, analysis, result.preview_sha)
    slugs = [r.slug for r in analysis.records]
    try:
        target, _warnings = await lifecycle.create_project(
            client,
            slug=request.target.slug,
            display_name=request.target.display_name,
            description=request.target.description,
            clone_settings_from=request.settings_from,
            clone_axes=_CLONE_AXES if request.settings_from else None,
            origin={'kind': 'combine', 'job_id': job_id, 'sources': slugs},
            activate=False,
        )
    except BaseException:
        shutil.rmtree(store.directory, ignore_errors=True)
        raise
    store.job.write(
        {
            'job_id': job_id,
            'status': 'queued',
            'phase': 'queued',
            'target': target.slug,
            'sources': slugs,
            'done': 0,
            'total': sum(s.images for s in analysis.stats),
            'started_at': now_iso(),
        }
    )
    _spawn(client, store, analysis.records, target)
    return {'job_id': job_id, 'target': target.slug}


def _spawn(
    client: Any, store: ImportStore, sources: list[ProjectRecord], target: ProjectRecord
) -> None:
    task = asyncio.create_task(_run(client, store, sources, target))
    _TASKS[store.import_id] = task
    task.add_done_callback(lambda _t: _TASKS.pop(store.import_id, None))


async def _run(
    client: Any, store: ImportStore, sources: list[ProjectRecord], target: ProjectRecord
) -> None:
    """Run the job, then settle the target: ``active`` on success, ``failed`` on
    a failure; a cancelled or interrupted job leaves it ``building`` so it can
    resume."""

    async def settle(ok: bool) -> None:
        await lifecycle.finish_building(client, target, ok=ok)
        if ok:
            from src.services.curation.event_hub import publish_global_event

            publish_global_event(
                'project.created', target=target.slug, status='active', revision=target.revision
            )

    await run_combine(
        client,
        store=store,
        plan=load_plan(store),
        sources=sources,
        target=target,
        embedding_dim=target_embedding_dim(),
        settle=settle,
    )


def job_state(job_id: str) -> dict[str, Any]:
    store = job_store.open_job(job_id)
    if store is None:
        raise api_error(404, 'combine_not_found', f"no combine job '{job_id}'")
    state = store.job.repair_if_stale(job_store.ACTIVE_STATUSES, error_prefix='combine')
    return {'job_id': job_id, **state}


def cancel(job_id: str) -> dict[str, Any]:
    store = job_store.open_job(job_id)
    if store is None:
        raise api_error(404, 'combine_not_found', f"no combine job '{job_id}'")
    if store.job.read().get('status') not in job_store.ACTIVE_STATUSES:
        raise api_error(409, 'combine_not_resumable', 'the job is not running')
    store.job.request_cancel()
    return job_state(job_id)


async def resume(client: Any, job_id: str) -> dict[str, Any]:
    """Resume an interrupted or cancelled combine. One worker per job: the
    claim (the ``queued`` write) is made under the job's start lock, and the
    sources go through the same :func:`resolve_sources` gate as a start. A job
    whose worker is still alive in this process is not resumable, stale
    heartbeat or not; across processes the worker that notices a newer claim
    stops at its next chunk (:func:`~src.services.projects.combine.execute.
    ensure_owner`)."""
    store = job_store.open_job(job_id)
    if store is None:
        raise api_error(404, 'combine_not_found', f"no combine job '{job_id}'")
    with exclusive_start_lock(store.directory / 'start.lock') as acquired:
        if not acquired:
            raise api_error(409, 'combine_not_resumable', 'the job is already being resumed')
        running = _TASKS.get(job_id)
        if running is not None and not running.done():
            raise api_error(409, 'combine_not_resumable', 'a worker for this job is still running')
        state = job_state(job_id)
        if state.get('status') not in job_store.RESUMABLE_STATUSES:
            raise api_error(
                409,
                'combine_not_resumable',
                f'the job is {state.get("status")}; only an interrupted or cancelled combine '
                'resumes',
            )
        plan = load_plan(store)
        sources, errors = await resolve_sources(plan.request, ignore_job=job_id)
        if errors:
            raise api_error(
                409,
                'combine_not_resumable',
                'a source can no longer be combined',
                report=report_of(errors),
            )
        states = [await source_state(client, r) for r in sources]
        if compute_preview_sha(plan.request, states) != store.read_request().get('preview_sha'):
            raise api_error(
                409,
                'preview_stale',
                'a source changed since the preview this combine was started from; '
                'start a new combine',
            )
        target, _seq, _term = await get_record_with_seq(client, state['target'])
        if target is None or target.status != 'building':
            raise api_error(
                409, 'combine_not_resumable', 'the target project is no longer building'
            )
        store.job.clear_signals()
        store.job.update(status='queued', error=None, finished_at=None)
        store.job.touch_heartbeat()
    _spawn(client, store, sources, target)
    return job_state(job_id)


__all__ = ['cancel', 'job_state', 'preview', 'resume', 'start']
