"""The queued assist run: one ``assist_run`` job per project at a time, state in the ``jobs`` row.

``POST /assist/runs`` inserts the row through ``start_job`` (a second start while one is
active is the queue's 409 ``job_busy``) and keeps the scope's members in ``job_items`` for the
staleness check; the request, the resolved scope and the sampled ids are the job's params, so
the worker runs exactly the sample the user was shown. Progress, the result and the error are
the row's. :func:`state_from_view` is the one place a row becomes an :class:`AssistRunState`.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

from src.config.project_context import current_project_id
from src.core.exceptions import RequestRejectedError
from src.db.repos import job_items
from src.db.session import project_tx
from src.services.curation.assist.models import (
    AssistProgress,
    AssistResult,
    AssistRunRequest,
    AssistRunState,
    ResolvedScope,
)
from src.services.jobs import status as job_status
from src.services.jobs.service import cancel_job, start_job


if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from src.db.repos.jobs import JobRow
    from src.schemas.jobs import JobView
    from src.services.curation.assist.prepare import PreparedRun

KIND = 'assist_run'
ACTIVE = frozenset({'queued', 'running', 'paused', 'cancelling'})
KEEP_RUNS = 100
"""How many of the newest runs the list scans (the 0.6 file store kept this many)."""
WAITING_STAGE = 'waiting_region'


async def start(
    session: AsyncSession,
    project: uuid.UUID,
    prepared: PreparedRun,
    *,
    members: list[str] | None,
) -> JobRow:
    """Queue the run. ``members`` is the cluster scope the proposal is about (``None`` for a
    bare selection or a filter). 409 ``job_busy`` while another run of the project is active."""
    params: dict[str, Any] = {
        'request': prepared.request.model_dump(mode='json'),
        'resolved': prepared.resolved().model_dump(mode='json'),
        'sampled_ids': list(prepared.scope.sampled_ids),
        'box_ids': prepared.scope.box_ids,
        'members_total': None if members is None else len(members),
    }

    async def rows(inner: AsyncSession, job: JobRow) -> None:
        if members:
            await job_items.add_items(inner, job.id, members)

    started = await start_job(
        session, KIND, project, params=params, requested_by='api', with_rows=rows
    )
    return started.job


def _iso(value: Any) -> str | None:
    return None if value is None else str(value.isoformat())


def _status(view: JobView) -> str:
    if view.status == 'running' and view.progress.get('stage') == WAITING_STAGE:
        return 'waiting_region'
    return 'running' if view.status in ('paused', 'cancelling') else view.status


def state_from_view(view: JobView) -> AssistRunState:
    """The wire state of an ``assist_run`` job row read with its params."""
    params = view.params or {}
    resolved = ResolvedScope.model_validate(params['resolved'])
    progress = view.progress
    return AssistRunState(
        run_id=view.id,
        status=_status(view),
        created_at=_iso(view.created_at),
        finished_at=_iso(view.finished_at),
        error=view.error_message,
        request=AssistRunRequest.model_validate(params['request']),
        resolved=resolved,
        progress=AssistProgress(
            processed=int(progress.get('done') or 0),
            total=int(progress.get('total') or resolved.sampled),
        ),
        result=None if view.result is None else AssistResult.model_validate(view.result),
    )


def queued_state(job: JobRow, prepared: PreparedRun) -> AssistRunState:
    """The answer to the start request: the run as it was just queued."""
    return AssistRunState(
        run_id=str(job.id),
        status='queued',
        created_at=_iso(job.created_at),
        request=prepared.request,
        resolved=prepared.resolved(),
        progress=AssistProgress(total=len(prepared.scope.sampled_ids)),
    )


def _parse(run_id: str) -> uuid.UUID | None:
    try:
        return uuid.UUID(run_id)
    except ValueError:
        return None


async def read_view(session: AsyncSession, run_id: str) -> JobView | None:
    """The bound project's ``assist_run`` job ``run_id``, or ``None`` for any other id."""
    job_id = _parse(run_id)
    if job_id is None:
        return None
    try:
        view = await job_status.get_job(session, job_id, project=current_project_id())
    except RequestRejectedError as exc:
        if exc.status_code == 404:
            return None
        raise
    return view if view.kind == KIND else None


async def read_members(session: AsyncSession, view: JobView) -> list[str] | None:
    """The member ids the run resolved, or ``None`` when it kept none (a filter target)."""
    total = (view.params or {}).get('members_total')
    if total is None:
        return None
    project = current_project_id()
    async with project_tx(session, project):
        return await job_items.read_items(session, uuid.UUID(view.id), limit=int(total))


def _names_scope(state: AssistRunState, cluster_id: int | None, subid: str | None) -> bool:
    target = state.request.target
    scope = getattr(target, 'context', None) or target
    got_id = getattr(scope, 'cluster_id', None)
    got_subid = getattr(scope, 'cluster_subid', None)
    if cluster_id is None:
        return True
    return got_id == cluster_id and (subid is None or got_subid == subid)


async def list_runs(
    session: AsyncSession, *, cluster_id: int | None, cluster_subid: str | None, limit: int
) -> list[tuple[AssistRunState, list[str] | None]]:
    """The newest runs, newest first, whose target or context names that scope, each with the
    member ids it kept (``None`` when it kept none)."""
    page = await job_status.list_jobs(
        session, project=current_project_id(), kinds=[KIND], limit=KEEP_RUNS
    )
    out: list[tuple[AssistRunState, list[str] | None]] = []
    for item in page.items:
        view = await read_view(session, item.id)
        if view is None:
            continue
        state = state_from_view(view)
        if _names_scope(state, cluster_id, cluster_subid):
            out.append((state, await read_members(session, view)))
            if len(out) >= limit:
                break
    return out


async def cancel_run(session: AsyncSession, run_id: str) -> bool:
    """Ask a live run to stop after its current chunk; ``False`` when there is none."""
    view = await read_view(session, run_id)
    if view is None or view.status not in ACTIVE:
        return False
    await cancel_job(session, uuid.UUID(view.id), current_project_id(), requested_by='api')
    return True


__all__ = [
    'ACTIVE',
    'KEEP_RUNS',
    'KIND',
    'WAITING_STAGE',
    'cancel_run',
    'list_runs',
    'queued_state',
    'read_members',
    'read_view',
    'start',
    'state_from_view',
]
