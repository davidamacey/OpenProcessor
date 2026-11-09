"""``op.periodic.near_dup_rescore`` and ``op.periodic.near_dup_full_build``.

Writers mark members stale (``enqueue_pending``); the rescore tick turns pending rows into an
incremental ``near_dup_rescore`` job per project and level, so grouping is never done at request
time. A level that was never built has nothing to rescore and is left to its first full build.
The weekly tick starts a full ``near_dup_score`` of the image level (the item level follows it) to
bound the drift an incremental path accumulates. Both start jobs of an exclusive group, so a level
with a job in flight answers ``job_busy`` and is skipped until the next tick; the idempotency key
carries the time slot, so a message delivered twice starts one job.
"""

from __future__ import annotations

import datetime as dt
import time

from src.config.advisory_locks import NEAR_DUP_RESCORE
from src.core.exceptions import RequestRejectedError
from src.db.repos import near_dup as repo
from src.db.session import async_session_factory, project_tx
from src.domain import near_dup as rules
from src.services.jobs.service import start_job
from src.services.near_duplicates import profiles
from src.services.periodic.runner import TickProject, TickReport, for_each_project


REQUESTER = 'beat'
RESCORE_SLOT_S = 30


async def _start(project: TickProject, kind: str, level: str, key: str) -> bool:
    """Start ``kind`` for the level; ``False`` when a job of its group is already live."""
    async with async_session_factory()() as session:
        try:
            await start_job(
                session,
                kind,
                project.id,
                params={'level': level},
                requested_by=REQUESTER,
                idempotency_key=key,
            )
        except RequestRejectedError as exc:
            if isinstance(exc.detail, dict) and exc.detail.get('error') == 'job_busy':
                return False
            raise
    return True


async def _has_profile(project: TickProject) -> bool:
    async with async_session_factory()() as session, project_tx(session, project.id):
        try:
            await profiles.active_profile(session, project.id)
        except profiles.NoActiveProfileError:
            return False
    return True


async def _built_levels(project: TickProject, levels: tuple[str, ...]) -> list[tuple[str, int]]:
    """``(level, pending rows)`` for each level that has a current run."""
    async with async_session_factory()() as session, project_tx(session, project.id):
        built = [
            level
            for level in levels
            if await repo.current_run(session, project.id, level) is not None
        ]
        return [(level, await repo.pending_count(session, project.id, level)) for level in built]


async def rescore_tick(*, now: float | None = None) -> TickReport:
    slot = int((now if now is not None else time.time()) // RESCORE_SLOT_S)

    async def one(project: TickProject) -> None:
        due = [
            level
            for level, pending in await _built_levels(project, rules.PROJECT_LEVELS)
            if pending
        ]
        if not due or not await _has_profile(project):
            return
        for level in due:
            key = f'near_dup_rescore:{project.id}:{level}:{slot}'
            await _start(project, 'near_dup_rescore', level, key)

    return await for_each_project(NEAR_DUP_RESCORE, one, task='op.periodic.near_dup_rescore')


async def full_build_tick(*, today: dt.date | None = None) -> TickReport:
    year, week, _ = (today or dt.datetime.now(dt.UTC).date()).isocalendar()

    async def one(project: TickProject) -> None:
        if not await _built_levels(project, ('image',)) or not await _has_profile(project):
            return
        key = f'near_dup_full_build:{project.id}:{year}-W{week:02d}'
        await _start(project, 'near_dup_score', 'image', key)

    return await for_each_project(NEAR_DUP_RESCORE, one, task='op.periodic.near_dup_full_build')
