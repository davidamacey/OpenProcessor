"""``op.periodic.cluster_refresh``: start a ``cluster_run`` when a project's item pool has grown.

Per project, under the ``CLUSTER_REFRESH`` advisory lock: read the project's active item
clustering profile (its ``refresh`` rule), observe the pool, the unfinished region work, recent
ingest, live runs and a live assist loop, let ``refresh_schedule.step`` decide, and start the run
with ``cluster_run`` (research clustering 5.2). Nothing is queued unless a ``cluster`` consumer
refreshed its ``worker_capabilities`` row within the last minute, so work is never left for
a worker that is not there. The project's clock (``cluster_refresh_clock``) carries deferral and
backoff from one tick to the next.
"""

from __future__ import annotations

import time
import uuid

from src.config.advisory_locks import CLUSTER_REFRESH
from src.config.cluster_profile import DEFAULT_REFRESH_INTERVAL_S, REFRESH_INTERVAL_ENV
from src.config.curation_accessors import images_index, items_index
from src.config.env import env_int
from src.config.queue import TaskQueues
from src.config.region_state import RegionStatus
from src.core.exceptions import RequestRejectedError
from src.core.logging import get_logger
from src.db.repos import worker_liveness
from src.db.session import async_session_factory, deployment_tx, project_tx
from src.domain.embedding_spaces import Target
from src.services.clustering.job import load_profile, start_run
from src.services.clustering.partitions import list_partitions
from src.services.clustering.refresh_schedule import (
    ACTIVE_JOB_STATUSES,
    Action,
    Observation,
    Policy,
    ProjectSchedule,
    step,
)
from src.services.curation.assist_loop import service as assist_loop
from src.services.curation.clustering.pool_snapshot import item_pool_query
from src.services.curation.ingest_policy_store import get_ingest_policy
from src.services.curation.job_binding import opensearch_client
from src.services.curation.policy_doc_store import PolicyConflictError
from src.services.embedding_spaces.resolve import embedding_space
from src.services.jobs import status as job_status
from src.services.periodic.cluster_refresh_clock import (
    RefreshState,
    load_refresh_state,
    save_refresh_state,
)
from src.services.periodic.runner import TickProject, TickReport, for_each_project


logger = get_logger(__name__)

REQUESTER = 'beat'
TARGET = 'item'
KIND = 'cluster_run'
_FINISHED_PARTITION = frozenset({'ready', 'discarded'})
_UNFINISHED = (RegionStatus.PENDING_DETECTION, RegionStatus.PENDING_VERIFICATION)


def refresh_interval_s() -> int:
    """The beat interval (``0`` turns the tick off); ingest within twice of it counts as recent."""
    return env_int(REFRESH_INTERVAL_ENV, DEFAULT_REFRESH_INTERVAL_S, minimum=0)


async def _consumer_live() -> bool:
    async with async_session_factory()() as session, deployment_tx(session, 'cluster consumer'):
        return await worker_liveness.live_consumers(session, TaskQueues.CLUSTER) > 0


async def _pool_facts(vector_field: str, settle_s: int) -> tuple[int, int, bool]:
    """``(pool size, items waiting for the region stage, any image ingested within settle_s)``."""
    from src.config import get_region_fields

    client = await opensearch_client()
    try:
        pool = await client.count(
            index=items_index(), body={'query': item_pool_query(vector_field)}
        )
        waiting = await client.count(
            index=items_index(),
            body={'query': {'terms': {get_region_fields().status: list(_UNFINISHED)}}},
        )
        # Only "any" matters: no documents, stop at the first match. ``indexed_at`` is written
        # by the ingest host and ``now-`` read on the OpenSearch node; a clock skew shifts the
        # window by the skew, and the deferral bound keeps a fast clock from starving the run.
        recent = await client.search(
            index=images_index(),
            body={
                'size': 0,
                'terminate_after': 1,
                'track_total_hits': 1,
                'query': {'range': {'indexed_at': {'gte': f'now-{settle_s}s'}}},
            },
        )
    finally:
        await client.close()
    return int(pool['count']), int(waiting['count']), int(recent['hits']['total']['value']) > 0


async def _loop_live() -> bool:
    async with async_session_factory()() as session:
        try:
            await assist_loop.require_no_live_loop(session)
        except RequestRejectedError as exc:
            if isinstance(exc.detail, dict) and exc.detail.get('error') == 'assist_loop_running':
                return True
            raise
    return False


async def _tracked_status(project: uuid.UUID, job_id: str | None) -> str | None:
    if job_id is None:
        return None
    try:
        job = uuid.UUID(job_id)  # a clock carried over from 0.6.0 names a job that no longer exists
    except ValueError:
        return 'lost'
    async with async_session_factory()() as session:
        try:
            return (await job_status.get_job(session, job, project=project)).status
        except RequestRejectedError:
            return 'lost'


async def _live_run(project: uuid.UUID) -> bool:
    async with async_session_factory()() as session:
        page = await job_status.list_jobs(
            session,
            project=project,
            kinds=(KIND,),
            statuses=tuple(sorted(ACTIVE_JOB_STATUSES)),
            limit=1,
        )
    return bool(page.items)


async def _built_pool(project: uuid.UUID, level: str, vector_field: str) -> int:
    """Pool size of the newest partition of this space that finished (ready, or later discarded)."""
    async with async_session_factory()() as session, project_tx(session, project):
        for row in await list_partitions(session, target=TARGET, limit=50):
            if (
                row.level == level
                and row.embedding_field == vector_field
                and row.status in _FINISHED_PARTITION
            ):
                return row.n_items or 0
    return 0


def _clock(state: RefreshState) -> ProjectSchedule:
    return ProjectSchedule(
        prev_count=state.prev_count,
        pending_since=state.pending_since,
        retry_not_before=state.retry_not_before,
        job_id=state.job_id,
    )


def _same(state: RefreshState, sched: ProjectSchedule) -> bool:
    return (state.prev_count, state.pending_since, state.retry_not_before, state.job_id) == (
        sched.prev_count,
        sched.pending_since,
        sched.retry_not_before,
        sched.job_id,
    )


async def _start(project: uuid.UUID, level: str, vector_field: str) -> str | None:
    """The new run's job id; ``None`` when a run of this space is already live."""
    async with async_session_factory()() as session:
        try:
            started = await start_run(
                session,
                project,
                target=TARGET,
                level=level,
                embedding_field=vector_field,
                requested_by=REQUESTER,
            )
        except RequestRejectedError as exc:
            if isinstance(exc.detail, dict) and exc.detail.get('error') == 'job_busy':
                return None
            raise
    return str(started.job.id)


async def _refresh(project: TickProject, now: float, settle_s: int) -> None:
    if not (await get_ingest_policy()).clustering.automatic:
        return
    if not await _consumer_live():
        logger.info('cluster_refresh_no_consumer', project=project.slug)
        return
    space = await embedding_space(Target.ITEM)
    async with async_session_factory()() as session, project_tx(session, project.id):
        profile, _pin = await load_profile(session, project.id, TARGET, space.vector_field)
    state = await load_refresh_state()
    sched = _clock(state)
    count, unfinished, recent = await _pool_facts(space.vector_field, settle_s)
    action = step(
        sched,
        Observation(
            count=count,
            built=await _built_pool(project.id, profile.level, space.vector_field),
            unfinished=unfinished,
            tracked_status=await _tracked_status(project.id, sched.job_id),
            busy=await _live_run(project.id),
            loop_live=await _loop_live(),
            recent_ingest=recent,
        ),
        now,
        Policy(
            threshold=profile.refresh.growth_threshold,
            max_deferral_s=profile.refresh.max_deferral_s,
            retry_backoff_s=profile.refresh.retry_backoff_s,
        ),
    )
    if action is Action.START:
        job_id = await _start(project.id, profile.level, space.vector_field)
        if job_id is not None:
            sched.on_started(job_id)
    logger.info(
        'cluster_refresh', project=project.slug, count=count, action=action.value, run=sched.job_id
    )
    if _same(state, sched):
        return
    try:
        await save_refresh_state(
            RefreshState(
                revision=state.revision,
                prev_count=sched.prev_count,
                pending_since=sched.pending_since,
                retry_not_before=sched.retry_not_before,
                job_id=sched.job_id,
            )
        )
    except PolicyConflictError:
        logger.warning('cluster_refresh_state_conflict', project=project.slug)


async def tick(*, now: float | None = None) -> TickReport:
    clock = time.time() if now is None else now
    settle_s = 2 * refresh_interval_s()

    async def one(project: TickProject) -> None:
        await _refresh(project, clock, settle_s)

    return await for_each_project(CLUSTER_REFRESH, one, task='op.periodic.cluster_refresh')
