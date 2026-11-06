"""Handlers of ``cluster_run`` and ``cluster_projection_sync`` and the calls that start them.

The handlers run in a worker with the project's scope bound by the job wrapper; they bind the
project's record for the index names and read vectors through the embedding registry
(``embedding_space``), never an environment field name.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

from src.config.cluster_profile import TARGET_AXES, default_profile_body
from src.config.curation_accessors import items_index
from src.config.project_context import bind_project
from src.db.repos import config as config_repo, config_activations
from src.domain.embedding_spaces import Target
from src.schemas.curation.research_config import ClusterProfileBody
from src.services.clustering import activation as activation_service
from src.services.clustering.partitions import PartitionSpace
from src.services.clustering.run import RunRefused, run_partition
from src.services.embedding_spaces.resolve import embedding_space
from src.services.jobs.service import start_job
from src.services.projects.script_binding import resolve_project


if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from src.tasks.runtime.context import JobContext

RUN_KIND = 'cluster_run'
PROJECTION_KIND = 'cluster_projection_sync'
ACTOR_AUTO = 'system:cluster_run'
_SPACE_TARGET = {'item': Target.ITEM, 'region_box': Target.BOX}


def scope_key(target: str, level: str, embedding_field: str) -> str:
    """The exclusivity scope of both kinds: one run or projection per (target, level, field)."""
    return f'{target}:{level}:{embedding_field}'


async def load_profile(
    session: AsyncSession, project: uuid.UUID, target: str
) -> tuple[ClusterProfileBody, tuple[uuid.UUID, int] | None]:
    """The project's active profile for ``target`` with its (document, revision) pin, else the defaults."""
    ref = await config_activations.get_activation(session, project, TARGET_AXES[target])
    if ref is not None and ref.state == 'active' and ref.name is not None:
        doc = await config_repo.get_config(
            session,
            project_id=project,
            kind='cluster_profile',
            name=ref.name,
            revision=ref.revision,
        )
        if doc is not None:
            return ClusterProfileBody.model_validate(doc.body), (doc.id, doc.revision)
    return ClusterProfileBody.model_validate(default_profile_body(target)), None


async def start_run(
    session: AsyncSession,
    project: uuid.UUID,
    *,
    target: str,
    level: str,
    embedding_field: str,
    requested_by: str,
) -> Any:
    """Queue a run; a second start for the same key while one is active is 409 ``job_busy``."""
    return await start_job(
        session,
        RUN_KIND,
        project,
        params={
            'target': target,
            'level': level,
            'scope_key': scope_key(target, level, embedding_field),
        },
        requested_by=requested_by,
    )


async def start_projection(
    session: AsyncSession,
    project: uuid.UUID,
    *,
    partition_id: uuid.UUID,
    target: str,
    level: str,
    embedding_field: str,
    requested_by: str,
) -> Any:
    return await start_job(
        session,
        PROJECTION_KIND,
        project,
        params={
            'partition_id': str(partition_id),
            'target': target,
            'level': level,
            'scope_key': scope_key(target, level, embedding_field),
        },
        idempotency_key=f'cluster_partition:{partition_id}',
        requested_by=requested_by,
    )


async def _snapshot(ctx: JobContext, target: str, vector_field: str) -> tuple[list[str], Any]:
    from src.services.curation.clustering import pool_snapshot

    client = await ctx.client('opensearch')
    if target == 'region_box':
        return await pool_snapshot.snapshot_boxes(client)
    return await pool_snapshot.snapshot_items(client, vector_field)


async def cluster_run(ctx: JobContext) -> dict[str, Any]:
    """Fit the pool, write the partition and, when the profile and the diff allow, activate it."""
    if ctx.project_id is None or ctx.project_slug is None:
        raise RunRefused('cluster_run needs a project')
    target = str(ctx.params['target'])
    level = str(ctx.params.get('level', 'class'))
    record = await resolve_project(ctx.project_slug)
    with bind_project(record):
        space = await embedding_space(_SPACE_TARGET[target])
        await ctx.progress(0, None, 'profile')
        async with ctx.session() as session:
            profile, pin = await load_profile(session, ctx.project_id, target)
            current = await activation_service.get_activation(
                session,
                project=ctx.project_id,
                target=target,
                level=level,
                embedding_field=space.vector_field,
            )
        await ctx.progress(0, None, 'snapshot')
        keys, vectors = await _snapshot(ctx, target, space.vector_field)
        if ctx.cancelled():
            return {'status': 'cancelled'}
        await ctx.progress(len(keys), len(keys), 'fit')
        async with ctx.session() as session:
            outcome = await run_partition(
                session,
                project=ctx.project_id,
                target=target,
                level=level,
                space=PartitionSpace.of(space),
                method=profile.method,
                params=profile.params,
                keys=keys,
                vectors=vectors,
                parent_partition_id=current.serving_partition_id if current else None,
                profile=pin,
                job_id=ctx.job_id,
                created_by=ACTOR_AUTO,
            )
            activated = False
            if activation_service.auto_activation_allowed(
                outcome.diff, activation=profile.activation
            ):
                await activation_service.activate(
                    session,
                    project=ctx.project_id,
                    partition_id=outcome.partition_id,
                    expected_serving=current.serving_partition_id if current else None,
                    expected_revision=current.revision if current else None,
                    by=ACTOR_AUTO,
                    reason='auto',
                )
                activated = True
    if activated:
        from src.db.session import async_session_factory

        async with async_session_factory()() as session:
            await start_projection(
                session,
                ctx.project_id,
                partition_id=outcome.partition_id,
                target=target,
                level=level,
                embedding_field=space.vector_field,
                requested_by=ACTOR_AUTO,
            )
    return {
        'status': 'success',
        'partition_id': str(outcome.partition_id),
        'n_items': outcome.n_items,
        'n_clusters': outcome.n_clusters,
        'activated': activated,
        'stability': outcome.stability,
    }


async def cluster_projection_sync(ctx: JobContext) -> dict[str, Any]:
    """Write a pending partition onto the items, fenced, and flip it to serving."""
    from src.services.clustering.projection import project_partition

    if ctx.project_id is None or ctx.project_slug is None:
        raise RunRefused('cluster_projection_sync needs a project')
    partition_id = uuid.UUID(str(ctx.params['partition_id']))
    target = str(ctx.params['target'])
    level = str(ctx.params.get('level', 'class'))
    record = await resolve_project(ctx.project_slug)
    client = await ctx.client('opensearch')
    with bind_project(record):
        async with ctx.session() as session:
            space = await embedding_space(_SPACE_TARGET[target])
            current = await activation_service.get_activation(
                session,
                project=ctx.project_id,
                target=target,
                level=level,
                embedding_field=space.vector_field,
            )
            old = (
                current.serving_partition_id
                if current and current.serving_partition_id != partition_id
                else None
            )

            async def on_batch(written: int) -> None:
                await ctx.progress(written, None, 'project')

            result = await project_partition(
                session,
                client,
                project=ctx.project_id,
                partition_id=partition_id,
                index=items_index(),
                target=target,
                old_partition_id=old,
                on_batch=on_batch,
            )
    return {
        'status': result.status,
        'written': result.written,
        'cleared': result.cleared,
        'failed': result.failed,
    }
