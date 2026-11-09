"""Rows of tables that have no stage-2 repository: written here, in the caller's transaction.

These are the job-shaped tables (``jobs`` and what points at them), the ingest sources, the
retired slugs, VLM probes, the view state of the projection and the deployment settings.
Everything is written with the ids the parsers derived, so a re-run gives the same rows. The
ledger tables go through :mod:`ledger_copy` instead.
"""

from __future__ import annotations

import datetime as dt
from typing import TYPE_CHECKING

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert

from src.db.models import (
    CombineRun,
    CombineSource,
    ConfigDocument,
    DatasetExport,
    DatasetImport,
    DeploymentCounter,
    DeploymentSetting,
    IngestSource,
    Model,
    ModelShare,
    Project,
    ProjectionState,
    ProjectSlug,
    Promotion,
    TrainingRun,
)
from src.db.repos.jobs_load import JobLoad, load_jobs


if TYPE_CHECKING:
    import uuid
    from collections.abc import Iterable, Mapping, Sequence

    from sqlalchemy.ext.asyncio import AsyncSession

    from src.services.upgrade.from_060.model import (
        CombineSrc,
        ExportSrc,
        GlobalData,
        ImportSrc,
        IngestSourceSrc,
        JobRow,
        PromotionSrc,
        RetiredSlug,
        TrainingRunSrc,
        ViewStateSrc,
    )

UPDATED_BY = 'upgrade_from_060'
MAX_QUEUE_WAIT = dt.timedelta(hours=1)
MAX_COMBINE_SOURCES = 8  # combine_sources.position BETWEEN 0 AND 7


async def retired_slugs(session: AsyncSession, retired: Iterable[RetiredSlug]) -> int:
    """A deleted project keeps its slug, retired, so the slug is never reused."""
    rows = [
        {'slug': r.slug, 'created_at': r.created_at, 'retired_at': r.retired_at} for r in retired
    ]
    if rows:
        await session.execute(insert(ProjectSlug).values(rows))
    return len(rows)


async def set_projects_revision(session: AsyncSession, revision: int) -> None:
    """The deployment counter keeps the 0.6.0 value when that is ahead of the creates' bumps."""
    stmt = insert(DeploymentCounter).values(name='projects_revision', value=revision)
    await session.execute(
        stmt.on_conflict_do_update(
            index_elements=[DeploymentCounter.name],
            set_={'value': func.greatest(DeploymentCounter.value, stmt.excluded.value)},
        )
    )


def _job(row: JobRow, project_ids: Mapping[str, uuid.UUID]) -> JobLoad:
    return JobLoad(
        id=row.id,
        project_id=project_ids[row.project_slug] if row.project_slug else None,
        kind=row.kind,
        queue=row.queue,
        status=row.status,
        params=row.params,
        progress=row.progress,
        result=row.result,
        error_code=row.error_code,
        error_message=row.error_message,
        max_queue_wait=MAX_QUEUE_WAIT,
        requested_by=row.requested_by,
        created_at=row.created_at,
        started_at=row.started_at,
        finished_at=row.finished_at,
        updated_at=row.finished_at or row.created_at,
    )


async def jobs(
    session: AsyncSession, rows: Iterable[JobRow], project_ids: Mapping[str, uuid.UUID]
) -> int:
    loads = [_job(row, project_ids) for row in rows]
    return await session.run_sync(load_jobs, loads)


async def ingest_sources(
    session: AsyncSession, sources: Iterable[IngestSourceSrc], project_ids: Mapping[str, uuid.UUID]
) -> int:
    taken: set[tuple[str, str]] = set()
    n = 0
    for src_row in sources:
        name = src_row.name
        while (src_row.project_slug, name) in taken:  # the 0.6.0 name was not unique, 0.7.0's is
            name = f'{name} ({src_row.config.get("legacy_source_id")})'
        taken.add((src_row.project_slug, name))
        session.add(
            IngestSource(
                id=src_row.id,
                project_id=project_ids[src_row.project_slug],
                name=name,
                kind=src_row.kind,
                config=src_row.config,
                schedule=src_row.schedule,
                enabled=src_row.enabled,
                scan_state=src_row.scan_state,
                last_poll_at=src_row.last_poll_at,
                created_at=src_row.created_at,
                updated_at=src_row.updated_at,
            )
        )
        n += 1
    await session.flush()
    return n


async def dataset_imports(
    session: AsyncSession, imports: Sequence[ImportSrc], project_ids: Mapping[str, uuid.UUID]
) -> int:
    await jobs(session, [i.job for i in imports], project_ids)
    for i in imports:
        session.add(
            DatasetImport(
                id=i.id,
                project_id=project_ids[i.project_slug],
                job_id=i.job.id,
                source_key=i.source_key,
                format=i.format,
                mapping=i.mapping,
                options=i.options,
                scan=i.scan,
                report=i.report,
                created_at=i.created_at,
                undone_at=i.undone_at,
            )
        )
    await session.flush()
    return len(imports)


async def exports(
    session: AsyncSession, rows: Sequence[ExportSrc], project_ids: Mapping[str, uuid.UUID]
) -> int:
    for e in rows:
        session.add(
            DatasetExport(
                id=e.id,
                project_id=project_ids[e.project_slug],
                kind=e.kind,
                class_names=list(e.class_names),
                manifest_sha256=e.manifest_sha256,
                artifact_prefix=e.artifact_prefix,
                split_seed=e.split_seed,
                created_at=e.created_at,
            )
        )
    await session.flush()
    return len(rows)


async def training_runs(
    session: AsyncSession,
    runs: Sequence[TrainingRunSrc],
    project_ids: Mapping[str, uuid.UUID],
    export_ids: Mapping[tuple[str, str], uuid.UUID],
) -> int:
    await jobs(session, [r.job for r in runs], project_ids)
    for r in runs:
        export_dir = str(r.spec.get('dataset_export_dir') or '').rstrip('/').rsplit('/', 1)[-1]
        session.add(
            TrainingRun(
                id=r.id,
                project_id=project_ids[r.project_slug],
                job_id=r.job.id,
                dataset_export_id=export_ids.get((r.project_slug, export_dir)),
                spec=r.spec,
                class_names=list(r.class_names),
                family=r.family,
                base_model=r.base_model,
                artifact_prefix=r.artifact_prefix,
                manifest=r.manifest,
                mlflow_run_id=r.mlflow_run_id,
                created_at=r.created_at,
                finished_at=r.finished_at,
            )
        )
    await session.flush()
    return len(runs)


async def promotions(
    session: AsyncSession,
    rows: Sequence[PromotionSrc],
    project_ids: Mapping[str, uuid.UUID],
    run_ids: Mapping[tuple[str, str], uuid.UUID],
) -> int:
    """Models, the live promotion of each and the shares of a shared model."""
    for p in rows:
        pid = project_ids[p.project_slug]
        session.add(
            Model(
                id=p.model_id,
                project_id=pid,
                name=p.triton_name,
                version=p.version,
                origin='trained',
                training_run_id=run_ids.get((p.project_slug, p.training_run_legacy_id or '')),
                family=p.family,
                task='detect',
                labels=list(p.labels),
                artifact_storage_key=p.artifact_key,
                created_at=p.promoted_at,
                created_by=UPDATED_BY,
            )
        )
    await session.flush()
    for p in rows:
        session.add(
            Promotion(
                id=p.promotion_id,
                project_id=project_ids[p.project_slug],
                model_id=p.model_id,
                triton_model_name=p.triton_name,
                triton_version=p.triton_version,
                class_remap=p.class_remap,
                promoted_at=p.promoted_at,
                promoted_by=UPDATED_BY,
            )
        )
        if p.shared:  # 0.6.0 sharing was one switch for every project; 0.7.0 names its consumers
            for slug, other in project_ids.items():
                if slug != p.project_slug:
                    session.add(
                        ModelShare(
                            model_id=p.model_id,
                            consumer_project_id=other,
                            created_at=p.promoted_at,
                            created_by=UPDATED_BY,
                        )
                    )
    await session.flush()
    return len(rows)


async def combines(
    session: AsyncSession, rows: Sequence[CombineSrc], project_ids: Mapping[str, uuid.UUID]
) -> int:
    await jobs(session, [c.job for c in rows], project_ids)
    for c in rows:
        session.add(
            CombineRun(
                job_id=c.job.id,
                target_project_id=project_ids[c.target_slug],
                holdout_mode=c.holdout_mode,
                class_mapping=c.class_mapping,
                created_at=c.created_at,
            )
        )
    await session.flush()
    for c in rows:
        for position, slug in enumerate(c.source_slugs[:MAX_COMBINE_SOURCES]):
            session.add(
                CombineSource(
                    job_id=c.job.id,
                    position=position,
                    source_project_id=project_ids.get(slug),
                    source_slug=slug,
                )
            )
    await session.flush()
    return len(rows)


async def local_vlm_desired(session: AsyncSession, global_data: GlobalData) -> bool:
    if not global_data.local_vlm_desired:
        return False
    session.add(
        DeploymentSetting(
            key='local_vlm_desired', value=global_data.local_vlm_desired, updated_by=UPDATED_BY
        )
    )
    await session.flush()
    return True


async def projection_state(
    session: AsyncSession, project_id: uuid.UUID, state: ViewStateSrc, reducer_key: str | None
) -> None:
    session.add(
        ProjectionState(
            project_id=project_id,
            projection_version=state.projection_version,
            scope=state.scope,
            cluster_uid=None,  # a 0.6.0 integer cluster id names no served cluster
            n_points=state.n_points,
            n_components=state.n_components,
            metric=state.metric,
            fitted_at=state.fitted_at,
            reducer_storage_key=reducer_key,
        )
    )
    await session.flush()


async def id_maps(
    session: AsyncSession,
) -> tuple[dict[str, uuid.UUID], dict[tuple[str | None, str, str], uuid.UUID]]:
    """``(slug -> project id, (slug or None, kind, name) -> config id)`` read back from the
    database, for the steps that run after the load committed (verify, a resumed run)."""
    projects = dict((await session.execute(select(Project.slug, Project.id))).all())
    owner = {pid: slug for slug, pid in projects.items()}
    rows = await session.execute(
        select(
            ConfigDocument.project_id, ConfigDocument.kind, ConfigDocument.name, ConfigDocument.id
        )
    )
    return projects, {
        (owner[pid] if pid else None, kind, name): cid for pid, kind, name, cid in rows
    }
