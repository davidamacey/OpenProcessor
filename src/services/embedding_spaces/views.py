"""Read models of the space routes: the catalog with availability, spaces with their model, runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from opensearchpy.exceptions import OpenSearchException

from src.core.logging import get_logger
from src.db.repos import embedding_spaces as repo
from src.db.session import deployment_tx, project_tx
from src.domain.embedding_spaces import Status
from src.services.embedding_spaces import reindex_runs
from src.services.embedding_spaces.catalog import Availability, availability


if TYPE_CHECKING:
    import datetime as dt
    import uuid
    from collections.abc import Sequence
    from typing import Any

    from sqlalchemy.ext.asyncio import AsyncSession

    from src.domain.embedding_spaces import Target
    from src.services.embedding_spaces.reindex import ReindexIndex
    from src.services.embedding_spaces.reindex_runs import LatestRun

logger = get_logger(__name__)


@dataclass(frozen=True)
class CatalogEntry:
    model: repo.ModelVersionRow
    availability: Availability


@dataclass(frozen=True)
class SpaceWithModel:
    space: repo.SpaceRow
    model: repo.ModelVersionRow


async def catalog_entries(session: AsyncSession) -> list[CatalogEntry]:
    async with deployment_tx(session, 'the embedding model catalog is deployment-wide'):
        versions = await session.run_sync(repo.list_model_versions)
        engines = await session.run_sync(repo.ready_engine_contracts)
    return [CatalogEntry(v, availability(v, engines)) for v in versions]


async def with_models(
    session: AsyncSession, project_id: uuid.UUID, rows: Sequence[repo.SpaceRow]
) -> list[SpaceWithModel]:
    found: list[SpaceWithModel] = []
    async with project_tx(session, project_id):
        for row in rows:
            model = await repo.get_model_version_by_id(session, row.model_version_id)
            if model is not None:
                found.append(SpaceWithModel(row, model))
    return found


async def docs_with_vector(index: ReindexIndex, space: repo.SpaceRow) -> int | None:
    """Documents holding a vector in ``space``; ``None`` for a space with no field yet, and when
    the index cannot answer (the list must still render; the failure is logged)."""
    if space.status not in (Status.BUILDING, Status.READY, Status.RETIRING):
        return None
    try:
        return await index.count_with_vector(space)
    except OpenSearchException as exc:
        logger.warning('embedding_space_count_failed', space=space.space_key, error=str(exc))
        return None


async def latest_runs(
    session: AsyncSession, project_id: uuid.UUID, rows: Sequence[repo.SpaceRow]
) -> dict[uuid.UUID, LatestRun]:
    """The newest run of each of ``rows`` that has one, by space id."""
    async with project_tx(session, project_id):
        return await reindex_runs.latest_runs_by_space(session, [r.id for r in rows])


@dataclass(frozen=True)
class EvaluationRow:
    """One evaluation run of a candidate space against a baseline, with the space keys named."""

    id: uuid.UUID
    baseline_space_key: str
    candidate_space_key: str
    protocol: str
    verdict: str
    metrics: dict[str, Any]
    sample_spec: dict[str, Any]
    job_id: uuid.UUID | None
    created_at: dt.datetime
    created_by: str
    against_primary: bool


@dataclass(frozen=True)
class SpaceEvaluations:
    space: repo.SpaceRow
    primary_space_key: str | None
    evaluations: list[EvaluationRow]


async def space_evaluations(
    session: AsyncSession, project_id: uuid.UUID, target: Target, key: str
) -> SpaceEvaluations | None:
    """The evaluation runs of one space (as the candidate), or ``None`` for an unknown space.

    ``against_primary`` is what a promote asks: a run whose baseline is the current primary.
    """
    async with project_tx(session, project_id):
        space = await repo.get_space(session, project_id, target, key)
        if space is None:
            return None
        primary = await repo.get_primary_space(session, project_id, target)
        keys = {s.id: s.space_key for s in await repo.list_spaces(session, project_id)}
        runs = await repo.list_evaluations(session, project_id, space.id)
    return SpaceEvaluations(
        space=space,
        primary_space_key=None if primary is None else primary.space_key,
        evaluations=[
            EvaluationRow(
                id=r.id,
                baseline_space_key=keys[r.baseline_space_id],
                candidate_space_key=space.space_key,
                protocol=r.protocol,
                verdict=r.verdict,
                metrics=r.metrics,
                sample_spec=r.sample_spec,
                job_id=r.job_id,
                created_at=r.created_at,
                created_by=r.created_by,
                against_primary=primary is not None and r.baseline_space_id == primary.id,
            )
            for r in runs
        ],
    )


async def run_view(
    session: AsyncSession, project_id: uuid.UUID, run_id: uuid.UUID
) -> dict[str, Any] | None:
    async with project_tx(session, project_id):
        return await reindex_runs.get_run_view(session, run_id)
