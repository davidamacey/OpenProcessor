"""Read models of the space routes: the catalog with availability, spaces with their model, runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.db.repos import embedding_spaces as repo
from src.db.session import deployment_tx, project_tx
from src.services.embedding_spaces import reindex_runs
from src.services.embedding_spaces.catalog import Availability, availability


if TYPE_CHECKING:
    import uuid
    from collections.abc import Sequence
    from typing import Any

    from sqlalchemy.ext.asyncio import AsyncSession


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


async def run_view(
    session: AsyncSession, project_id: uuid.UUID, run_id: uuid.UUID
) -> dict[str, Any] | None:
    async with project_tx(session, project_id):
        return await reindex_runs.get_run_view(session, run_id)
