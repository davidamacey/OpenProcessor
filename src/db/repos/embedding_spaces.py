"""Embedding registry rows: the deployment model catalog and the per-project spaces.

Every function takes a session the caller already scoped (``project_tx`` for spaces,
``deployment_tx`` for the catalog) and never commits. The catalog functions take a sync
``Session`` because ``db-migrate`` is synchronous; an async caller reaches them through
``AsyncSession.run_sync``. The space functions are async.

PostgreSQL partial unique indexes (the one-primary rule) are enforced by the database, not here:
https://www.postgresql.org/docs/current/indexes-partial.html
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from sqlalchemy import func, select, update

from src.db.models.embeddings import EmbeddingEvaluationRun, EmbeddingModelVersion, EmbeddingSpace
from src.db.models.projects import Project
from src.db.models.training import ModelEngine
from src.domain.embedding_spaces import LIVE_STATUSES, KnnMethodSpec, Role, Status, Target


if TYPE_CHECKING:
    import datetime as dt
    import uuid
    from collections.abc import Sequence

    from sqlalchemy.ext.asyncio import AsyncSession
    from sqlalchemy.orm import Session


@dataclass(frozen=True)
class ModelVersionRow:
    id: uuid.UUID
    model_key: str
    version: int
    modalities: tuple[str, ...]
    native_dim: int
    matryoshka_dims: tuple[int, ...]
    distance: str
    input_spec: dict[str, Any]
    input_spec_sha256: str
    weights_ref: str
    weights_sha256: str
    precision: str
    triton_image_model: str | None
    triton_text_model: str | None
    engine_parity_min_cos: float | None
    licence: str
    status: str


@dataclass(frozen=True)
class NewModelVersion:
    """Every column of a catalog row that the writer supplies (the id is generated)."""

    model_key: str
    version: int
    modalities: tuple[str, ...]
    native_dim: int
    matryoshka_dims: tuple[int, ...]
    distance: str
    input_spec: dict[str, Any]
    input_spec_sha256: str
    weights_ref: str
    weights_sha256: str
    precision: str
    triton_image_model: str | None
    triton_text_model: str | None
    licence: str
    status: str
    created_by: str


@dataclass(frozen=True)
class SpaceRow:
    id: uuid.UUID
    project_id: uuid.UUID
    space_key: str
    model_version_id: uuid.UUID
    target: Target
    dim: int
    role: Role
    status: Status
    index_role: str
    vector_field: str
    state_field: str | None
    nested_path: str | None
    knn_method: KnnMethodSpec
    revision: int
    built_by_job_id: uuid.UUID | None
    ready_at: dt.datetime | None
    retired_at: dt.datetime | None
    created_at: dt.datetime
    created_by: str


@dataclass(frozen=True)
class NewSpace:
    project_id: uuid.UUID
    space_key: str
    model_version_id: uuid.UUID
    target: Target
    dim: int
    role: Role
    status: Status
    index_role: str
    vector_field: str
    state_field: str | None
    nested_path: str | None
    knn_method: KnnMethodSpec
    created_by: str


def _version_row(m: EmbeddingModelVersion) -> ModelVersionRow:
    return ModelVersionRow(
        id=m.id,
        model_key=m.model_key,
        version=m.version,
        modalities=tuple(m.modalities),
        native_dim=m.native_dim,
        matryoshka_dims=tuple(m.matryoshka_dims),
        distance=m.distance,
        input_spec=m.input_spec,
        input_spec_sha256=m.input_spec_sha256,
        weights_ref=m.weights_ref,
        weights_sha256=m.weights_sha256,
        precision=m.precision,
        triton_image_model=m.triton_image_model,
        triton_text_model=m.triton_text_model,
        engine_parity_min_cos=m.engine_parity_min_cos,
        licence=m.licence,
        status=m.status,
    )


def _space_row(s: EmbeddingSpace) -> SpaceRow:
    return SpaceRow(
        id=s.id,
        project_id=s.project_id,
        space_key=s.space_key,
        model_version_id=s.model_version_id,
        target=Target(s.target),
        dim=s.dim,
        role=Role(s.role),
        status=Status(s.status),
        index_role=s.index_role,
        vector_field=s.vector_field,
        state_field=s.state_field,
        nested_path=s.nested_path,
        knn_method=KnnMethodSpec.model_validate(s.knn_method),
        revision=s.revision,
        built_by_job_id=s.built_by_job_id,
        ready_at=s.ready_at,
        retired_at=s.retired_at,
        created_at=s.created_at,
        created_by=s.created_by,
    )


# --- the deployment catalog (sync: db-migrate is synchronous) -----------------------------------


def get_model_version(session: Session, model_key: str, version: int) -> ModelVersionRow | None:
    row = session.scalars(
        select(EmbeddingModelVersion).where(
            EmbeddingModelVersion.model_key == model_key, EmbeddingModelVersion.version == version
        )
    ).one_or_none()
    return None if row is None else _version_row(row)


def insert_model_version(session: Session, new: NewModelVersion) -> ModelVersionRow:
    row = EmbeddingModelVersion(
        model_key=new.model_key,
        version=new.version,
        modalities=list(new.modalities),
        native_dim=new.native_dim,
        matryoshka_dims=list(new.matryoshka_dims),
        distance=new.distance,
        input_spec=new.input_spec,
        input_spec_sha256=new.input_spec_sha256,
        weights_ref=new.weights_ref,
        weights_sha256=new.weights_sha256,
        precision=new.precision,
        triton_image_model=new.triton_image_model,
        triton_text_model=new.triton_text_model,
        licence=new.licence,
        status=new.status,
        created_by=new.created_by,
    )
    session.add(row)
    session.flush()
    return _version_row(row)


def set_model_version_status(session: Session, version_id: uuid.UUID, status: str) -> None:
    session.execute(
        update(EmbeddingModelVersion)
        .where(EmbeddingModelVersion.id == version_id)
        .values(status=status)
    )


# --- spaces (async, inside the caller's project_tx) ----------------------------------------------


async def get_model_version_by_id(
    session: AsyncSession, version_id: uuid.UUID
) -> ModelVersionRow | None:
    row = await session.get(EmbeddingModelVersion, version_id)
    return None if row is None else _version_row(row)


async def add_space(session: AsyncSession, new: NewSpace) -> SpaceRow:
    row = EmbeddingSpace(
        project_id=new.project_id,
        space_key=new.space_key,
        model_version_id=new.model_version_id,
        target=new.target.value,
        dim=new.dim,
        role=new.role.value,
        status=new.status.value,
        index_role=new.index_role,
        vector_field=new.vector_field,
        state_field=new.state_field,
        nested_path=new.nested_path,
        knn_method=new.knn_method.model_dump(mode='json', exclude_none=True),
        created_by=new.created_by,
    )
    session.add(row)
    await session.flush()
    await session.refresh(row)
    return _space_row(row)


async def get_space(
    session: AsyncSession, project_id: uuid.UUID, target: Target, space_key: str
) -> SpaceRow | None:
    row = (
        await session.scalars(
            select(EmbeddingSpace).where(
                EmbeddingSpace.project_id == project_id,
                EmbeddingSpace.target == target.value,
                EmbeddingSpace.space_key == space_key,
            )
        )
    ).one_or_none()
    return None if row is None else _space_row(row)


async def get_primary_space(
    session: AsyncSession, project_id: uuid.UUID, target: Target
) -> SpaceRow | None:
    row = (
        await session.scalars(
            select(EmbeddingSpace).where(
                EmbeddingSpace.project_id == project_id,
                EmbeddingSpace.target == target.value,
                EmbeddingSpace.role == Role.PRIMARY.value,
            )
        )
    ).one_or_none()
    return None if row is None else _space_row(row)


async def list_spaces(
    session: AsyncSession,
    project_id: uuid.UUID,
    *,
    target: Target | None = None,
    live_only: bool = False,
) -> list[SpaceRow]:
    stmt = select(EmbeddingSpace).where(EmbeddingSpace.project_id == project_id)
    if target is not None:
        stmt = stmt.where(EmbeddingSpace.target == target.value)
    if live_only:
        stmt = stmt.where(EmbeddingSpace.status.in_([s.value for s in LIVE_STATUSES]))
    stmt = stmt.order_by(EmbeddingSpace.target, EmbeddingSpace.created_at, EmbeddingSpace.space_key)
    return [_space_row(r) for r in await session.scalars(stmt)]


async def count_live_spaces(session: AsyncSession, project_id: uuid.UUID, target: Target) -> int:
    return int(
        await session.scalar(
            select(func.count())
            .select_from(EmbeddingSpace)
            .where(
                EmbeddingSpace.project_id == project_id,
                EmbeddingSpace.target == target.value,
                EmbeddingSpace.status.in_([s.value for s in LIVE_STATUSES]),
            )
        )
        or 0
    )


async def move_status(
    session: AsyncSession,
    space_id: uuid.UUID,
    *,
    expected: Status,
    to: Status,
    job_id: uuid.UUID | None = None,
) -> SpaceRow | None:
    """Compare-and-set the status; ``None`` when the space is not in ``expected`` any more.

    ``ready_at`` and ``retired_at`` are stamped by the move into those states, and every move
    bumps ``revision`` so a concurrent flip that read the old row fails its own fence.
    """
    values: dict[str, Any] = {'status': to.value, 'revision': EmbeddingSpace.revision + 1}
    if to is Status.READY:
        values['ready_at'] = func.now()
    if to is Status.RETIRED:
        values['retired_at'] = func.now()
    if job_id is not None:
        values['built_by_job_id'] = job_id
    row = (
        await session.scalars(
            update(EmbeddingSpace)
            .where(EmbeddingSpace.id == space_id, EmbeddingSpace.status == expected.value)
            .values(**values)
            .returning(EmbeddingSpace)
            .execution_options(populate_existing=True)
        )
    ).one_or_none()
    return None if row is None else _space_row(row)


async def set_role(
    session: AsyncSession, space_id: uuid.UUID, *, expected_revision: int, role: Role
) -> SpaceRow | None:
    """Set the role if the row is still at ``expected_revision`` (the flip's fence)."""
    row = (
        await session.scalars(
            update(EmbeddingSpace)
            .where(EmbeddingSpace.id == space_id, EmbeddingSpace.revision == expected_revision)
            .values(role=role.value, revision=EmbeddingSpace.revision + 1)
            .returning(EmbeddingSpace)
            .execution_options(populate_existing=True)
        )
    ).one_or_none()
    return None if row is None else _space_row(row)


async def has_evaluation(
    session: AsyncSession, project_id: uuid.UUID, baseline_id: uuid.UUID, candidate_id: uuid.UUID
) -> bool:
    found = await session.scalar(
        select(EmbeddingEvaluationRun.id)
        .where(
            EmbeddingEvaluationRun.project_id == project_id,
            EmbeddingEvaluationRun.baseline_space_id == baseline_id,
            EmbeddingEvaluationRun.candidate_space_id == candidate_id,
        )
        .limit(1)
    )
    return found is not None


# --- the project's configuration epoch (the projects row is owned by the projects repository) -----


async def bump_config_epoch(session: AsyncSession, project_id: uuid.UUID) -> int:
    """Increment ``projects.config_epoch`` and return the new value (every worker re-reads on it)."""
    epoch = await session.scalar(
        update(Project)
        .where(Project.id == project_id)
        .values(config_epoch=Project.config_epoch + 1)
        .returning(Project.config_epoch)
    )
    if epoch is None:
        msg = f'no project {project_id}'
        raise LookupError(msg)
    return int(epoch)


async def add_spaces(session: AsyncSession, rows: Sequence[NewSpace]) -> list[SpaceRow]:
    """Insert several spaces in order (the project create writes its three defaults this way)."""
    return [await add_space(session, r) for r in rows]


def ready_engine_contracts(session: Session) -> dict[str, dict[str, Any]]:
    """``triton_model_name -> input_contract`` of every ready engine (the installed inventory)."""
    rows = session.execute(
        select(ModelEngine.triton_model_name, ModelEngine.input_contract).where(
            ModelEngine.status == 'ready'
        )
    )
    # not ``dict(rows)``: a Result has ``keys()``, so ``dict`` would read it as a mapping
    return {row[0]: row[1] for row in rows}
