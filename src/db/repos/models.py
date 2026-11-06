"""Models, promotions, shares and engines: the model registry in PostgreSQL.

Every function takes the caller's session inside the caller's ``project_tx`` or ``deployment_tx``
and never commits. A promote is one transaction: the previous live promotion of the Triton model
name gets ``superseded_at`` and the new one is inserted, so the partial unique index
``uq_promotions_live`` never sees two live rows. Two promotes of one name serialise on a
transaction-scoped advisory lock keyed by the name
(https://www.postgresql.org/docs/current/functions-admin.html#FUNCTIONS-ADVISORY-LOCKS), so the
second finds the first committed and supersedes it instead of failing on the index.

A share is a row per consumer project (``model_shares``). The sharing wire keeps one boolean:
"shared" means a row for every consumer the caller names, and the sharing revision the wire
carries is derived from the rows (1 private, 2 shared), so a stale write is still told.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from sqlalchemy import delete, func, select, text, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import IntegrityError

from src.db.models.jobs import Job
from src.db.models.projects import Project
from src.db.models.training import Model, ModelShare, Promotion


if TYPE_CHECKING:
    import datetime as dt
    import uuid
    from collections.abc import Collection, Sequence

    from sqlalchemy import Select
    from sqlalchemy.ext.asyncio import AsyncSession


@dataclass(frozen=True)
class ModelRow:
    id: uuid.UUID
    project_id: uuid.UUID | None
    name: str
    version: int
    origin: str
    training_run_id: uuid.UUID | None
    family: str
    task: str
    labels: tuple[str, ...]
    artifact_storage_key: str
    created_at: dt.datetime
    created_by: str


@dataclass(frozen=True)
class NewModel:
    """What a promote decides; the database assigns the id and the version number."""

    project_id: uuid.UUID | None
    name: str
    origin: str
    family: str
    task: str
    labels: Sequence[str]
    artifact_storage_key: str
    created_by: str
    training_run_id: uuid.UUID | None = None


@dataclass(frozen=True)
class PromotionRow:
    id: uuid.UUID
    project_id: uuid.UUID
    model_id: uuid.UUID
    triton_model_name: str
    triton_version: int
    class_remap: Any | None
    job_id: uuid.UUID | None
    promoted_at: dt.datetime
    promoted_by: str
    superseded_at: dt.datetime | None


@dataclass(frozen=True)
class LiveModel:
    """A live promotion with its model and the owner project's slug."""

    promotion: PromotionRow
    model: ModelRow
    owner_slug: str
    run_job_id: str | None


class LivePromotionConflictError(RuntimeError):
    """Another project holds the live promotion of this Triton model name."""

    def __init__(self, triton_model_name: str) -> None:
        super().__init__(f'a live promotion of {triton_model_name!r} already exists')
        self.triton_model_name = triton_model_name


def _model_row(row: Model) -> ModelRow:
    return ModelRow(
        id=row.id,
        project_id=row.project_id,
        name=row.name,
        version=row.version,
        origin=row.origin,
        training_run_id=row.training_run_id,
        family=row.family,
        task=row.task,
        labels=tuple(row.labels),
        artifact_storage_key=row.artifact_storage_key,
        created_at=row.created_at,
        created_by=row.created_by,
    )


def _promotion_row(row: Promotion) -> PromotionRow:
    return PromotionRow(
        id=row.id,
        project_id=row.project_id,
        model_id=row.model_id,
        triton_model_name=row.triton_model_name,
        triton_version=row.triton_version,
        class_remap=row.class_remap,
        job_id=row.job_id,
        promoted_at=row.promoted_at,
        promoted_by=row.promoted_by,
        superseded_at=row.superseded_at,
    )


async def _lock_name(session: AsyncSession, triton_model_name: str) -> None:
    await session.execute(
        text('SELECT pg_advisory_xact_lock(hashtextextended(:name, 0))'),
        {'name': f'promote:{triton_model_name}'},
    )


async def next_version(session: AsyncSession, project_id: uuid.UUID | None, name: str) -> int:
    stmt = select(func.coalesce(func.max(Model.version), 0)).where(Model.name == name)
    stmt = stmt.where(
        Model.project_id.is_(None) if project_id is None else Model.project_id == project_id
    )
    return int((await session.execute(stmt)).scalar_one()) + 1


async def insert_model(session: AsyncSession, new: NewModel) -> ModelRow:
    """Insert the model under the next version number of ``(project, name)``."""
    row = Model(
        project_id=new.project_id,
        name=new.name,
        version=await next_version(session, new.project_id, new.name),
        origin=new.origin,
        training_run_id=new.training_run_id,
        family=new.family,
        task=new.task,
        labels=list(new.labels),
        artifact_storage_key=new.artifact_storage_key,
        created_by=new.created_by,
    )
    session.add(row)
    await session.flush()
    await session.refresh(row)
    return _model_row(row)


async def record_promotion(
    session: AsyncSession,
    *,
    model: NewModel,
    triton_model_name: str,
    triton_version: int,
    class_remap: Any | None,
    job_id: uuid.UUID | None,
    promoted_by: str,
) -> tuple[ModelRow, PromotionRow]:
    """Insert the model and its promotion, superseding the name's live promotion.

    ``model.project_id`` is the owner and is required. The previous live row (if any) keeps its
    model; only ``superseded_at`` moves.
    """
    if model.project_id is None:
        raise ValueError('a promotion belongs to a project')
    await _lock_name(session, triton_model_name)
    await session.execute(
        update(Promotion)
        .where(Promotion.triton_model_name == triton_model_name, Promotion.superseded_at.is_(None))
        .values(superseded_at=func.now())
    )
    inserted = await insert_model(session, model)
    row = Promotion(
        project_id=model.project_id,
        model_id=inserted.id,
        triton_model_name=triton_model_name,
        triton_version=triton_version,
        class_remap=class_remap,
        job_id=job_id,
        promoted_by=promoted_by,
    )
    session.add(row)
    try:
        await session.flush()
    except IntegrityError as exc:
        # Another project's live promotion holds the name (the index spans projects, and this
        # transaction cannot see that row): fail closed rather than serve a name twice.
        raise LivePromotionConflictError(triton_model_name) from exc
    await session.refresh(row)
    return inserted, _promotion_row(row)


async def withdraw_promotion(session: AsyncSession, promotion_id: uuid.UUID) -> None:
    """Undo a promote that did not serve: drop its rows and make the previous one live again."""
    found = (
        await session.execute(select(Promotion).where(Promotion.id == promotion_id))
    ).scalar_one_or_none()
    if found is None:
        return
    await _lock_name(session, found.triton_model_name)
    previous = (
        await session.execute(
            select(Promotion.id)
            .where(
                Promotion.triton_model_name == found.triton_model_name,
                Promotion.id != found.id,
                Promotion.superseded_at.is_not(None),
            )
            .order_by(Promotion.promoted_at.desc())
            .limit(1)
        )
    ).scalar_one_or_none()
    await session.execute(delete(Promotion).where(Promotion.id == found.id))
    await session.execute(delete(Model).where(Model.id == found.model_id))
    if previous is not None:
        await session.execute(
            update(Promotion).where(Promotion.id == previous).values(superseded_at=None)
        )


async def latest_promote_job_id(session: AsyncSession, run_job_id: str) -> uuid.UUID | None:
    """The newest ``promote`` job of the training run, in the caller's project scope."""
    stmt = (
        select(Job.id)
        .where(Job.kind == 'promote', Job.params['run_job_id'].astext == run_job_id)
        .order_by(Job.created_at.desc(), Job.id.desc())
        .limit(1)
    )
    return (await session.execute(stmt)).scalar_one_or_none()


def _live_select() -> Select[Any]:
    return (
        select(Promotion, Model, Project.slug, Job.params['run_job_id'].astext)
        .join(Model, Model.id == Promotion.model_id)
        .join(Project, Project.id == Promotion.project_id)
        .outerjoin(Job, Job.id == Promotion.job_id)
        .where(Promotion.superseded_at.is_(None))
    )


def _live(found: Any) -> LiveModel:
    return LiveModel(_promotion_row(found[0]), _model_row(found[1]), found[2], found[3])


async def get_live(session: AsyncSession, triton_model_name: str) -> LiveModel | None:
    """The live promotion of ``triton_model_name`` with its model and owner, or ``None``."""
    found = (
        await session.execute(
            _live_select().where(Promotion.triton_model_name == triton_model_name)
        )
    ).first()
    return None if found is None else _live(found)


async def list_live(
    session: AsyncSession, *, owner_project_id: uuid.UUID | None = None
) -> list[LiveModel]:
    """Every live promotion (of one owner when given), by Triton name."""
    stmt = _live_select().order_by(Promotion.triton_model_name)
    if owner_project_id is not None:
        stmt = stmt.where(Promotion.project_id == owner_project_id)
    return [_live(found) for found in (await session.execute(stmt)).all()]


async def list_live_shared_with(
    session: AsyncSession, consumer_project_id: uuid.UUID
) -> list[LiveModel]:
    """Live promotions of other projects that are shared with ``consumer_project_id``."""
    stmt = (
        _live_select()
        .join(ModelShare, ModelShare.model_id == Model.id)
        .where(ModelShare.consumer_project_id == consumer_project_id)
        .order_by(Promotion.triton_model_name)
    )
    return [_live(found) for found in (await session.execute(stmt)).all()]


# --- sharing ---------------------------------------------------------------------------------


async def share_consumers(session: AsyncSession, model_id: uuid.UUID) -> list[uuid.UUID]:
    stmt = select(ModelShare.consumer_project_id).where(ModelShare.model_id == model_id)
    return list((await session.execute(stmt.order_by(ModelShare.consumer_project_id))).scalars())


def sharing_revision(consumer_count: int) -> int:
    """The sharing state as the wire's revision: 1 while the model is private, 2 once shared.

    It is a function of the rows, so a writer holding the revision it read can only be refused
    when the state moved under it (the unshare it meant to undo already happened).
    """
    return 2 if consumer_count else 1


async def lock_model(session: AsyncSession, model_id: uuid.UUID) -> None:
    """Serialise sharing writers of one model (``FOR UPDATE`` on its row)."""
    await session.execute(select(Model.id).where(Model.id == model_id).with_for_update())


async def replace_shares(
    session: AsyncSession,
    model_id: uuid.UUID,
    consumers: Collection[uuid.UUID],
    created_by: str,
) -> None:
    """Make ``consumers`` exactly the model's consumer projects."""
    stale = delete(ModelShare).where(ModelShare.model_id == model_id)
    if consumers:
        stale = stale.where(ModelShare.consumer_project_id.not_in(list(consumers)))
    await session.execute(stale)
    if consumers:
        await session.execute(
            pg_insert(ModelShare)
            .values(
                [
                    {'model_id': model_id, 'consumer_project_id': c, 'created_by': created_by}
                    for c in sorted(consumers)
                ]
            )
            .on_conflict_do_nothing()
        )


async def supersede_live(session: AsyncSession, triton_model_name: str) -> bool:
    """End the name's live promotion (the model was unloaded and deleted); ``True`` if one ended."""
    await _lock_name(session, triton_model_name)
    done = await session.execute(
        update(Promotion)
        .where(Promotion.triton_model_name == triton_model_name, Promotion.superseded_at.is_(None))
        .values(superseded_at=func.now())
        .returning(Promotion.id)
    )
    return done.first() is not None
