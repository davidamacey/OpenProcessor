"""Config store repository: documents, immutable revisions, activations, probes, runtime reports.

Every function takes a session that is already inside ``project_tx`` or ``deployment_tx`` and
never commits. A deployment-wide document (``project_id`` ``None``: a VLM endpoint, a serving
profile) is visible only to a deployment-scoped transaction, so the caller opens one for those.

Concurrency follows the PostgreSQL documentation for READ COMMITTED
(https://www.postgresql.org/docs/current/transaction-iso.html#XACT-READ-COMMITTED): an
``UPDATE ... WHERE current_revision = :expected`` blocks on a row a concurrent writer holds and
re-evaluates its predicate after that writer commits, so exactly one of two saves that name the
same revision matches. The first save and the first activation use ``INSERT ... ON CONFLICT DO
NOTHING`` (https://www.postgresql.org/docs/current/sql-insert.html#SQL-ON-CONFLICT) and read the
zero-row result as the conflict.

An explicit deactivation ("off") is not a row in ``config_activations`` (its ``config_id`` is
not nullable). It is an event with no target (both ``to_*`` columns NULL): the axis then reads as
``off`` until the next activation, and "never activated" (no event) stays distinct from "off".
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from sqlalchemy import String, case, cast, delete, func, insert, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import aliased

from src.db.models.config import (
    ConfigDocument,
    ConfigRevision,
    ConfigRuntimeReport,
    VlmEndpointProbe,
)
from src.db.models.projects import DeploymentSetting, Project
from src.db.repos.config_activations import (
    activate,
    get_activation,
    get_activations,
    list_activation_events,
    load_activation_events,
)
from src.db.repos.config_base import (
    AXIS_KIND,
    DEFAULT_ACTOR,
    GLOBAL_EPOCH_COUNTER,
    ActivationEvent,
    ActivationEventLoad,
    ActiveConflictError,
    ActiveRef,
    AxisKindError,
    ConfigDoc,
    ConfigDocumentLoad,
    ConfigInUseError,
    ConfigRevisionLoad,
    ConfigScopeError,
    Probe,
    ProbeLoad,
    RevisionConflictError,
    RevisionInfo,
    RuntimeReport,
    VlmEndpointLoad,
    body_sha256,
    bump_epoch,
    check_scope,
    get_epoch,
)


if TYPE_CHECKING:
    import uuid

    from sqlalchemy.ext.asyncio import AsyncSession

__all__ = [
    'AXIS_KIND',
    'DEFAULT_ACTOR',
    'GLOBAL_EPOCH_COUNTER',
    'ActivationEvent',
    'ActivationEventLoad',
    'ActiveConflictError',
    'ActiveRef',
    'AxisKindError',
    'ConfigDoc',
    'ConfigDocumentLoad',
    'ConfigInUseError',
    'ConfigRevisionLoad',
    'ConfigScopeError',
    'DeploymentSettingRow',
    'Probe',
    'ProbeLoad',
    'RevisionConflictError',
    'RevisionInfo',
    'RuntimeReport',
    'VlmEndpointLoad',
    'activate',
    'body_sha256',
    'bump_epoch',
    'check_scope',
    'delete_config',
    'ensure_config',
    'get_activation',
    'get_activations',
    'get_config',
    'get_config_by_id',
    'get_deployment_setting',
    'get_epoch',
    'list_activation_events',
    'list_configs',
    'list_probes',
    'list_revisions',
    'list_runtime_reports',
    'load_activation_events',
    'load_config_document',
    'load_vlm_endpoint',
    'put_deployment_setting',
    'record_probe',
    'rollback',
    'save_config',
    'upsert_runtime_report',
]


_SOURCE = aliased(ConfigDocument)
_SOURCE_PROJECT = aliased(Project)


def _doc_select() -> Any:
    """Documents joined to one of their revisions (the caller adds the revision predicate) and,
    for a clone, to its source document for the ``cloned_from`` tag."""
    ref = cast(ConfigDocument.cloned_from_revision, String)
    tag = case(
        (_SOURCE.id.is_(None), None),
        (_SOURCE.project_id.is_(None), _SOURCE.name + '@' + ref),
        else_=_SOURCE_PROJECT.slug + ':' + _SOURCE.name + '@' + ref,
    )
    return (
        select(
            ConfigDocument.id,
            ConfigDocument.project_id,
            ConfigDocument.kind,
            ConfigDocument.name,
            ConfigRevision.revision,
            ConfigRevision.body,
            ConfigRevision.description,
            ConfigDocument.created_at,
            ConfigRevision.created_at,
            ConfigRevision.created_by,
            ConfigDocument.cloned_from_id,
            ConfigDocument.cloned_from_revision,
            ConfigDocument.current_revision,
            tag,
        )
        .select_from(ConfigDocument)
        .join(ConfigRevision, ConfigRevision.config_id == ConfigDocument.id)
        .outerjoin(_SOURCE, _SOURCE.id == ConfigDocument.cloned_from_id)
        .outerjoin(_SOURCE_PROJECT, _SOURCE_PROJECT.id == _SOURCE.project_id)
    )


def _to_doc(row: Any) -> ConfigDoc:
    return ConfigDoc(*row)


def _scope_filter(project_id: uuid.UUID | None) -> Any:
    return (
        ConfigDocument.project_id.is_(None)
        if project_id is None
        else ConfigDocument.project_id == project_id
    )


async def _current_revision(
    session: AsyncSession, project_id: uuid.UUID | None, kind: str, name: str
) -> int:
    value = (
        await session.execute(
            select(ConfigDocument.current_revision).where(
                _scope_filter(project_id), ConfigDocument.kind == kind, ConfigDocument.name == name
            )
        )
    ).scalar_one_or_none()
    return int(value or 0)


async def save_config(
    session: AsyncSession,
    *,
    project_id: uuid.UUID | None,
    kind: str,
    name: str,
    body: dict[str, Any],
    expected_revision: int | None,
    description: str = '',
    cloned_from_id: uuid.UUID | None = None,
    cloned_from_revision: int | None = None,
    by: str = DEFAULT_ACTOR,
) -> ConfigDoc:
    """Create (``expected_revision`` ``None``) or save a new revision of a document.

    Raises :class:`RevisionConflictError` when the name exists on a create or the revision is
    not the current one on a save. Bumps the epoch the readers poll.
    """
    check_scope(kind, project_id)
    if expected_revision is None:
        created = await session.execute(
            pg_insert(ConfigDocument)
            .values(
                project_id=project_id,
                kind=kind,
                name=name,
                current_revision=1,
                description=description,
                cloned_from_id=cloned_from_id,
                cloned_from_revision=cloned_from_revision,
                updated_by=by,
            )
            .on_conflict_do_nothing(constraint='uq_config_documents_project_id_kind_name')
            .returning(ConfigDocument.id)
        )
        config_id = created.scalar_one_or_none()
        if config_id is None:
            raise RevisionConflictError(await _current_revision(session, project_id, kind, name))
        revision = 1
    else:
        updated = await session.execute(
            update(ConfigDocument)
            .where(
                _scope_filter(project_id),
                ConfigDocument.kind == kind,
                ConfigDocument.name == name,
                ConfigDocument.current_revision == expected_revision,
            )
            .values(
                current_revision=ConfigDocument.current_revision + 1,
                description=description,
                updated_at=func.now(),
                updated_by=by,
            )
            .returning(ConfigDocument.id, ConfigDocument.current_revision)
        )
        row = updated.one_or_none()
        if row is None:
            raise RevisionConflictError(await _current_revision(session, project_id, kind, name))
        config_id, revision = row
    await session.execute(
        insert(ConfigRevision).values(
            config_id=config_id,
            revision=revision,
            body=body,
            body_sha256=body_sha256(body),
            description=description,
            created_by=by,
        )
    )
    await bump_epoch(session, project_id)
    saved = await get_config(session, project_id=project_id, kind=kind, name=name)
    if saved is None:  # the row was written in this transaction
        msg = f'{kind} {name!r} vanished inside its own transaction'
        raise RuntimeError(msg)
    return saved


async def get_config(
    session: AsyncSession,
    *,
    project_id: uuid.UUID | None,
    kind: str,
    name: str,
    revision: int | None = None,
) -> ConfigDoc | None:
    """The document at ``revision`` (the current one when ``None``), or ``None``."""
    stmt = _doc_select().where(
        _scope_filter(project_id), ConfigDocument.kind == kind, ConfigDocument.name == name
    )
    if revision is None:
        stmt = stmt.where(ConfigRevision.revision == ConfigDocument.current_revision)
    else:
        stmt = stmt.where(ConfigRevision.revision == revision)
    row = (await session.execute(stmt)).one_or_none()
    return None if row is None else _to_doc(row)


async def get_config_by_id(
    session: AsyncSession, config_id: uuid.UUID, revision: int | None = None
) -> ConfigDoc | None:
    stmt = _doc_select().where(ConfigDocument.id == config_id)
    if revision is None:
        stmt = stmt.where(ConfigRevision.revision == ConfigDocument.current_revision)
    else:
        stmt = stmt.where(ConfigRevision.revision == revision)
    row = (await session.execute(stmt)).one_or_none()
    return None if row is None else _to_doc(row)


async def list_configs(
    session: AsyncSession, *, project_id: uuid.UUID | None, kind: str | None = None
) -> list[ConfigDoc]:
    """Every document of the scope at its current revision, ordered by kind and name."""
    stmt = (
        _doc_select()
        .where(
            _scope_filter(project_id), ConfigRevision.revision == ConfigDocument.current_revision
        )
        .order_by(ConfigDocument.kind, ConfigDocument.name)
    )
    if kind is not None:
        stmt = stmt.where(ConfigDocument.kind == kind)
    return [_to_doc(r) for r in (await session.execute(stmt)).all()]


async def list_revisions(session: AsyncSession, config_id: uuid.UUID) -> list[RevisionInfo]:
    """Every revision of a document, oldest first."""
    stmt = _doc_select().where(ConfigDocument.id == config_id).order_by(ConfigRevision.revision)
    return [
        RevisionInfo(d.revision, d.updated_at, d.updated_by, d.description, d.cloned_from)
        for d in (_to_doc(r) for r in (await session.execute(stmt)).all())
    ]


async def delete_config(
    session: AsyncSession,
    *,
    project_id: uuid.UUID | None,
    kind: str,
    name: str,
    expected_revision: int,
) -> None:
    """Delete a document and its revisions. ``KeyError`` when absent, a conflict on a stale revision,
    :class:`ConfigInUseError` while an activation points at it."""
    current = await _current_revision(session, project_id, kind, name)
    if not current:
        raise KeyError(name)
    if current != expected_revision:
        raise RevisionConflictError(current)
    try:
        async with session.begin_nested():
            result = await session.execute(
                delete(ConfigDocument).where(
                    _scope_filter(project_id),
                    ConfigDocument.kind == kind,
                    ConfigDocument.name == name,
                    ConfigDocument.current_revision == expected_revision,
                )
            )
    except IntegrityError as exc:
        raise ConfigInUseError(name) from exc
    if result.rowcount == 0:  # type: ignore[attr-defined]
        raise RevisionConflictError(await _current_revision(session, project_id, kind, name))
    await bump_epoch(session, project_id)


async def rollback(
    session: AsyncSession,
    *,
    project_id: uuid.UUID,
    axis: str,
    expected_active: dict[str, Any] | None,
    by: str = DEFAULT_ACTOR,
) -> ActiveRef:
    """Reactivate what the latest event moved away from.

    ``LookupError('no_previous')`` when the axis has no earlier state, ``LookupError('previous_deleted')``
    when that document or revision no longer exists.
    """
    events = await list_activation_events(session, project_id, axis, limit=2)
    last = events[0] if events else None
    if last is None:
        msg = 'no_previous'
        raise LookupError(msg)
    if last.from_config_id is None:
        # Moved away from nothing: from "off" when an earlier event exists, else from never-activated.
        if len(events) < 2:
            msg = 'no_previous'
            raise LookupError(msg)
        return await activate(
            session,
            project_id=project_id,
            axis=axis,
            config_id=None,
            revision=None,
            expected_active=expected_active,
            by=by,
            reason='rollback',
        )
    doc = await get_config_by_id(session, last.from_config_id, last.from_revision)
    if doc is None or last.from_revision is None:
        msg = 'previous_deleted'
        raise LookupError(msg)
    return await activate(
        session,
        project_id=project_id,
        axis=axis,
        config_id=last.from_config_id,
        revision=last.from_revision,
        expected_active=expected_active,
        by=by,
        reason='rollback',
    )


async def ensure_config(
    session: AsyncSession,
    *,
    project_id: uuid.UUID | None,
    kind: str,
    name: str,
    body: dict[str, Any],
    description: str,
) -> ConfigDoc:
    """A stored revision that carries exactly ``body``, created by ``system`` when needed.

    Activating a built-in, file-registered or environment configuration needs a stored document so
    the activation's foreign key is honest. The current revision is reused while its body digest
    matches; otherwise a new revision is saved on top of it (a concurrent save is a
    :class:`RevisionConflictError`).
    """
    current = await get_config(session, project_id=project_id, kind=kind, name=name)
    if current is not None and body_sha256(current.body) == body_sha256(body):
        return current
    return await save_config(
        session,
        project_id=project_id,
        kind=kind,
        name=name,
        body=body,
        expected_revision=None if current is None else current.current_revision,
        description=description,
        by=DEFAULT_ACTOR,
    )


async def record_probe(
    session: AsyncSession, *, config_id: uuid.UUID, revision: int, ok: bool, result: dict[str, Any]
) -> None:
    """The latest probe of one endpoint revision (a re-probe replaces it)."""
    stmt = pg_insert(VlmEndpointProbe).values(
        config_id=config_id, revision=revision, ok=ok, result=result
    )
    await session.execute(
        stmt.on_conflict_do_update(
            index_elements=['config_id', 'revision'],
            set_={'ok': stmt.excluded.ok, 'result': stmt.excluded.result, 'probed_at': func.now()},
        )
    )
    await bump_epoch(session, None)


async def list_probes(session: AsyncSession) -> list[Probe]:
    rows = await session.execute(select(VlmEndpointProbe))
    return [
        Probe(p.config_id, p.revision, p.probed_at, p.ok, p.result) for p in rows.scalars().all()
    ]


async def upsert_runtime_report(
    session: AsyncSession, *, project_id: uuid.UUID, process: str, host: str, config_epoch: int
) -> None:
    stmt = pg_insert(ConfigRuntimeReport).values(
        project_id=project_id, process=process, host=host, config_epoch=config_epoch
    )
    await session.execute(
        stmt.on_conflict_do_update(
            index_elements=['project_id', 'process', 'host'],
            set_={'config_epoch': stmt.excluded.config_epoch, 'applied_at': func.now()},
        )
    )


async def list_runtime_reports(
    session: AsyncSession, *, project_id: uuid.UUID, process: str, max_age_s: float
) -> list[RuntimeReport]:
    """Reports of ``process`` refreshed within ``max_age_s`` (an older one belongs to a replaced container)."""
    rows = await session.execute(
        select(ConfigRuntimeReport).where(
            ConfigRuntimeReport.project_id == project_id,
            ConfigRuntimeReport.process == process,
            ConfigRuntimeReport.applied_at >= func.now() - dt.timedelta(seconds=max_age_s),
        )
    )
    return [
        RuntimeReport(r.project_id, r.process, r.host, r.config_epoch, r.applied_at)
        for r in rows.scalars().all()
    ]


@dataclass(frozen=True)
class DeploymentSettingRow:
    key: str
    value: Any
    revision: int
    updated_at: dt.datetime
    updated_by: str


async def get_deployment_setting(session: AsyncSession, key: str) -> DeploymentSettingRow | None:
    row = (
        await session.execute(select(DeploymentSetting).where(DeploymentSetting.key == key))
    ).scalar_one_or_none()
    if row is None:
        return None
    return DeploymentSettingRow(row.key, row.value, row.revision, row.updated_at, row.updated_by)


async def put_deployment_setting(
    session: AsyncSession,
    *,
    key: str,
    value: Any,
    expected_revision: int | None,
    by: str = DEFAULT_ACTOR,
) -> DeploymentSettingRow:
    """Write a deployment setting with OCC: ``None`` creates, a number must be the current revision."""
    if expected_revision is None:
        created = await session.execute(
            pg_insert(DeploymentSetting)
            .values(key=key, value=value, updated_by=by)
            .on_conflict_do_nothing(index_elements=['key'])
            .returning(DeploymentSetting.revision)
        )
        if created.scalar_one_or_none() is None:
            current = await get_deployment_setting(session, key)
            raise RevisionConflictError(current.revision if current else None)
    else:
        updated = await session.execute(
            update(DeploymentSetting)
            .where(DeploymentSetting.key == key, DeploymentSetting.revision == expected_revision)
            .values(
                value=value,
                revision=DeploymentSetting.revision + 1,
                updated_at=func.now(),
                updated_by=by,
            )
            .returning(DeploymentSetting.revision)
        )
        if updated.scalar_one_or_none() is None:
            current = await get_deployment_setting(session, key)
            raise RevisionConflictError(current.revision if current else None)
    await bump_epoch(session, None)
    row = await get_deployment_setting(session, key)
    if row is None:
        msg = f'deployment setting {key!r} vanished inside its own transaction'
        raise RuntimeError(msg)
    return row


async def load_config_document(session: AsyncSession, doc: ConfigDocumentLoad) -> uuid.UUID:
    """Insert a document with every revision, preserving ids and timestamps. Raises on a conflict.

    No epoch bump and no event: the upgrade runs before any process polls. The caller owns the
    transaction.
    """
    check_scope(doc.kind, doc.project_id)
    revisions = sorted(doc.revisions, key=lambda r: r.revision)
    if not revisions:
        msg = f'{doc.kind} {doc.name!r} has no revision to load'
        raise ValueError(msg)
    current = revisions[-1]
    values: dict[str, Any] = {
        'project_id': doc.project_id,
        'kind': doc.kind,
        'name': doc.name,
        'current_revision': current.revision,
        'description': current.description,
        'cloned_from_id': doc.cloned_from_id,
        'cloned_from_revision': doc.cloned_from_revision,
        'updated_by': doc.updated_by,
        'created_at': doc.created_at or revisions[0].created_at,
        'updated_at': doc.updated_at or current.created_at,
    }
    if doc.id is not None:
        values['id'] = doc.id
    config_id = (
        await session.execute(insert(ConfigDocument).values(**values).returning(ConfigDocument.id))
    ).scalar_one()
    await session.execute(
        insert(ConfigRevision),
        [
            {
                'config_id': config_id,
                'revision': r.revision,
                'body': r.body,
                'body_sha256': body_sha256(r.body),
                'description': r.description,
                'created_at': r.created_at,
                'created_by': r.created_by,
            }
            for r in revisions
        ],
    )
    return config_id


async def load_vlm_endpoint(session: AsyncSession, row: VlmEndpointLoad) -> None:
    """Load a deployment-wide endpoint document, its revisions and its probes."""
    if row.document.kind != 'vlm_endpoint' or row.document.project_id is not None:
        msg = 'a VLM endpoint is a deployment-wide vlm_endpoint document'
        raise ConfigScopeError(msg)
    config_id = await load_config_document(session, row.document)
    for probe in row.probes:
        await session.execute(
            insert(VlmEndpointProbe).values(
                config_id=config_id,
                revision=probe.revision,
                probed_at=probe.probed_at,
                ok=probe.ok,
                result=probe.result,
            )
        )
