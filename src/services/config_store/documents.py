"""Config-store service primitives over PostgreSQL.

The repository (:mod:`src.db.repos.config`) takes a caller-owned session; these functions open
the transaction, so a caller in a route or a worker just awaits one. Every transaction here is a
deployment transaction: a clone reads another project's source document for its ``cloned_from``
tag, and a project's VLM activation names a deployment-wide endpoint. The project filter is in
every repository query, not left to row-level security.

Epochs: a write bumps ``projects.config_epoch`` (a project document or activation) or the
deployment's ``global_config_epoch`` (a deployment-wide document, probe or setting). The functions
return the epoch their write produced so the caller can stamp its own snapshot with it
(:meth:`~src.services.config_store.store.ConfigStore.apply_local`).
"""

from __future__ import annotations

import datetime as dt
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from src.config.env import env_float
from src.db.models.config import DEPLOYMENT_CONFIG_KINDS
from src.db.repos import config as repo, settings as settings_repo
from src.db.repos.config import ActiveConflictError, RevisionConflictError
from src.db.repos.config_base import AXIS_KIND
from src.db.session import async_session_factory, deployment_tx
from src.services.config_store.bound_project import bound_project_id, project_id_for
from src.services.config_store.registered import is_registered
from src.services.config_store.snapshot import StoredConfig


if TYPE_CHECKING:
    import uuid
    from collections.abc import AsyncIterator

    from sqlalchemy.ext.asyncio import AsyncSession

    from src.services.config_store.snapshot import AxisRef

# What a shipped template's own description says; a clone must not inherit it.
TEMPLATE_DESCRIPTION = 'Template (clone to use)'

ConfigKind = Literal[
    'prompt_pack', 'region_profile', 'vlm_endpoint', 'open_vocab_set', 'ingest_profile'
]
ConfigAxis = Literal['prompt_pack', 'detection_profile', 'vlm', 'open_vocab', 'ingest_profile']

OP_CONFIG_POLL_S_DEFAULT = 5.0
# Workers refresh their report every 60 s; one several beats old belongs to a replaced container
# (the host is the container id, so the old row is never overwritten).
RUNTIME_REPORT_MAX_AGE_S = 300.0
VLM_ACK_KEY = 'vlm_external_ack'
_MAX_ACKED_REFS = 200
_ACK_ATTEMPTS = 5


@asynccontextmanager
async def tx(reason: str = 'config store') -> AsyncIterator[AsyncSession]:
    """A deployment transaction over the config tables; commits on a clean exit."""
    async with async_session_factory()() as session, deployment_tx(session, reason):
        yield session


def config_poll_interval_s() -> float:
    """How often each process compares the config epochs."""
    return env_float('OP_CONFIG_POLL_S', OP_CONFIG_POLL_S_DEFAULT)


@dataclass(frozen=True)
class CloneSource:
    """The stored document a new document is cloned from (``project`` ``None``: deployment-wide).

    A built-in, file-registered or template source has no stored document: the clone then records
    no source.
    """

    project: str | None
    kind: ConfigKind
    name: str
    revision: int | None


@dataclass(frozen=True)
class Activated:
    """The result of an activation write: the axis's new ref and the epoch the write produced."""

    ref: AxisRef
    epoch: int
    activated_at: str | None


@dataclass(frozen=True)
class Registered:
    """The body of a built-in, file-registered or environment config being activated by name.

    It has no stored document, so the activation stores its body as a system revision first.
    """

    body: dict[str, Any]
    description: str


@dataclass(frozen=True)
class PreviousTarget:
    """What a rollback would reactivate; ``name`` is ``None`` when it would switch the axis off."""

    name: str | None
    revision: int | None


@dataclass(frozen=True)
class ActivationDetail:
    """The axis's current ref and what it moved away from (``previous`` ``None``: nothing before)."""

    ref: AxisRef
    activated_at: str | None
    previous: tuple[str | None, int | None] | None


def _iso(value: dt.datetime | None) -> str | None:
    return None if value is None else value.isoformat()


def stored_from_doc(doc: repo.ConfigDoc) -> StoredConfig:
    return StoredConfig(
        kind=doc.kind,
        name=doc.name,
        revision=doc.revision,
        body=doc.body,
        description=doc.description,
        created_at=_iso(doc.created_at),
        updated_at=_iso(doc.updated_at),
        cloned_from=doc.cloned_from,
    )


def _doc_scope(kind: str, project_id: uuid.UUID) -> uuid.UUID | None:
    return None if kind in DEPLOYMENT_CONFIG_KINDS else project_id


async def _scope_of(kind: str) -> uuid.UUID | None:
    return None if kind in DEPLOYMENT_CONFIG_KINDS else await bound_project_id()


async def _epoch_of(session: AsyncSession, scope: uuid.UUID | None) -> int:
    return await repo.get_epoch(session, scope)


async def save_config(
    *,
    kind: ConfigKind,
    name: str,
    body: dict[str, Any],
    expected_revision: int | None,
    description: str = '',
    cloned_from: CloneSource | None = None,
    project_id: uuid.UUID | None = None,
) -> tuple[StoredConfig, int]:
    """Create (``expected_revision`` ``None``) or save a new revision of a stored config.

    ``project_id`` names the project for a per-project kind (default: the bound project). Raises
    :class:`RevisionConflictError` on a stale ``expected_revision`` or an existing name on create.
    """
    scope = None if kind in DEPLOYMENT_CONFIG_KINDS else project_id or await bound_project_id()
    source_scope: uuid.UUID | None = None
    if cloned_from is not None and cloned_from.project is not None:
        source_scope = await project_id_for(cloned_from.project)
    async with tx('save config') as session:
        source_id: uuid.UUID | None = None
        if cloned_from is not None and cloned_from.revision is not None:
            source = await repo.get_config(
                session,
                project_id=source_scope,
                kind=cloned_from.kind,
                name=cloned_from.name,
                revision=cloned_from.revision,
            )
            if source is not None:
                source_id = source.id
        doc = await repo.save_config(
            session,
            project_id=scope,
            kind=kind,
            name=name,
            body=body,
            expected_revision=expected_revision,
            description=description,
            cloned_from_id=source_id,
            cloned_from_revision=cloned_from.revision
            if source_id is not None and cloned_from
            else None,
        )
        epoch = await _epoch_of(session, scope)
    return stored_from_doc(doc), epoch


async def delete_config(*, kind: ConfigKind, name: str, expected_revision: int) -> int:
    """Delete a document and its revisions; returns the epoch.

    ``KeyError`` when absent, :class:`RevisionConflictError` on a stale revision,
    :class:`~src.db.repos.config.ConfigInUseError` while an activation points at it.
    """
    scope = await _scope_of(kind)
    async with tx('delete config') as session:
        await repo.delete_config(
            session, project_id=scope, kind=kind, name=name, expected_revision=expected_revision
        )
        return await _epoch_of(session, scope)


async def get_revision(
    *, kind: ConfigKind, name: str, revision: int, project_id: uuid.UUID | None = None
) -> StoredConfig | None:
    """One immutable revision of a stored config, or ``None``."""
    scope = None if kind in DEPLOYMENT_CONFIG_KINDS else project_id or await bound_project_id()
    async with tx('read config revision') as session:
        doc = await repo.get_config(
            session, project_id=scope, kind=kind, name=name, revision=revision
        )
    return None if doc is None else stored_from_doc(doc)


async def list_revisions(*, kind: ConfigKind, name: str) -> list[repo.RevisionInfo] | None:
    """Every revision of a stored config, newest first; ``None`` when there is no such document."""
    scope = await _scope_of(kind)
    async with tx('list config revisions') as session:
        doc = await repo.get_config(session, project_id=scope, kind=kind, name=name)
        if doc is None:
            return None
        revisions = await repo.list_revisions(session, doc.id)
    return sorted(revisions, key=lambda r: r.revision, reverse=True)


async def current_revision(*, kind: ConfigKind, name: str) -> int | None:
    scope = await _scope_of(kind)
    async with tx('read config') as session:
        doc = await repo.get_config(session, project_id=scope, kind=kind, name=name)
    return None if doc is None else doc.current_revision


# --- activations ------------------------------------------------------------------------


def wire_revision(axis: str, name: str | None, revision: int | None) -> int | None:
    """The revision a client sees: a registered config (a built-in pack, the ``env`` VLM
    endpoint, ...) has none, however many system revisions of its body are stored."""
    return None if is_registered(axis, name) else revision


def ref_of(active: repo.ActiveRef | None) -> AxisRef:
    """An activation as the snapshot's axis ref: ``None`` never activated, ``'off'``, or ``(name, revision)``."""
    if active is None:
        return None
    if active.state == 'off' or active.name is None:
        return 'off'
    return (active.name, wire_revision(active.axis, active.name, active.revision))


async def _record_ack(session: AsyncSession, project_id: uuid.UUID, ack_ref: str) -> None:
    """Add ``ack_ref`` to the project's acknowledged external endpoints (a merge, so a lost race retries)."""
    for _ in range(_ACK_ATTEMPTS):
        current = await settings_repo.get_state(session, project_id, VLM_ACK_KEY)
        acked: dict[str, str] = dict(
            ((current.value if current else {}) or {}).get('acked_refs') or {}
        )
        acked.setdefault(ack_ref, dt.datetime.now(dt.UTC).isoformat())
        if len(acked) > _MAX_ACKED_REFS:
            acked = dict(sorted(acked.items(), key=lambda kv: kv[1])[-_MAX_ACKED_REFS:])
        try:
            await settings_repo.put_state(
                session,
                project_id,
                VLM_ACK_KEY,
                {'acked_refs': acked},
                expected_revision=current.revision if current else 0,
            )
        except RevisionConflictError:
            continue
        return
    msg = f'could not record the acknowledgement of {ack_ref} after {_ACK_ATTEMPTS} attempts'
    raise RuntimeError(msg)


async def get_acked_refs() -> dict[str, str]:
    """``name@revision -> acknowledged-at`` of the external endpoints the bound project acknowledged."""
    project_id = await bound_project_id()
    async with tx('read vlm acknowledgements') as session:
        state = await settings_repo.get_state(session, project_id, VLM_ACK_KEY)
    return dict(((state.value if state else {}) or {}).get('acked_refs') or {})


async def activate(
    *,
    axis: ConfigAxis,
    name: str | None,
    revision: int | None,
    expected_active: dict[str, Any] | None,
    registered: Registered | None = None,
    ack_ref: str | None = None,
    reason: Literal['activate', 'rollback', 'clone'] = 'activate',
    project_id: uuid.UUID | None = None,
) -> Activated:
    """Point ``axis`` of the bound project at ``name@revision`` (``name`` ``None``: switch it off).

    ``expected_active`` is ``{name, revision}`` or ``None`` for "nothing active"; a stale one is
    :class:`ActiveConflictError`. A ``revision`` of ``None`` names a built-in, file-registered or
    environment config: ``registered`` carries its body, stored first as a system revision so the
    activation points at a real document. ``ack_ref`` records an external-endpoint
    acknowledgement in the same transaction. ``KeyError`` for an unknown name or revision.
    """
    project = project_id or await bound_project_id()
    kind = AXIS_KIND[axis]
    scope = _doc_scope(kind, project)
    async with tx('activate config') as session:
        config_id: uuid.UUID | None = None
        pinned: int | None = None
        if name is not None:
            if revision is None:
                if registered is None:
                    raise KeyError(name)
                doc = await repo.ensure_config(
                    session,
                    project_id=scope,
                    kind=kind,
                    name=name,
                    body=registered.body,
                    description=registered.description,
                )
            else:
                found = await repo.get_config(
                    session, project_id=scope, kind=kind, name=name, revision=revision
                )
                if found is None:
                    raise KeyError(f'{name}@{revision}')
                doc = found
            config_id, pinned = doc.id, doc.revision
        if ack_ref is not None:
            await _record_ack(session, project, ack_ref)
        if (
            expected_active
            and expected_active.get('revision') is None
            and is_registered(axis, expected_active.get('name'))
        ):
            # the wire shows a registered config without a revision; the compare needs the stored one
            held = await repo.get_activation(session, project, axis)
            if held is not None and held.name == expected_active.get('name'):
                expected_active = {'name': held.name, 'revision': held.revision}
        active = await repo.activate(
            session,
            project_id=project,
            axis=axis,
            config_id=config_id,
            revision=pinned,
            expected_active=expected_active,
            reason=reason,
        )
        epoch = await _epoch_of(session, project)
    return Activated(ref_of(active), epoch, _iso(active.activated_at))


async def previous_target(axis: ConfigAxis) -> PreviousTarget:
    """What a rollback of ``axis`` would reactivate.

    ``LookupError('no_previous')`` when the axis has no earlier state, ``LookupError
    ('previous_deleted')`` when that document or revision no longer exists.
    """
    project = await bound_project_id()
    async with tx('read rollback target') as session:
        events = await repo.list_activation_events(session, project, axis, limit=2)
        last = events[0] if events else None
        if last is None:
            raise LookupError('no_previous')
        if last.from_config_id is None:
            if len(events) < 2:
                raise LookupError('no_previous')
            return PreviousTarget(None, None)
        doc = await repo.get_config_by_id(session, last.from_config_id, last.from_revision)
    if doc is None:
        raise LookupError('previous_deleted')
    return PreviousTarget(doc.name, wire_revision(axis, doc.name, doc.revision))


async def get_activation_detail(axis: ConfigAxis) -> ActivationDetail | None:
    """The bound project's activation of ``axis`` with its previous state; ``None`` if never activated."""
    project = await bound_project_id()
    async with tx('read activation') as session:
        active = await repo.get_activation(session, project, axis)
        if active is None:
            return None
        events = await repo.list_activation_events(session, project, axis, limit=2)
    previous: tuple[str | None, int | None] | None = None
    if events:
        last = events[0]
        if last.from_config_id is not None:
            previous = (last.from_name, wire_revision(axis, last.from_name, last.from_revision))
        elif len(events) > 1:
            previous = (None, None)
    return ActivationDetail(ref_of(active), _iso(active.activated_at), previous)


@dataclass(frozen=True)
class ActivationRead:
    """Another project's activation of one axis, with the activated revision's body."""

    state: Literal['active', 'off']
    name: str | None
    revision: int | None
    body: dict[str, Any] | None

    def as_expected(self) -> dict[str, Any]:
        """The ``expected_active`` an activation of the same project must present."""
        return {'name': self.name, 'revision': self.revision}


async def read_activation(slug: str, axis: ConfigAxis) -> ActivationRead | None:
    """Project ``slug``'s activation of ``axis`` read from the database (never a cached snapshot);
    ``None`` when the axis was never activated. The body is the ACTIVATED revision's, which
    diverges from the document's current one once it is saved again without reactivating.
    ``body`` is ``None`` for an off axis or a revision that is gone."""
    project = await project_id_for(slug)
    async with tx('read a project activation') as session:
        active = await repo.get_activation(session, project, axis)
        if active is None:
            return None
        if active.state != 'active' or active.config_id is None:
            return ActivationRead('off', None, None, None)
        doc = await repo.get_config_by_id(session, active.config_id, active.revision)
    return ActivationRead(
        'active',
        active.name,
        wire_revision(axis, active.name, active.revision),
        None if doc is None else doc.body,
    )


# --- runtime reports ---------------------------------------------------------------------


async def report_runtime(*, process: str, host: str, config_epoch: int) -> None:
    """Record that ``process`` on ``host`` applied ``config_epoch`` of the bound project."""
    project = await bound_project_id()
    async with tx('report applied config') as session:
        await repo.upsert_runtime_report(
            session, project_id=project, process=process, host=host, config_epoch=config_epoch
        )


async def live_runtime_reports(*, process: str) -> list[repo.RuntimeReport]:
    """Reports of ``process`` for the bound project refreshed within :data:`RUNTIME_REPORT_MAX_AGE_S`."""
    project = await bound_project_id()
    async with tx('read applied config') as session:
        return await repo.list_runtime_reports(
            session, project_id=project, process=process, max_age_s=RUNTIME_REPORT_MAX_AGE_S
        )


__all__ = [
    'TEMPLATE_DESCRIPTION',
    'Activated',
    'ActivationDetail',
    'ActivationRead',
    'ActiveConflictError',
    'CloneSource',
    'ConfigAxis',
    'ConfigKind',
    'PreviousTarget',
    'Registered',
    'RevisionConflictError',
    'activate',
    'config_poll_interval_s',
    'current_revision',
    'delete_config',
    'get_acked_refs',
    'get_activation_detail',
    'get_revision',
    'list_revisions',
    'live_runtime_reports',
    'previous_target',
    'read_activation',
    'ref_of',
    'report_runtime',
    'save_config',
    'stored_from_doc',
    'tx',
    'wire_revision',
]
