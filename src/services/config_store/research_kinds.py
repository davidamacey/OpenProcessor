"""The research config kinds on the generic repository: ``serving_profile`` (deployment-wide),
``cluster_profile`` and ``assist_loop_preset`` (per project).

Bodies are validated with their pydantic models before they are stored; a stored revision is
immutable. A project kind activates on a project axis (a cluster profile on ``clustering_item`` or
``clustering_region_box``, a preset on ``assist_loop_preset``). The one active serving profile of a
deployment is ``deployment_settings('active_serving_profile')`` = ``{config_id, revision}``, written
with OCC on the setting's own revision.
"""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from src.db.models.config import ACTIVATION_AXES, DEPLOYMENT_CONFIG_KINDS
from src.db.repos import config as repo
from src.db.session import async_session_factory, deployment_tx, project_tx
from src.schemas.curation.research_config import RESEARCH_BODY_MODELS, ResearchConfigDoc
from src.services.config_store.bound_project import bound_project_id


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from sqlalchemy.ext.asyncio import AsyncSession

ActiveConflictError = repo.ActiveConflictError
RevisionConflictError = repo.RevisionConflictError
ACTIVE_SERVING_KEY = 'active_serving_profile'
RESEARCH_KINDS = tuple(RESEARCH_BODY_MODELS)
# The axes a project kind can be activated on.
KIND_AXES: dict[str, tuple[str, ...]] = {
    kind: tuple(axis for axis, k in ACTIVATION_AXES if k == kind)
    for kind in RESEARCH_KINDS
    if kind not in DEPLOYMENT_CONFIG_KINDS
}


class InvalidConfigBodyError(ValueError):
    """The body does not fit its kind's model; ``errors`` is the pydantic error list."""

    def __init__(self, errors: list[Any]) -> None:
        self.errors = errors
        super().__init__(f'invalid config body: {len(errors)} error(s)')


class UnknownConfigError(LookupError):
    """No document (or no such revision) under that kind and name."""


def _validated(kind: str, body: dict[str, Any]) -> dict[str, Any]:
    try:
        return RESEARCH_BODY_MODELS[kind].model_validate(body).model_dump(mode='json')
    except ValidationError as exc:
        raise InvalidConfigBodyError(exc.errors(include_url=False, include_context=False)) from exc


@asynccontextmanager
async def _tx(kind: str) -> AsyncIterator[tuple[AsyncSession, uuid.UUID | None]]:
    """The transaction a kind lives in: deployment scope for a deployment-wide kind, else the bound
    project's."""
    project_id = None if kind in DEPLOYMENT_CONFIG_KINDS else await bound_project_id()
    async with async_session_factory()() as session:
        scope = (
            deployment_tx(session, f'{kind} documents are deployment-wide')
            if project_id is None
            else project_tx(session, project_id)
        )
        async with scope:
            yield session, project_id


async def list_documents(kind: str) -> list[repo.ConfigDoc]:
    async with _tx(kind) as (session, project_id):
        return await repo.list_configs(session, project_id=project_id, kind=kind)


async def get_document(kind: str, name: str, revision: int | None = None) -> repo.ConfigDoc | None:
    async with _tx(kind) as (session, project_id):
        return await repo.get_config(
            session, project_id=project_id, kind=kind, name=name, revision=revision
        )


async def save_document(
    kind: str,
    name: str,
    body: dict[str, Any],
    *,
    expected_revision: int | None,
    description: str = '',
) -> repo.ConfigDoc:
    """Create (``expected_revision`` ``None``) or save a new revision. A stale revision is
    :class:`~src.db.repos.config.RevisionConflictError`."""
    checked = _validated(kind, body)
    async with _tx(kind) as (session, project_id):
        return await repo.save_config(
            session,
            project_id=project_id,
            kind=kind,
            name=name,
            body=checked,
            expected_revision=expected_revision,
            description=description,
        )


async def delete_document(kind: str, name: str, *, expected_revision: int) -> None:
    async with _tx(kind) as (session, project_id):
        await repo.delete_config(
            session,
            project_id=project_id,
            kind=kind,
            name=name,
            expected_revision=expected_revision,
        )


async def activate_project_axis(
    axis: str,
    name: str,
    *,
    revision: int | None,
    expected_active: dict[str, Any] | None,
) -> repo.ActiveRef:
    """Activate ``name`` (at ``revision``, else its current one) on a project axis.

    A name that is not a document of the axis's kind is :class:`UnknownConfigError`; a stale
    ``expected_active`` is :class:`~src.db.repos.config.ActiveConflictError`.
    """
    kind = repo.AXIS_KIND.get(axis)
    if kind is None or kind in DEPLOYMENT_CONFIG_KINDS:
        msg = f'{axis!r} is not a project axis'
        raise UnknownConfigError(msg)
    async with _tx(kind) as (session, project_id):
        assert project_id is not None  # a project kind
        doc = await repo.get_config(
            session, project_id=project_id, kind=kind, name=name, revision=revision
        )
        if doc is None:
            msg = f'no {kind} {name!r}' + (f' at revision {revision}' if revision else '')
            raise UnknownConfigError(msg)
        return await repo.activate(
            session,
            project_id=project_id,
            axis=axis,
            config_id=doc.id,
            revision=doc.revision,
            expected_active=expected_active,
        )


async def get_active(axis: str) -> repo.ActiveRef | None:
    kind = repo.AXIS_KIND[axis]
    async with _tx(kind) as (session, project_id):
        assert project_id is not None
        return await repo.get_activation(session, project_id, axis)


async def activate_serving_profile(
    name: str, *, revision: int | None, expected_revision: int | None
) -> repo.DeploymentSettingRow:
    """Make ``name`` the deployment's serving profile. ``expected_revision`` is the revision of the
    ``active_serving_profile`` setting the caller read (``None``: none was active); a stale one is
    :class:`~src.db.repos.config.RevisionConflictError`."""
    async with _tx('serving_profile') as (session, _):
        doc = await repo.get_config(
            session, project_id=None, kind='serving_profile', name=name, revision=revision
        )
        if doc is None:
            msg = f'no serving_profile {name!r}' + (f' at revision {revision}' if revision else '')
            raise UnknownConfigError(msg)
        return await repo.put_deployment_setting(
            session,
            key=ACTIVE_SERVING_KEY,
            value={'config_id': str(doc.id), 'revision': doc.revision},
            expected_revision=expected_revision,
        )


async def active_serving_profile() -> tuple[repo.ConfigDoc, int] | None:
    """The active profile document at its pinned revision, with the setting's revision."""
    async with _tx('serving_profile') as (session, _):
        setting = await repo.get_deployment_setting(session, ACTIVE_SERVING_KEY)
        if setting is None:
            return None
        doc = await repo.get_config_by_id(
            session, uuid.UUID(setting.value['config_id']), setting.value['revision']
        )
        return None if doc is None else (doc, setting.revision)


def wire_doc(doc: repo.ConfigDoc) -> ResearchConfigDoc:
    return ResearchConfigDoc(
        kind=doc.kind,
        name=doc.name,
        revision=doc.revision,
        description=doc.description,
        body=doc.body,
        created_at=doc.created_at.isoformat(),
        updated_at=doc.updated_at.isoformat(),
        updated_by=doc.updated_by,
    )
