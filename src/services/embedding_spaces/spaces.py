"""Embedding spaces of a project: register, build, ready, retire (embeddings plan 5.2, 5.5).

Every function opens its own ``project_tx`` on a session with no transaction in progress and
commits on a clean exit. A refusal is an ``EmbeddingSpaceError`` carrying the stable wire code.
At most three live spaces exist per (project, target); the one-primary rule is the database's
partial unique index, not this module's.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from sqlalchemy.exc import IntegrityError

from src.clients.curation_opensearch.base import default_knn_spec, knn_field_body
from src.db.repos import embedding_spaces as repo
from src.db.session import project_tx
from src.domain.embedding_spaces import (
    MAX_LIVE_SPACES_PER_TARGET,
    TRANSITIONS,
    EmbeddingSpaceError,
    KnnMethodSpec,
    Role,
    Status,
    Target,
    index_role_for,
    legacy_names,
    physical_names,
    space_key,
)
from src.services.embedding_spaces.catalog import availability


if TYPE_CHECKING:
    import uuid
    from collections.abc import Mapping

    from sqlalchemy.ext.asyncio import AsyncSession

#: ``put_mapping(index_role, body)``: the caller adds the fields to its project index.
PutMapping = Callable[[str, dict[str, Any]], Awaitable[None]]

_TEXT_TARGETS = frozenset({Target.ITEM_TEXT})
DEFAULT_TARGETS = (Target.ITEM, Target.BOX, Target.FRAME)


def mapping_properties(space: repo.SpaceRow) -> dict[str, Any]:
    """The additive ``properties`` that make a new space's fields exist (OpenSearch put mapping)."""
    vector = knn_field_body(space.knn_method, space.dim)
    if space.nested_path is not None:
        return {
            space.nested_path: {
                'type': 'nested',
                'properties': {
                    'box_id': {'type': 'keyword'},
                    'bbox_norm': {'type': 'float', 'index': False},
                    'embedding': vector,
                },
            }
        }
    props: dict[str, Any] = {space.vector_field: vector}
    if space.state_field is not None:
        props[space.state_field] = {'type': 'keyword'}
    return props


def default_space_rows(
    project_id: uuid.UUID,
    version: repo.ModelVersionRow,
    *,
    created_by: str,
    knn_method: KnnMethodSpec | None = None,
) -> list[repo.NewSpace]:
    """The three ``primary``, ``ready`` default spaces (item, box, frame) of a project.

    They keep the 0.6.0 physical names, so an upgraded project and a new one read the same
    fields. The project create (and the upgrade, with ``knn_method`` read from the live mapping)
    inserts these through ``add_spaces`` inside its own transaction.
    """
    method = knn_method or default_knn_spec()
    rows = []
    for target in DEFAULT_TARGETS:
        names = legacy_names(target)
        rows.append(
            repo.NewSpace(
                project_id=project_id,
                space_key=space_key(version.model_key, version.version),
                model_version_id=version.id,
                target=target,
                dim=version.native_dim,
                role=Role.PRIMARY,
                status=Status.READY,
                index_role=index_role_for(target),
                vector_field=names.vector_field,
                state_field=names.state_field,
                nested_path=names.nested_path,
                knn_method=method,
                created_by=created_by,
            )
        )
    return rows


async def _version(session: AsyncSession, model_key: str, version: int) -> repo.ModelVersionRow:
    row = await session.run_sync(lambda s: repo.get_model_version(s, model_key, version))
    if row is None:
        raise EmbeddingSpaceError(
            'embedding_model_unknown', f'no catalog version {model_key} v{version}', status=404
        )
    return row


async def register_space(
    session: AsyncSession,
    project_id: uuid.UUID,
    *,
    model_key: str,
    version: int,
    target: Target,
    dim: int | None = None,
    role: Role = Role.CANDIDATE,
    created_by: str,
    knn_method: KnnMethodSpec | None = None,
    ready_engines: Mapping[str, Mapping[str, Any]] | None = None,
) -> repo.SpaceRow:
    """Create a ``pending`` space. A new space is never primary: the flip makes it so."""
    if role is Role.PRIMARY:
        raise EmbeddingSpaceError('embedding_space_role_invalid', 'a new space is not primary')
    async with project_tx(session, project_id):
        model = await _version(session, model_key, version)
        engines = (
            ready_engines
            if ready_engines is not None
            else await session.run_sync(repo.ready_engine_contracts)
        )
        state = availability(model, engines)
        if not state.available:
            raise EmbeddingSpaceError('embedding_model_unavailable', state.reason or 'unavailable')
        wanted = 'text' if target in _TEXT_TARGETS else 'image'
        if wanted not in model.modalities:
            raise EmbeddingSpaceError(
                'embedding_modality_mismatch', f'{model_key} v{version} has no {wanted} encoder'
            )
        use_dim = dim or model.native_dim
        if use_dim != model.native_dim and use_dim not in model.matryoshka_dims:
            raise EmbeddingSpaceError(
                'embedding_dim_invalid', f'{use_dim} is not a dimension of {model_key} v{version}'
            )
        key = space_key(model_key, version, use_dim, model.native_dim)
        if await repo.count_live_spaces(session, project_id, target) >= MAX_LIVE_SPACES_PER_TARGET:
            raise EmbeddingSpaceError(
                'embedding_space_limit',
                f'at most {MAX_LIVE_SPACES_PER_TARGET} live spaces per target',
            )
        if await repo.get_space(session, project_id, target, key) is not None:
            raise EmbeddingSpaceError(
                'embedding_space_exists', f'{key} already exists for {target.value}'
            )
        names = physical_names(key, target)
        try:
            return await repo.add_space(
                session,
                repo.NewSpace(
                    project_id=project_id,
                    space_key=key,
                    model_version_id=model.id,
                    target=target,
                    dim=use_dim,
                    role=role,
                    status=Status.PENDING,
                    index_role=index_role_for(target),
                    vector_field=names.vector_field,
                    state_field=names.state_field,
                    nested_path=names.nested_path,
                    knn_method=knn_method or default_knn_spec(),
                    created_by=created_by,
                ),
            )
        except IntegrityError as exc:
            raise EmbeddingSpaceError(
                'embedding_space_exists', f'{key} conflicts with an existing space'
            ) from exc


async def _get(
    session: AsyncSession, project_id: uuid.UUID, target: Target, key: str
) -> repo.SpaceRow:
    row = await repo.get_space(session, project_id, target, key)
    if row is None:
        raise EmbeddingSpaceError(
            'embedding_space_not_found', f'no {target.value} space {key}', status=404
        )
    return row


async def _move(
    session: AsyncSession,
    row: repo.SpaceRow,
    to: Status,
    *,
    job_id: uuid.UUID | None = None,
) -> repo.SpaceRow:
    if to not in TRANSITIONS[row.status]:
        raise EmbeddingSpaceError(
            'embedding_space_state',
            f'{row.space_key} is {row.status.value}; cannot become {to.value}',
        )
    moved = await repo.move_status(session, row.id, expected=row.status, to=to, job_id=job_id)
    if moved is None:
        raise EmbeddingSpaceError(
            'embedding_space_state',
            f'{row.space_key} changed while it was being moved to {to.value}',
        )
    return moved


async def begin_build(
    session: AsyncSession,
    project_id: uuid.UUID,
    *,
    target: Target,
    key: str,
    put_mapping: PutMapping,
    job_id: uuid.UUID | None = None,
) -> repo.SpaceRow:
    """``pending -> building`` (or a ``failed`` retry): add the fields, then record the move.

    ``put_mapping`` is additive and idempotent. The status move and the call share one
    transaction, so a refused mapping leaves the space as it was.
    """
    async with project_tx(session, project_id):
        row = await _get(session, project_id, target, key)
        moved = await _move(session, row, Status.BUILDING, job_id=job_id)
        await put_mapping(moved.index_role, mapping_properties(moved))
        return moved


async def mark_ready(
    session: AsyncSession, project_id: uuid.UUID, *, target: Target, key: str
) -> repo.SpaceRow:
    async with project_tx(session, project_id):
        return await _move(session, await _get(session, project_id, target, key), Status.READY)


async def mark_failed(
    session: AsyncSession, project_id: uuid.UUID, *, target: Target, key: str
) -> repo.SpaceRow:
    async with project_tx(session, project_id):
        return await _move(session, await _get(session, project_id, target, key), Status.FAILED)


async def retire_space(
    session: AsyncSession, project_id: uuid.UUID, *, target: Target, key: str
) -> repo.SpaceRow:
    """``-> retiring``; the ``clear`` run then empties the field and ``finish_retire`` ends it.

    The primary can never be retired. A space that never became ready is retired directly.
    """
    async with project_tx(session, project_id):
        row = await _get(session, project_id, target, key)
        if row.role is Role.PRIMARY:
            raise EmbeddingSpaceError(
                'embedding_space_primary', 'the primary space cannot be retired; promote another'
            )
        if row.status in (Status.PENDING, Status.FAILED):
            return await _move(session, row, Status.RETIRED)
        return await _move(session, row, Status.RETIRING)


async def finish_retire(
    session: AsyncSession, project_id: uuid.UUID, *, target: Target, key: str
) -> repo.SpaceRow:
    async with project_tx(session, project_id):
        return await _move(session, await _get(session, project_id, target, key), Status.RETIRED)


async def list_project_spaces(
    session: AsyncSession, project_id: uuid.UUID, target: Target | None = None
) -> list[repo.SpaceRow]:
    async with project_tx(session, project_id):
        return await repo.list_spaces(session, project_id, target=target)
