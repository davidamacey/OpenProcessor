"""Which space does a reader use? ``embedding_space(target, key=None)`` (embeddings plan 5.4).

The bound project's primary space for a target, or the one named by ``key``, as a
``ResolvedSpace`` carrying every physical field name, so no reader computes one. Services and
routers call it and pass the field to a client (a client never imports this module).

The answer is cached per process and per project slug together with the ``config_epoch`` it was
read at. The configuration poll (the epoch poll of the config store) reports the current epoch
with ``note_config_epoch``; a cached entry is used only while its epoch equals the reported one.
Until a poll has reported for a slug, the epoch is read from the projects row on every call, so
a flip is never missed. An unbound call raises ``ProjectNotBound`` (fail closed).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from sqlalchemy import select

from src.config.project_context import current_project
from src.db.models.projects import Project
from src.db.repos import embedding_spaces as repo
from src.db.session import async_session_factory, deployment_tx, project_tx
from src.domain.embedding_spaces import EmbeddingSpaceError, ResolvedSpace, Role, Status, Target


if TYPE_CHECKING:
    import uuid


@dataclass(frozen=True)
class _Entry:
    project_id: uuid.UUID
    epoch: int
    spaces: tuple[ResolvedSpace, ...]
    primary_keys: tuple[tuple[Target, str], ...]


_cache: dict[str, _Entry] = {}
_polled_epoch: dict[str, int] = {}


def note_config_epoch(slug: str, epoch: int) -> None:
    """Called by the epoch poll: the newest ``config_epoch`` seen for ``slug``."""
    _polled_epoch[slug] = epoch


def clear_cache() -> None:
    _cache.clear()
    _polled_epoch.clear()


async def _load(slug: str) -> _Entry:
    async with async_session_factory()() as session:
        async with deployment_tx(session, 'resolve a project id and config epoch'):
            found = (
                await session.execute(
                    select(Project.id, Project.config_epoch).where(Project.slug == slug)
                )
            ).one_or_none()
        if found is None:
            raise EmbeddingSpaceError('project_not_found', f'no project {slug!r}', status=404)
        project_id, epoch = found
        async with project_tx(session, project_id):
            rows = await repo.list_spaces(session, project_id, live_only=True)
            versions = {
                r.model_version_id: await repo.get_model_version_by_id(session, r.model_version_id)
                for r in rows
            }
    resolved: list[ResolvedSpace] = []
    primaries: list[tuple[Target, str]] = []
    for r in rows:
        v = versions[r.model_version_id]
        if v is None:  # unreachable: the foreign key restricts the delete
            continue
        resolved.append(
            ResolvedSpace(
                space_key=r.space_key,
                target=r.target,
                vector_field=r.vector_field,
                state_field=r.state_field,
                nested_path=r.nested_path,
                dim=r.dim,
                index_role=r.index_role,
                model_key=v.model_key,
                model_version=v.version,
                triton_image_model=v.triton_image_model,
                triton_text_model=v.triton_text_model,
                knn_method=r.knn_method,
            )
        )
        if r.role is Role.PRIMARY and r.status is Status.READY:
            primaries.append((r.target, r.space_key))
    return _Entry(project_id, int(epoch), tuple(resolved), tuple(primaries))


async def _current_epoch(slug: str) -> int:
    polled = _polled_epoch.get(slug)
    if polled is not None:
        return polled
    async with async_session_factory()() as session, deployment_tx(session, 'read config epoch'):
        epoch = await session.scalar(select(Project.config_epoch).where(Project.slug == slug))
    if epoch is None:
        raise EmbeddingSpaceError('project_not_found', f'no project {slug!r}', status=404)
    return int(epoch)


async def _entry(slug: str) -> _Entry:
    entry = _cache.get(slug)
    if entry is None or entry.epoch != await _current_epoch(slug):
        entry = await _load(slug)
        _cache[slug] = entry
    return entry


async def live_spaces() -> tuple[ResolvedSpace, ...]:
    """Every live space of the bound project, every target (the vector-field exclusion lists)."""
    return (await _entry(current_project().record.slug)).spaces


async def embedding_space(target: Target, key: str | None = None) -> ResolvedSpace:
    """The bound project's primary space for ``target``, or its live space named ``key``."""
    slug = current_project().record.slug
    entry = await _entry(slug)
    if key is None:
        wanted = next((k for t, k in entry.primary_keys if t is target), None)
        if wanted is None:
            raise EmbeddingSpaceError(
                'embedding_space_not_found',
                f'{slug} has no primary {target.value} space',
                status=404,
            )
        key = wanted
    for space in entry.spaces:
        if space.target is target and space.space_key == key:
            return space
    raise EmbeddingSpaceError(
        'embedding_space_not_found', f'{slug} has no live {target.value} space {key}', status=404
    )
