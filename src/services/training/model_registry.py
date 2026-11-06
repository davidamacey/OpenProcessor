"""Model registry reads and writes outside the bound project's own scope.

A promoted model is owned by one project and may be shared with others, and the Triton repository
holds every project's models side by side, so the questions asked here ("who owns this Triton
name", "which models does this project own", "is it shared with me") cross projects. They run in
a deployment transaction of their own and return plain values; callers hold no session.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.db.repos import models as repo, projects as projects_repo
from src.db.session import async_session_factory, deployment_tx, project_tx


if TYPE_CHECKING:
    import uuid
    from collections.abc import Awaitable, Callable, Collection

    from sqlalchemy.ext.asyncio import AsyncSession


class SharingRevisionConflictError(Exception):
    """``expected_revision`` does not match the stored sharing revision."""

    def __init__(self, current_revision: int) -> None:
        super().__init__(f'sharing revision is {current_revision}')
        self.current_revision = current_revision


class ModelNotOwnedError(LookupError):
    """The project has no live promotion of this Triton name."""


async def _read[T](reason: str, work: Callable[[AsyncSession], Awaitable[T]]) -> T:
    async with async_session_factory()() as session, deployment_tx(session, reason):
        return await work(session)


async def live_model(triton_name: str) -> repo.LiveModel | None:
    """The live promotion of ``triton_name`` with its model and owner slug."""
    return await _read('model registry read', lambda s: repo.get_live(s, triton_name))


async def owner_slug(triton_name: str) -> str | None:
    """The project that promoted ``triton_name``; ``None`` for a model no project promoted."""
    live = await live_model(triton_name)
    return None if live is None else live.owner_slug


async def owned_live(slug: str) -> list[repo.LiveModel]:
    """Every live promotion ``slug`` owns, private and shared alike."""

    async def work(session: AsyncSession) -> list[repo.LiveModel]:
        owner = await projects_repo.get_by_slug(session, slug)
        if owner is None:
            return []
        return await repo.list_live(session, owner_project_id=owner.id)

    return await _read('models a project owns', work)


async def owned_names(slug: str) -> list[str]:
    return [m.promotion.triton_model_name for m in await owned_live(slug)]


async def all_live() -> list[repo.LiveModel]:
    return await _read('every live promotion', repo.list_live)


async def shared_with(slug: str) -> list[repo.LiveModel]:
    """Live promotions of other projects that are shared with ``slug``."""

    async def work(session: AsyncSession) -> list[repo.LiveModel]:
        consumer = await projects_repo.get_by_slug(session, slug)
        if consumer is None:
            return []
        return await repo.list_live_shared_with(session, consumer.id)

    return await _read('models shared with a project', work)


async def consumers(triton_name: str) -> list[str]:
    """The slugs of the projects ``triton_name`` is shared with (empty when private)."""

    async def work(session: AsyncSession) -> list[str]:
        live = await repo.get_live(session, triton_name)
        if live is None:
            return []
        slugs = []
        for project_id in await repo.share_consumers(session, live.model.id):
            row = await projects_repo.get_by_id(session, project_id)
            if row is not None:
                slugs.append(row.slug)
        return sorted(slugs)

    return await _read('who a model is shared with', work)


async def retire(triton_name: str) -> bool:
    """The model was unloaded and its directory deleted: end its live promotion."""
    return await _read('retire a deleted model', lambda s: repo.supersede_live(s, triton_name))


async def set_sharing(
    owner_project_id: uuid.UUID,
    triton_name: str,
    *,
    shared: bool,
    expected_revision: int,
    consumers: Collection[uuid.UUID],
    requested_by: str,
) -> int:
    """Share the owner's model with ``consumers`` (or with nobody) if ``expected_revision`` is
    current; return the new revision. Runs in the owner's scope, so only the owner's rows are
    reachable, and holds the model row's lock so two writers of one model serialise."""
    async with async_session_factory()() as session, project_tx(session, owner_project_id):
        live = await repo.get_live(session, triton_name)
        if live is None or live.promotion.project_id != owner_project_id:
            raise ModelNotOwnedError(triton_name)
        await repo.lock_model(session, live.model.id)
        current = repo.sharing_revision(len(await repo.share_consumers(session, live.model.id)))
        if expected_revision != current:
            raise SharingRevisionConflictError(current)
        wanted = [c for c in consumers if c != owner_project_id] if shared else []
        await repo.replace_shares(session, live.model.id, wanted, requested_by)
        return repo.sharing_revision(len(wanted))
