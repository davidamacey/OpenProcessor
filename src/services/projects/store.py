"""Run one projects-repository call in its own deployment transaction and speak the wire's errors.

Every lifecycle write goes through :func:`run_repo`: it opens a session, wraps the call in
``deployment_tx`` (committing before it returns), and turns the repository's refusals into the
409 and 404 bodies the routes have always answered.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.db.repos import projects as repo
from src.db.session import async_session_factory, deployment_tx
from src.services.rejections import rejection


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from sqlalchemy.ext.asyncio import AsyncSession


def rejection_for(exc: repo.ProjectRepoError, *, action: str | None = None) -> Exception:
    slug = exc.slug
    if isinstance(exc, repo.SlugRetiredError):
        return rejection(409, 'slug_retired', str(exc), project=slug)
    if isinstance(exc, repo.SlugTakenError):
        return rejection(409, 'slug_taken', str(exc), project=slug)
    if isinstance(exc, repo.ProjectNotFoundError):
        return rejection(404, 'project_not_found', str(exc), project=slug)
    if isinstance(exc, repo.InvalidTransitionError):
        detail: dict[str, Any] = {'project': slug, 'project_status': exc.status}
        if action is not None:
            detail['action'] = action
        message = (
            f"cannot {action} '{slug}' while it is {exc.status}"
            if action
            else f"'{slug}' is no longer in a state this call may act on (now '{exc.status}')"
        )
        return rejection(409, 'invalid_transition', message, **detail)
    if isinstance(exc, repo.RevisionConflictError):
        return rejection(
            409,
            'revision_conflict',
            f"'{slug}' was modified by another request; refresh and retry",
            project=slug,
            current_revision=exc.current,
        )
    raise exc


async def run_repo[T](
    reason: str,
    fn: Callable[..., Awaitable[T]],
    /,
    *args: Any,
    action: str | None = None,
    **kwargs: Any,
) -> T:
    """``fn(session, *args, **kwargs)`` in a deployment transaction that commits on return."""
    try:
        async with async_session_factory()() as session, deployment_tx(session, reason):
            return await fn(session, *args, **kwargs)
    except repo.ProjectRepoError as exc:
        raise rejection_for(exc, action=action) from exc


async def get_row(slug: str) -> repo.ProjectRow:
    """The stored project, fresh from the database (never the snapshot); 404 when none."""

    async def read(session: AsyncSession) -> repo.ProjectRow:
        row = await repo.get_by_slug(session, slug)
        if row is None:
            raise repo.ProjectNotFoundError(slug)
        return row

    return await run_repo('read a project', read)
