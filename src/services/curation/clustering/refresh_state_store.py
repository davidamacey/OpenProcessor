"""The periodic cluster refresh's per-project clock, in ``project_state`` key ``cluster_refresh``.

The beat tick is stateless between runs; what it must carry over is when a refresh became due
(``pending_since``), when a failed run may be retried (``retry_not_before``), the pool size it saw
(``prev_count``, to tell ingest is still growing it) and the run it started (``job_id``). All
times are wall-clock epoch seconds. A thin adapter over the generic repository.
"""

from __future__ import annotations

from pydantic import BaseModel

from src.db.repos import settings as settings_repo
from src.db.repos.config import RevisionConflictError
from src.db.session import async_session_factory, project_tx
from src.services.config_store.bound_project import bound_project_id
from src.services.curation.policy_doc_store import PolicyConflictError


KEY = 'cluster_refresh'


class RefreshState(BaseModel):
    revision: int = 0
    prev_count: int | None = None
    pending_since: float | None = None
    retry_not_before: float = 0.0
    job_id: str | None = None


async def load_refresh_state() -> RefreshState:
    """The bound project's state; the defaults when none was ever written. Raises when the
    database cannot be read."""
    project_id = await bound_project_id()
    async with async_session_factory()() as session, project_tx(session, project_id):
        row = await settings_repo.get_state(session, project_id, KEY)
    if row is None:
        return RefreshState()
    return RefreshState(**row.value, revision=row.revision)


async def save_refresh_state(state: RefreshState) -> RefreshState:
    """Store ``state`` and return it with the next revision. Raises
    :class:`~src.services.curation.policy_doc_store.PolicyConflictError` when the stored revision
    is not the one ``state`` was built from (or a writer won the race)."""
    project_id = await bound_project_id()
    try:
        async with async_session_factory()() as session, project_tx(session, project_id):
            row = await settings_repo.put_state(
                session,
                project_id,
                KEY,
                state.model_dump(exclude={'revision'}),
                expected_revision=state.revision,
            )
    except RevisionConflictError as exc:
        raise PolicyConflictError(str(exc)) from exc
    return RefreshState(**row.value, revision=row.revision)
