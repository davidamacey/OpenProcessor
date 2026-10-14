"""Storage of the cluster-refresh daemon's per-project growth state.

It lives in the project's settings document under ``cluster_refresh`` (stored, never
indexed), written with the optimistic-revision store of
:mod:`~src.services.curation.policy_doc_store`. The daemon restores it on its first
poll of a project, so a restart neither retrains a project that was just retrained
nor starts a retrain in the middle of an ingest.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from src.services.curation.policy_doc_store import read_policy, write_policy


KEY = 'cluster_refresh'


class RefreshState(BaseModel):
    revision: int = 0
    last_trained_count: int = 0
    trained_before: int | None = None
    prev_count: int | None = None
    job_id: str | None = None


async def load_refresh_state(client: Any) -> RefreshState:
    """The bound project's state; the defaults when none was ever written. Raises
    when the settings document cannot be read."""
    return (await read_policy(client, KEY, RefreshState))[0]


async def save_refresh_state(client: Any, state: RefreshState) -> RefreshState:
    """Store ``state`` and return it with the next revision. Raises
    :class:`~src.services.curation.policy_doc_store.PolicyConflictError` when the
    stored revision is not the one ``state`` was built from (or a writer won the race)."""
    current, resp = await read_policy(client, KEY, RefreshState)
    if current.revision != state.revision:
        from src.services.curation.policy_doc_store import PolicyConflictError

        msg = f'stored revision is {current.revision}, expected {state.revision}'
        raise PolicyConflictError(msg)
    new = state.model_copy(update={'revision': current.revision + 1})
    await write_policy(client, KEY, new, resp)
    return new
