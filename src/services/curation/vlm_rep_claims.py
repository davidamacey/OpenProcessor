"""The representative crops the VLM scope has already claimed, per cluster.

Scope ``representatives`` asks the VLM about the ``per_cluster`` crops nearest
each cluster centre. A VLM class write moves the crop out of its candidate
cluster into the class cluster, so ranking the cluster's *current* members
again would hand out the next K, then the next, until the cluster is drained.
Claims make the choice sticky: once a cluster's reps are named they stay named
(labelled or not) and the cluster is never ranked again for more.

Stored in the project's settings document next to the policies, on purpose
independent of the policy revision (an edit of any other policy field must not
re-open the pool) and of the worker's in-memory cache (a restart must not
either). Same optimistic-revision contract as the policies.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from src.services.curation.policy_doc_store import read_policy, write_policy


KEY = 'vlm_scope_reps'


class RepClaim(BaseModel):
    cluster_id: int
    crop_ids: list[str]
    """Nearest-first; may be longer than ``per_cluster`` after it was lowered, so
    raising it again returns the same crops instead of claiming new ones."""


class RepClaims(BaseModel):
    revision: int = 0
    claims: list[RepClaim] = Field(default_factory=list)


async def read_rep_claims(client: Any) -> tuple[RepClaims, dict[str, Any] | None]:
    return await read_policy(client, KEY, RepClaims)


async def write_rep_claims(client: Any, new: RepClaims, resp: dict[str, Any] | None) -> None:
    """Raises :class:`~src.services.curation.policy_doc_store.PolicyConflictError`
    when another writer stored claims since ``resp`` was read."""
    await write_policy(client, KEY, new, resp)
