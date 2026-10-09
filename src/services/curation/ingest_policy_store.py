"""Storage of the per-project ingest policy.

It lives in the project's settings document under ``ingest_policy`` (stored,
never indexed): per project, no versions or activation (a change affects only
future ingests and explicit embed runs, never stored data), cloned with the
project. The read/write mechanics are the shared optimistic-revision store in
:mod:`~src.services.curation.policy_doc_store`.
"""

from __future__ import annotations

from typing import Any

from src.services.curation.ingest_policy import IngestPolicy, IngestPolicyBody
from src.services.curation.policy_doc_store import PolicyConflictError, read_policy, write_policy


KEY = 'ingest_policy'


async def get_ingest_policy(client: Any) -> IngestPolicy:
    """The bound project's policy; the defaults when none was ever written."""
    return (await read_policy(client, KEY, IngestPolicy))[0]


async def put_ingest_policy(
    client: Any, body: IngestPolicyBody, *, expected_revision: int
) -> IngestPolicy:
    """Replace the policy. Raises :class:`PolicyConflictError` when
    ``expected_revision`` is not the stored revision (or a concurrent writer won)."""
    current, resp = await read_policy(client, KEY, IngestPolicy)
    if current.revision != expected_revision:
        msg = f'stored revision is {current.revision}, expected {expected_revision}'
        raise PolicyConflictError(msg)
    new = IngestPolicy(revision=current.revision + 1, **body.model_dump())
    await write_policy(client, KEY, new, resp)
    return new
