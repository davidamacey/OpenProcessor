"""Storage of the per-project VLM scope policy (settings document key ``vlm_policy``).

Same optimistic-revision contract as the ingest policy
(:mod:`~src.services.curation.policy_doc_store`): per project, cloned with the
project, no versions.
"""

from __future__ import annotations

from typing import Any

from src.services.curation.policy_doc_store import PolicyConflictError, read_policy, write_policy
from src.services.curation.vlm_policy import VlmPolicy, VlmPolicyBody


KEY = 'vlm_policy'


async def get_vlm_policy(client: Any) -> VlmPolicy:
    """The bound project's policy; the defaults (scope ``all``) when never written."""
    return (await read_policy(client, KEY, VlmPolicy))[0]


async def put_vlm_policy(client: Any, body: VlmPolicyBody, *, expected_revision: int) -> VlmPolicy:
    """Replace the policy. Raises :class:`PolicyConflictError` when
    ``expected_revision`` is not the stored revision (or a concurrent writer won)."""
    current, resp = await read_policy(client, KEY, VlmPolicy)
    if current.revision != expected_revision:
        msg = f'stored revision is {current.revision}, expected {expected_revision}'
        raise PolicyConflictError(msg)
    new = VlmPolicy(revision=current.revision + 1, **body.model_dump())
    await write_policy(client, KEY, new, resp)
    return new
