"""Storage of the per-project ingest policy.

It lives in the project's settings document under ``ingest_policy`` (stored,
never indexed): per project, no versions or activation (a change affects only
future ingests and explicit embed runs, never stored data), cloned with the
project. Reads are one real-time GET by id (no cache, so a write is visible to
the very next ingest); writes are optimistic: the caller names the revision it
read and a stale one raises :class:`PolicyConflictError`.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from src.config import IndexRole, get_curation_config, index_name
from src.services.curation.ingest_policy import IngestPolicy, IngestPolicyBody


SETTINGS_DOC_ID = 'default'


class PolicyConflictError(Exception):
    """The stored policy changed since the caller read it."""


def _is_not_found(exc: Exception) -> bool:
    msg = str(exc).lower()
    return 'notfound' in msg or 'not found' in msg or '404' in msg


async def _read(client: Any) -> tuple[IngestPolicy, dict[str, Any] | None]:
    index = index_name(get_curation_config(), IndexRole.SETTINGS)
    try:
        resp = await client.get(index=index, id=SETTINGS_DOC_ID)
    except Exception as exc:
        if _is_not_found(exc):
            return IngestPolicy(), None
        raise
    stored = (resp.get('_source') or {}).get('ingest_policy')
    return (IngestPolicy.model_validate(stored) if stored else IngestPolicy()), resp


async def get_ingest_policy(client: Any) -> IngestPolicy:
    """The bound project's policy; the defaults when none was ever written."""
    return (await _read(client))[0]


async def put_ingest_policy(
    client: Any, body: IngestPolicyBody, *, expected_revision: int
) -> IngestPolicy:
    """Replace the policy. Raises :class:`PolicyConflictError` when
    ``expected_revision`` is not the stored revision (or a concurrent writer won)."""
    current, resp = await _read(client)
    if current.revision != expected_revision:
        msg = f'stored revision is {current.revision}, expected {expected_revision}'
        raise PolicyConflictError(msg)
    new = IngestPolicy(revision=current.revision + 1, **body.model_dump())
    index = index_name(get_curation_config(), IndexRole.SETTINGS)
    doc = {'ingest_policy': new.model_dump(), 'updated_at': datetime.now(UTC).isoformat()}
    try:
        if resp is None:
            await client.index(index=index, id=SETTINGS_DOC_ID, body=doc, op_type='create')
        else:
            await client.index(
                index=index,
                id=SETTINGS_DOC_ID,
                body={**(resp.get('_source') or {}), **doc},
                if_seq_no=resp.get('_seq_no'),
                if_primary_term=resp.get('_primary_term'),
            )
    except Exception as exc:
        if 'conflict' in str(exc).lower() or '409' in str(exc):
            raise PolicyConflictError(str(exc)) from exc
        raise
    return new
