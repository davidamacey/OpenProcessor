"""Optimistic-revision storage of a per-project policy in the settings document.

Each policy lives under its own key of the project's settings document (stored,
never indexed). Reads are one real-time GET by id (no cache, so a write is
visible to the very next reader); writes are optimistic: the caller names the
revision it read and a stale one raises :class:`PolicyConflictError`. Policies
share the document, so a write merges into the stored source and never drops
another policy's key.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel

from src.config import IndexRole, get_curation_config, index_name


SETTINGS_DOC_ID = 'default'


class PolicyConflictError(Exception):
    """The stored policy changed since the caller read it."""


def _is_not_found(exc: Exception) -> bool:
    msg = str(exc).lower()
    return 'notfound' in msg or 'not found' in msg or '404' in msg


async def read_policy[P: BaseModel](
    client: Any, key: str, model: type[P]
) -> tuple[P, dict[str, Any] | None]:
    """The stored policy under ``key`` (the model's defaults when never written)
    and the raw settings response (``None`` when the document does not exist)."""
    index = index_name(get_curation_config(), IndexRole.SETTINGS)
    try:
        resp = await client.get(index=index, id=SETTINGS_DOC_ID)
    except Exception as exc:
        if _is_not_found(exc):
            return model(), None
        raise
    stored = (resp.get('_source') or {}).get(key)
    return (model.model_validate(stored) if stored else model()), resp


async def write_policy(client: Any, key: str, new_policy: Any, resp: dict[str, Any] | None) -> None:
    """Store ``new_policy`` under ``key``, guarded by the sequence number of
    ``resp`` (the read it was built from). A concurrent writer raises
    :class:`PolicyConflictError`."""
    index = index_name(get_curation_config(), IndexRole.SETTINGS)
    doc = {key: new_policy.model_dump(), 'updated_at': datetime.now(UTC).isoformat()}
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
