"""Remove staged VLM batch images that no live job needs.

The verify-region route stages its images under ``cache/vlm_batch/<batch_id>/`` and the job's
handler deletes them when it ends. A job that never ran (cancelled while queued, failed at
enqueue, failed by the reaper) leaves them behind; this sweep deletes every prefix whose job is
terminal. A prefix with no job at all is deleted once its newest file is older than the grace,
because the route stages before it inserts the job row.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from sqlalchemy import text

from src.db.session import sync_deployment_tx, sync_session_factory
from src.domain import job_states as js
from src.domain.storage_keys import make_key
from src.services.curation.vlm_batch import STAGING_DIR, STAGING_ROOT, staging_prefix, staging_store


if TYPE_CHECKING:
    from sqlalchemy.orm import Session

KIND = 'vlm_verify_region_batch'
MISSING_JOB_GRACE_S = 3600.0
_STATUS = text("SELECT status FROM jobs WHERE kind = :kind AND params ->> 'batch_id' = :batch")


def _batches() -> dict[str, list[str]]:
    by_batch: dict[str, list[str]] = {}
    for key in staging_store().list_prefix(make_key(STAGING_ROOT, STAGING_DIR)):
        parts = key.split('/')
        if len(parts) >= 4:
            by_batch.setdefault(parts[2], []).append(key)
    return by_batch


def _newest_age_s(keys: list[str], now: float) -> float:
    store = staging_store()
    return now - max(store.stat(k).mtime for k in keys)


def sweep(session: Session, *, now: float | None = None) -> list[str]:
    """Delete the unneeded prefixes; the batch ids removed."""
    clock = time.time() if now is None else now
    removed = []
    with sync_deployment_tx(session, 'sweep staged vlm batch images'):
        for batch_id, keys in _batches().items():
            status = session.execute(_STATUS, {'kind': KIND, 'batch': batch_id}).scalar()
            gone = (
                _newest_age_s(keys, clock) > MISSING_JOB_GRACE_S
                if status is None
                else status in js.TERMINAL
            )
            if gone:
                staging_store().delete_prefix(staging_prefix(batch_id))
                removed.append(batch_id)
    return removed


def run_sweep() -> list[str]:
    with sync_session_factory()() as session:
        return sweep(session)
