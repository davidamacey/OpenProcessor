"""The 0.6.0 upgrade's load of ``jobs`` rows: ids and clocks preserved, no commit.

Split from :mod:`src.db.repos.jobs` (the one start path) by concern and size; the single-entry
gate (``tests/test_jobs_single_entry.py``) names both modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from sqlalchemy import insert

from src.db.models.jobs import Job


if TYPE_CHECKING:
    import datetime as dt
    import uuid
    from collections.abc import Mapping, Sequence

    from sqlalchemy.orm import Session


@dataclass(frozen=True)
class JobLoad:
    """A finished or interrupted job from a 0.6.0 upgrade, with the id and clocks it already had.

    The normal start path lets the database assign them; the upgrade load preserves them.
    """

    id: uuid.UUID
    project_id: uuid.UUID | None
    kind: str
    queue: str
    status: str
    max_queue_wait: dt.timedelta
    requested_by: str
    created_at: dt.datetime
    updated_at: dt.datetime
    params: Mapping[str, Any] = field(default_factory=dict)
    progress: Mapping[str, Any] = field(default_factory=dict)
    result: Any | None = None
    error_code: str | None = None
    error_message: str | None = None
    started_at: dt.datetime | None = None
    finished_at: dt.datetime | None = None
    scope: str = ''
    exclusive_group: str | None = None
    auto_resume: bool = False


def load_jobs(session: Session, rows: Sequence[JobLoad]) -> int:
    """Insert the upgrade's job rows with their given ids and times; no commit, raises on conflict.

    The upgrade's caller owns the deployment transaction (the load contract of the 0.6.0
    upgrade). Writes the columns ``insert_queued`` sets, with params and progress as JSON objects.
    """
    if not rows:
        return 0
    session.execute(
        insert(Job),
        [
            {
                'id': row.id,
                'project_id': row.project_id,
                'kind': row.kind,
                'scope': row.scope,
                'exclusive_group': row.exclusive_group,
                'queue': row.queue,
                'status': row.status,
                'params': dict(row.params),
                'progress': dict(row.progress),
                'result': row.result,
                'error_code': row.error_code,
                'error_message': row.error_message,
                'auto_resume': row.auto_resume,
                'max_queue_wait': row.max_queue_wait,
                'requested_by': row.requested_by,
                'created_at': row.created_at,
                'started_at': row.started_at,
                'finished_at': row.finished_at,
                'updated_at': row.updated_at,
            }
            for row in rows
        ],
    )
    return len(rows)
