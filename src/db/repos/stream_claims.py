"""``stream_worker_claims`` and the worker liveness rows that decide each replica's fair share.

The detection and VLM workers stay stream workers over the pending items of a project (one
Celery message per item would duplicate the index state and lose batching); each replica claims
whole projects (02b section 5, D-53). The claim statement is the design's, verbatim in effect:
the fencing token moves up only when a different worker takes a project over (review R-8), so a
renewal keeps it and a takeover tells the old holder it lost the project. The token does not
fence the OpenSearch writes: their correctness is the per-item OCC and the lock rule.

Functions take the caller's sync session inside its deployment transaction and never commit.
"""

from __future__ import annotations

import json
import uuid
from typing import TYPE_CHECKING, Any

from sqlalchemy import text


if TYPE_CHECKING:
    from sqlalchemy.orm import Session

_CLAIM = text(
    'INSERT INTO stream_worker_claims (worker_kind, project_id, worker_id, lease_expires_at) '
    'VALUES (:kind, :project, :me, now() + make_interval(secs => CAST(:ttl AS double precision))) '
    'ON CONFLICT (worker_kind, project_id) DO UPDATE SET worker_id = excluded.worker_id, '
    'fencing_token = CASE WHEN stream_worker_claims.worker_id <> excluded.worker_id '
    'THEN stream_worker_claims.fencing_token + 1 ELSE stream_worker_claims.fencing_token END, '
    'lease_expires_at = excluded.lease_expires_at '
    'WHERE stream_worker_claims.lease_expires_at < now() '
    'OR stream_worker_claims.worker_id = excluded.worker_id RETURNING fencing_token'
)
_RENEW = text(
    'UPDATE stream_worker_claims SET '
    'lease_expires_at = now() + make_interval(secs => CAST(:ttl AS double precision)) '
    'WHERE worker_kind = :kind AND worker_id = :me RETURNING project_id, fencing_token'
)
_TOUCH_WORKER = text(
    'INSERT INTO worker_capabilities (worker_id, queue, host, capabilities, seen_at) '
    'VALUES (:me, :kind, :host, CAST(:capabilities AS jsonb), now()) '
    'ON CONFLICT (worker_id) DO UPDATE SET queue = excluded.queue, host = excluded.host, '
    'capabilities = excluded.capabilities, seen_at = now()'
)
_LIVE_WORKERS = text(
    'SELECT count(*) FROM worker_capabilities WHERE queue = :kind AND '
    'seen_at > now() - make_interval(secs => CAST(:within AS double precision))'
)


def worker_id(kind: str, host: str) -> str:
    """``<kind>@<hostname>``, the key of a replica's ``worker_capabilities`` row."""
    return f'{kind}@{host}'


def touch_worker(
    session: Session, kind: str, me: str, host: str, capabilities: dict[str, Any] | None = None
) -> None:
    """Upsert this replica's liveness row, so a replica with no claim yet still counts."""
    session.execute(
        _TOUCH_WORKER,
        {'me': me, 'kind': kind, 'host': host, 'capabilities': json.dumps(capabilities or {})},
    )


def live_workers(session: Session, kind: str, within_s: float) -> int:
    """Replicas of ``kind`` seen within ``within_s`` seconds (the asking replica included)."""
    return int(session.execute(_LIVE_WORKERS, {'kind': kind, 'within': within_s}).scalar_one())


def claim(session: Session, kind: str, project: uuid.UUID, me: str, ttl_s: float) -> int | None:
    """Claim or renew one project: its fencing token, or ``None`` when another live worker holds it."""
    token = session.execute(
        _CLAIM, {'kind': kind, 'project': project, 'me': me, 'ttl': ttl_s}
    ).scalar_one_or_none()
    return None if token is None else int(token)


def renew_held(session: Session, kind: str, me: str, ttl_s: float) -> dict[uuid.UUID, int]:
    """Extend every claim this worker still holds; ``{project: token}`` of the ones it does."""
    found = session.execute(_RENEW, {'kind': kind, 'me': me, 'ttl': ttl_s})
    return {uuid.UUID(str(r.project_id)): int(r.fencing_token) for r in found}


def release(session: Session, kind: str, project: uuid.UUID, me: str) -> bool:
    deleted = session.execute(
        text(
            'DELETE FROM stream_worker_claims WHERE worker_kind = :kind AND project_id = :project '
            'AND worker_id = :me'
        ),
        {'kind': kind, 'project': project, 'me': me},
    )
    return bool(deleted.rowcount)  # type: ignore[attr-defined]  # a DELETE returns a CursorResult


def release_all(session: Session, kind: str, me: str) -> int:
    """Shutdown: give every project back at once."""
    deleted = session.execute(
        text('DELETE FROM stream_worker_claims WHERE worker_kind = :kind AND worker_id = :me'),
        {'kind': kind, 'me': me},
    )
    return int(deleted.rowcount)  # type: ignore[attr-defined]  # a DELETE returns a CursorResult


def forget_worker(session: Session, me: str) -> None:
    """Shutdown: stop counting this replica toward the others' fair share."""
    session.execute(text('DELETE FROM worker_capabilities WHERE worker_id = :me'), {'me': me})
