"""``worker_capabilities`` as a liveness table: is a consumer of a queue there right now?

A worker that consumes a queue no GPU card is attached to (the ``cluster`` queue) writes its row
from a timer thread every :data:`INTERVAL_S` seconds; a reader treats a row seen within
:data:`FRESH_S` as a live consumer. Both calls take the caller's session inside a deployment
transaction and never commit.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from sqlalchemy import text


if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession
    from sqlalchemy.orm import Session

INTERVAL_S = 15.0
FRESH_S = 60.0

_TOUCH = text(
    'INSERT INTO worker_capabilities (worker_id, queue, host, capabilities, seen_at) '
    'VALUES (:worker_id, :queue, :host, CAST(:capabilities AS jsonb), now()) '
    'ON CONFLICT (worker_id) DO UPDATE SET queue = excluded.queue, host = excluded.host, '
    'seen_at = now()'
)
_LIVE = text(
    'SELECT count(*) FROM worker_capabilities WHERE queue = :queue AND '
    'seen_at > now() - make_interval(secs => CAST(:within AS double precision))'
)


def touch(
    session: Session, queue: str, host: str, capabilities: dict[str, Any] | None = None
) -> None:
    """Record that this host consumes ``queue`` now (sync: a worker's timer thread)."""
    session.execute(
        _TOUCH,
        {
            'worker_id': f'{queue}@{host}',
            'queue': queue,
            'host': host,
            'capabilities': json.dumps(capabilities or {}),
        },
    )


async def live_consumers(session: AsyncSession, queue: str, within_s: float = FRESH_S) -> int:
    """How many hosts refreshed their row for ``queue`` within ``within_s`` seconds."""
    return int((await session.execute(_LIVE, {'queue': queue, 'within': within_s})).scalar_one())
