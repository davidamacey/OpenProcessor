"""``op.periodic.job_stats``: queue and job gauges from the database, the truth, not the broker.

One grouped read over ``jobs`` sets ``op_jobs{kind,status}`` (queued, running, paused,
interrupted), ``op_jobs_oldest_queued_seconds{kind}`` and ``op_jobs_failed_24h{kind}`` for every
registered job kind, zero where there is none, so a dashboard never sees a missing series. The
gauges use ``livemostrecent``: each beat tick runs in some pool child, and the most recent value of
a live process wins (https://prometheus.github.io/client_python/multiprocess/).
"""

from __future__ import annotations

from prometheus_client import Gauge
from sqlalchemy import text

from src.config.job_kinds import KINDS
from src.db.session import async_session_factory, deployment_tx
from src.domain import job_states as js


STATUSES = (js.QUEUED, js.RUNNING, js.PAUSED, js.INTERRUPTED)

OP_JOBS = Gauge(
    'op_jobs',
    'Jobs by kind and status (queued, running, paused, interrupted).',
    ('kind', 'status'),
    multiprocess_mode='livemostrecent',
)
OP_JOBS_OLDEST_QUEUED_SECONDS = Gauge(
    'op_jobs_oldest_queued_seconds',
    'Age of the oldest queued job of each kind.',
    ('kind',),
    multiprocess_mode='livemostrecent',
)
OP_JOBS_FAILED_24H = Gauge(
    'op_jobs_failed_24h',
    'Jobs of each kind that failed in the last 24 hours.',
    ('kind',),
    multiprocess_mode='livemostrecent',
)

_COUNTS = text(
    'SELECT kind, status, count(*) AS n FROM jobs '
    'WHERE status = ANY(:statuses) GROUP BY kind, status'
)
_OLDEST = text(
    'SELECT kind, extract(epoch FROM now() - min(queued_at)) AS age FROM jobs '
    "WHERE status = 'queued' GROUP BY kind"
)
_FAILED = text(
    'SELECT kind, count(*) AS n FROM jobs '
    "WHERE status = 'failed' AND finished_at > now() - interval '24 hours' GROUP BY kind"
)


async def tick() -> dict[str, int]:
    """Read the three groupings in one transaction and set every gauge; the queued total."""
    async with async_session_factory()() as session, deployment_tx(session, 'job gauges'):
        counts = {
            (r.kind, r.status): int(r.n)
            for r in (await session.execute(_COUNTS, {'statuses': list(STATUSES)}))
        }
        oldest = {r.kind: float(r.age) for r in (await session.execute(_OLDEST))}
        failed = {r.kind: int(r.n) for r in (await session.execute(_FAILED))}
    kinds = set(KINDS) | {kind for kind, _ in counts} | set(oldest) | set(failed)
    for kind in kinds:
        for status in STATUSES:
            OP_JOBS.labels(kind, status).set(counts.get((kind, status), 0))
        OP_JOBS_OLDEST_QUEUED_SECONDS.labels(kind).set(oldest.get(kind, 0.0))
        OP_JOBS_FAILED_24H.labels(kind).set(failed.get(kind, 0))
    return {'queued': sum(n for (_, status), n in counts.items() if status == js.QUEUED)}
