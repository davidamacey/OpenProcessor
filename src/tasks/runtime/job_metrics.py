"""Job metrics observed by the task wrapper: how long a delivery ran and how it ended.

``op_job_duration_seconds{kind,outcome}`` is one observation per delivery; ``outcome`` is the
status the fenced terminal write left on the row (``completed``, ``completed_with_errors``,
``failed``, ``cancelled``, ``queued`` after a yield, ``paused``, ``interrupted``).
``op_jobs_failed_total{kind}`` counts the deliveries that ended ``failed``; it is a counter, so
alerts use ``increase()`` (https://prometheus.io/docs/practices/instrumentation/#counter-vs-gauge-summary-vs-histogram).
Files live in ``PROMETHEUS_MULTIPROC_DIR`` and ``src/tasks/worker_metrics.py`` serves them.
"""

from __future__ import annotations

from prometheus_client import Counter, Histogram

from src.domain import job_states as js


# Seconds: a quick bookkeeping job to a multi-hour ingest; the soft limits of the kinds fit.
DURATION_BUCKETS = (0.5, 1, 5, 15, 60, 300, 900, 1800, 3600, 7200, 14400, 28800)

OP_JOB_DURATION_SECONDS = Histogram(
    'op_job_duration_seconds',
    'Seconds one delivery of a job ran, by kind and the status it left the job in.',
    ('kind', 'outcome'),
    buckets=DURATION_BUCKETS,
)
OP_JOBS_FAILED_TOTAL = Counter(
    'op_jobs_failed_total',
    'Deliveries of each job kind that ended failed.',
    ('kind',),
)


def observe_delivery(kind: str, status: str, seconds: float) -> None:
    OP_JOB_DURATION_SECONDS.labels(kind=kind, outcome=status).observe(max(seconds, 0.0))
    if status == js.FAILED:
        OP_JOBS_FAILED_TOTAL.labels(kind=kind).inc()
