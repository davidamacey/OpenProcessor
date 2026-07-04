"""
Prometheus metrics for the Visual AI API.

Exposes HTTP-level instrumentation via the ``/metrics`` endpoint. Metric
objects live here (not in routers) so middleware and any future worker
code can import them without circular imports.

Multi-worker note: with several uvicorn workers each process keeps its
own registry, so a scrape only sees one worker unless
``PROMETHEUS_MULTIPROC_DIR`` is set — in that mode prometheus_client
aggregates across processes via files in that directory (it must exist
and be writable before workers start).
"""

import os

from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Histogram, generate_latest


# Request latency by route template. Buckets skew low because most
# endpoints answer in tens of milliseconds; the tail buckets catch
# batch/ingest calls.
HTTP_REQUEST_DURATION_SECONDS = Histogram(
    'http_request_duration_seconds',
    'HTTP request latency by method, route template, and status code.',
    labelnames=('method', 'route', 'status'),
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
)


def render_metrics() -> tuple[bytes, str]:
    """Render the metrics exposition payload.

    Returns ``(body, content_type)``. Uses the multiprocess collector
    when ``PROMETHEUS_MULTIPROC_DIR`` is configured, otherwise the
    default per-process registry.
    """
    if os.environ.get('PROMETHEUS_MULTIPROC_DIR'):
        from prometheus_client import multiprocess

        registry = CollectorRegistry()
        multiprocess.MultiProcessCollector(registry)
        return generate_latest(registry), CONTENT_TYPE_LATEST
    return generate_latest(), CONTENT_TYPE_LATEST
