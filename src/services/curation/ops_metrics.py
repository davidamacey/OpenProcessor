"""Operator-facing ``op_*`` metrics behind the Grafana dashboards (issue #94).

Definitions plus the few helpers the call sites share. Nothing here talks to
OpenSearch; the scrape-time snapshot gauges are filled by
:mod:`src.services.curation.ops_metrics_refresh`.

Label cardinality: no per-item, per-crop or per-image label exists. The only
unbounded-in-principle label is ``project``; :func:`project_label` bounds it to
at most ``OP_METRICS_MAX_PROJECT_LABELS`` (default 50) real slugs, chosen
deterministically (sorted slug order of the project registry) so every API
worker and every worker container picks the same set, and folds the rest into
``other``. ``model`` is the configured VLM model id (a handful per install).

Gauge modes for the multi-process API: snapshot gauges use
``livemostrecent`` (the value last written by any live uvicorn worker, so one
worker refreshing for all is enough and a stale worker cannot pin a high
value); there is deliberately no ``livesum`` gauge because every gauge here is
a snapshot, not a per-process partial.
"""

from __future__ import annotations

import os
from typing import Any

from prometheus_client import Counter, Gauge, Histogram


OTHER_PROJECT = 'other'
_DEFAULT_MAX_PROJECT_LABELS = 50

INGEST_OUTCOMES = ('ok', 'failed', 'skipped')
QUEUES = ('ingest', 'embed', 'label', 'segment')
EMBEDDING_STATES = ('pending', 'embedded', 'failed')
SEGMENTER_OUTCOMES = ('hit', 'miss', 'error')
DETECTION_OUTCOMES = ('ok', 'skipped', 'failed')
VLM_OUTCOMES = ('ok', 'error')

# Heartbeat file name (src.services.curation.worker_liveness) -> ``worker`` label.
WORKER_LABELS = {
    'detection_worker': 'detection',
    'vlm_worker': 'vlm',
    'auto_label_worker': 'curation',
}

OP_INGEST_IMAGES_TOTAL = Counter(
    'op_ingest_images_total',
    'Images through ingest, by project and outcome (ok / failed / skipped = duplicate).',
    labelnames=('project', 'outcome'),
)
OP_INGEST_ITEMS_TOTAL = Counter(
    'op_ingest_items_total',
    'Items (crops) indexed by ingest, by project.',
    labelnames=('project',),
)

OP_QUEUE_DEPTH = Gauge(
    'op_queue_depth',
    'Items waiting in a pipeline queue (embed, label, segment), by project. Snapshot.',
    labelnames=('queue', 'project'),
    multiprocess_mode='livemostrecent',
)
OP_QUEUE_OLDEST_ITEM_AGE_SECONDS = Gauge(
    'op_queue_oldest_item_age_seconds',
    'Age of the longest-waiting item in a queue, worst project. Snapshot.',
    labelnames=('queue',),
    multiprocess_mode='livemostrecent',
)
OP_EMBEDDING_STATE_ITEMS = Gauge(
    'op_embedding_state_items',
    'Items by embedding state (pending = deferred, embedded, failed), by project. Snapshot.',
    labelnames=('project', 'state'),
    multiprocess_mode='livemostrecent',
)

OP_WORKER_LAST_HEARTBEAT_TIMESTAMP_SECONDS = Gauge(
    'op_worker_last_heartbeat_timestamp_seconds',
    'Unix time of the last liveness heartbeat the worker wrote.',
    labelnames=('worker',),
    multiprocess_mode='livemostrecent',
)
OP_WORKER_UP = Gauge(
    'op_worker_up',
    '1 when the worker last reported every sub-task alive (segmenter: probe answered), else 0.',
    labelnames=('worker',),
    multiprocess_mode='livemostrecent',
)

_SEGMENTER_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0)
OP_REGION_SEGMENTER_REQUEST_SECONDS = Histogram(
    'op_region_segmenter_request_seconds',
    'Crop-stage segmenter request duration, by outcome (hit / miss / error).',
    labelnames=('outcome',),
    buckets=_SEGMENTER_BUCKETS,
)
OP_REGION_SEGMENTER_REQUESTS_TOTAL = Counter(
    'op_region_segmenter_requests_total',
    'Crop-stage segmenter requests, by outcome (hit / miss / error).',
    labelnames=('outcome',),
)
OP_DETECTION_WORKER_ITEMS_TOTAL = Counter(
    'op_detection_worker_items_total',
    'Items the detection worker finished, by outcome (ok = written, skipped, failed = write error).',
    labelnames=('outcome',),
)

OP_VLM_TOKENS_TOTAL = Counter(
    'op_vlm_tokens_total',
    'VLM tokens reported by the endpoint usage block, by direction and model.',
    labelnames=('direction', 'model'),
)
OP_VLM_REQUEST_SECONDS = Histogram(
    'op_vlm_request_seconds',
    'VLM chat-completion request duration, by model and outcome (ok / error).',
    labelnames=('model', 'outcome'),
    buckets=(0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0),
)

OP_OPENSEARCH_SHARDS = Gauge(
    'op_opensearch_shards',
    'OpenSearch shards (primaries x (1 + replicas)) per project and index role. Snapshot.',
    labelnames=('project', 'index_role'),
    multiprocess_mode='livemostrecent',
)
OP_OPENSEARCH_STORE_BYTES = Gauge(
    'op_opensearch_store_bytes',
    'OpenSearch on-disk store bytes (primaries + replicas) per project and index role. Snapshot.',
    labelnames=('project', 'index_role'),
    multiprocess_mode='livemostrecent',
)


def max_project_labels() -> int:
    try:
        return max(
            1, int(os.environ.get('OP_METRICS_MAX_PROJECT_LABELS', _DEFAULT_MAX_PROJECT_LABELS))
        )
    except ValueError:
        return _DEFAULT_MAX_PROJECT_LABELS


def allowed_project_slugs(slugs: Any) -> frozenset[str]:
    """The first ``OP_METRICS_MAX_PROJECT_LABELS`` slugs in sorted order."""
    return frozenset(sorted(slugs)[: max_project_labels()])


def project_label(slug: str) -> str:
    """``slug`` when it is within the bounded label set, else ``other``.

    The set comes from the project registry snapshot so it is identical in every
    process. ``slug`` is a bound project, hence a registry project by construction,
    so it is ranked as a member even when this process's snapshot has not caught up
    yet (a project created seconds ago, or a refresh that failed): a lagging snapshot
    must not file a real project under ``other``. An empty registry passes ``slug``
    through.
    """
    from src.services.projects.registry import get_project_registry

    known = get_project_registry().snapshot().keys()
    if not known:
        return slug
    return slug if slug in allowed_project_slugs({*known, slug}) else OTHER_PROJECT


_INGEST_OUTCOME_BY_STATUS = {'success': 'ok', 'failed': 'failed', 'duplicate': 'skipped'}


def record_ingest_result(result: Any) -> Any:
    """Count one finished image (and its items) against the bound project; returns ``result``."""
    from src.config.project_context import try_current_project

    bound = try_current_project()
    project = project_label(bound.record.slug) if bound is not None else OTHER_PROJECT
    outcome = _INGEST_OUTCOME_BY_STATUS.get(result.status, 'failed')
    OP_INGEST_IMAGES_TOTAL.labels(project=project, outcome=outcome).inc()
    if outcome == 'ok' and result.n_crops > 0:
        OP_INGEST_ITEMS_TOTAL.labels(project=project).inc(result.n_crops)
    return result


def record_segmenter_request(outcome: str, seconds: float) -> None:
    OP_REGION_SEGMENTER_REQUESTS_TOTAL.labels(outcome=outcome).inc()
    OP_REGION_SEGMENTER_REQUEST_SECONDS.labels(outcome=outcome).observe(seconds)


def record_vlm_request(model: str, outcome: str, seconds: float, response: Any = None) -> None:
    OP_VLM_REQUEST_SECONDS.labels(model=model, outcome=outcome).observe(seconds)
    usage = response.get('usage') if isinstance(response, dict) else None
    if not isinstance(usage, dict):
        return
    for direction, key in (('prompt', 'prompt_tokens'), ('completion', 'completion_tokens')):
        n = usage.get(key)
        if isinstance(n, int | float) and not isinstance(n, bool) and n > 0:
            OP_VLM_TOKENS_TOTAL.labels(direction=direction, model=model).inc(n)


def record_worker_heartbeat(name: str, tasks: Any, ts: float) -> None:
    worker = WORKER_LABELS.get(name)
    if worker is None:
        return
    OP_WORKER_LAST_HEARTBEAT_TIMESTAMP_SECONDS.labels(worker=worker).set(ts)
    OP_WORKER_UP.labels(worker=worker).set(1 if all(tasks.values()) else 0)


def start_worker_metrics_server(port_setting: str) -> int | None:
    """Serve this worker's registry on ``/metrics`` (threaded, no event loop).

    ``port_setting`` is the raw env value (callers read it with a literal
    ``os.environ.get`` so the env-surface check sees it). Returns the bound
    port, or ``None`` when it is ``0``/``off`` or the port is taken (a second
    process on the same port must not crash the worker; it just is not
    scrapeable).
    """
    from prometheus_client import start_http_server

    raw = port_setting.strip()
    if raw in ('', '0', 'off'):
        return None
    try:
        port = int(raw)
        start_http_server(port)
    except (ValueError, OSError):
        return None
    return port
