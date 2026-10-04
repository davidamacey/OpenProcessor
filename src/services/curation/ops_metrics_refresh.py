"""Fills the snapshot gauges of :mod:`src.services.curation.ops_metrics`.

Queue depth, embedding state, oldest-item age and OpenSearch shard/store gauges
describe state that lives in OpenSearch, not events a code path emits, so the API
samples it on a timer. Every uvicorn worker runs the loop; a stamp file in the
multiprocess directory lets one of them per interval do the OpenSearch work (the
gauges are ``livemostrecent``, so whichever process wrote last is what a scrape
shows). A failed read leaves the previous value in place -- never a made-up zero.

Queue mapping (what each ``queue`` label means in this codebase):

* ``segment`` -- items awaiting region detection (``pending_detection`` and its
  legacy alias ``pending``).
* ``label`` -- items awaiting VLM verification (``pending_verification`` and the
  legacy ``pending_verify``).
* ``embed`` -- items whose vector was deferred (``embedding_state=deferred``).
* ``ingest`` -- not emitted: ingest is a synchronous request, nothing is persisted
  as waiting.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.config import get_region_fields
from src.config.curation import IndexRole
from src.config.projects import project_index_prefix
from src.config.region_state import RegionStatus
from src.core.logging import get_logger
from src.services.curation import embedding_state as es
from src.services.curation.ops_metrics import (
    OP_EMBEDDING_STATE_ITEMS,
    OP_OPENSEARCH_SHARDS,
    OP_OPENSEARCH_STORE_BYTES,
    OP_QUEUE_DEPTH,
    OP_QUEUE_OLDEST_ITEM_AGE_SECONDS,
    OP_WORKER_LAST_HEARTBEAT_TIMESTAMP_SECONDS,
    OP_WORKER_UP,
    OTHER_PROJECT,
    allowed_project_slugs,
)


if TYPE_CHECKING:
    from collections.abc import Iterable

    from src.config.projects import ProjectRecord


logger = get_logger(__name__)

DEFAULT_INTERVAL_S = 30.0
_SEGMENT_STATUSES = (RegionStatus.PENDING_DETECTION.value, 'pending')
_LABEL_STATUSES = (RegionStatus.PENDING_VERIFICATION.value, 'pending_verify')


def _oldest(agg: dict[str, Any], now: float) -> float | None:
    value = (agg.get('oldest') or {}).get('value')
    return max(0.0, now - float(value) / 1000.0) if value is not None else None


def _agg_body() -> dict[str, Any]:
    oldest = {'oldest': {'min': {'field': 'updated_at'}}}
    return {
        'size': 0,
        'aggs': {
            'segment': {
                'filter': {'terms': {get_region_fields().status: list(_SEGMENT_STATUSES)}},
                'aggs': oldest,
            },
            'label': {
                'filter': {'terms': {get_region_fields().status: list(_LABEL_STATUSES)}},
                'aggs': oldest,
            },
            'embed': {'filter': es.state_clause(es.DEFERRED), 'aggs': oldest},
            'embedded': {'filter': es.embedded_clause()},
            'failed': {'filter': es.state_clause(es.FAILED)},
        },
    }


async def _project_snapshot(client: Any, record: ProjectRecord, now: float) -> dict[str, Any]:
    items = record.resources.indexes[IndexRole.ITEMS]
    resp = await client.search(index=items, body=_agg_body())
    aggs = resp.get('aggregations') or {}

    def count(name: str) -> int:
        return int((aggs.get(name) or {}).get('doc_count', 0))

    return {
        'depth': {q: count(q) for q in ('segment', 'label', 'embed')},
        'age': {q: _oldest(aggs.get(q) or {}, now) for q in ('segment', 'label', 'embed')},
        'embedding': {
            'pending': count('embed'),
            'embedded': count('embedded'),
            'failed': count('failed'),
        },
    }


async def refresh_queues(client: Any, records: Iterable[ProjectRecord], *, now: float) -> None:
    """Set queue depth, oldest age and embedding-state gauges for ``records``."""
    records = list(records)
    allowed = allowed_project_slugs(r.slug for r in records)
    depth: dict[tuple[str, str], int] = defaultdict(int)
    embedding: dict[tuple[str, str], int] = defaultdict(int)
    oldest: dict[str, float] = {}
    ok = 0
    for record in records:
        try:
            snap = await _project_snapshot(client, record, now)
        except Exception as exc:
            logger.warning('ops_metrics_queue_snapshot_failed', project=record.slug, error=str(exc))
            continue
        ok += 1
        label = record.slug if record.slug in allowed else OTHER_PROJECT
        for queue, n in snap['depth'].items():
            depth[queue, label] += n
        for state, n in snap['embedding'].items():
            embedding[label, state] += n
        for queue, age in snap['age'].items():
            if age is not None:
                oldest[queue] = max(oldest.get(queue, 0.0), age)
    if not ok:
        return
    for (queue, project), n in depth.items():
        OP_QUEUE_DEPTH.labels(queue=queue, project=project).set(n)
    for (project, state), n in embedding.items():
        OP_EMBEDDING_STATE_ITEMS.labels(project=project, state=state).set(n)
    for queue in ('segment', 'label', 'embed'):
        OP_QUEUE_OLDEST_ITEM_AGE_SECONDS.labels(queue=queue).set(oldest.get(queue, 0.0))


async def refresh_storage(client: Any, records: Iterable[ProjectRecord]) -> None:
    """Set shard and store-byte gauges per project and index role."""
    records = list(records)
    allowed = allowed_project_slugs(r.slug for r in records)
    by_index: dict[str, tuple[str, str]] = {}
    for record in records:
        project = record.slug if record.slug in allowed else OTHER_PROJECT
        for role, name in record.resources.indexes.items():
            # Roles folded onto one shared index count once (first role wins).
            by_index.setdefault(name, (project, role.value))
    try:
        rows = await client.cat.indices(
            index=f'{project_index_prefix()}*',
            format='json',
            bytes='b',
            h='index,pri,rep,store.size',
        )
    except Exception as exc:
        logger.warning('ops_metrics_storage_snapshot_failed', error=str(exc))
        return
    shards: dict[tuple[str, str], int] = defaultdict(int)
    store: dict[tuple[str, str], int] = defaultdict(int)
    for row in rows or []:
        key = by_index.get(str(row.get('index')))
        if key is None:
            continue
        pri, rep = int(row.get('pri') or 0), int(row.get('rep') or 0)
        shards[key] += pri * (1 + rep)
        store[key] += int(row.get('store.size') or 0)
    for (project, role_name), n in shards.items():
        OP_OPENSEARCH_SHARDS.labels(project=project, index_role=role_name).set(n)
    for (project, role_name), n in store.items():
        OP_OPENSEARCH_STORE_BYTES.labels(project=project, index_role=role_name).set(n)


async def refresh_segmenter_up(*, now: float) -> None:
    """Probe the segmenter's ``/health``: ``op_worker_up{worker="segmenter"}``."""
    from src.services.detection.segmenter_http import first_segmenter_url, segmenter_instances

    url = first_segmenter_url()
    if not url:
        return
    up = await segmenter_instances(url) is not None
    OP_WORKER_UP.labels(worker='segmenter').set(1 if up else 0)
    if up:
        OP_WORKER_LAST_HEARTBEAT_TIMESTAMP_SECONDS.labels(worker='segmenter').set(now)


async def refresh_once(client: Any, records: Iterable[ProjectRecord]) -> None:
    records = list(records)
    now = time.time()
    await refresh_queues(client, records, now=now)
    await refresh_storage(client, records)
    await refresh_segmenter_up(now=now)


def refresh_interval_s() -> float:
    try:
        return float(os.environ.get('OP_METRICS_REFRESH_INTERVAL_S', DEFAULT_INTERVAL_S))
    except ValueError:
        return DEFAULT_INTERVAL_S


def _claim_turn(interval_s: float) -> bool:
    """True when this process should refresh now: no other API worker did within
    the interval. Best effort -- a race only means two refreshes, never none."""
    directory = os.environ.get('PROMETHEUS_MULTIPROC_DIR')
    if not directory:
        return True
    stamp = Path(directory) / 'ops_metrics_refresh.stamp'
    try:
        if time.time() - stamp.stat().st_mtime < interval_s * 0.8:
            return False
    except FileNotFoundError:
        pass
    except OSError:
        return True
    with contextlib.suppress(OSError):
        stamp.touch()
    return True


async def refresh_loop() -> None:
    """Long-lived API task; one failing tick never kills it."""
    from src.core.dependencies import get_opensearch
    from src.services.projects.registry import get_project_registry

    interval = refresh_interval_s()
    if interval <= 0:
        return
    await asyncio.sleep(random.uniform(0, interval))
    while True:
        try:
            if _claim_turn(interval):
                wrapper = await get_opensearch()
                client = getattr(wrapper, 'client', wrapper)
                await refresh_once(client, get_project_registry().active_projects())
        except Exception as exc:
            logger.warning('ops_metrics_refresh_failed', error=str(exc))
        await asyncio.sleep(interval)
