"""Prometheus metric registry for curation OCC + worker conflict + VLM combined-call paths.

This module is import-only: it defines the :class:`Counter` and
:class:`Histogram` objects for the curation subsystem and exposes them
at module scope. It deliberately does NOT wire any of these metrics
into the worker / OCC / VLM code paths — the owning services are
responsible for incrementing / observing them where appropriate.

Metric names use the ``op_*`` prefix (renamed from the reference
implementation's original vendor-specific prefix during the naming
sweep). See ``docs/design/curation_design_rationale.md`` §6 (known gaps).
"""

from __future__ import annotations

from prometheus_client import Counter, Histogram

from src.core.metrics import HTTP_REQUEST_DURATION_SECONDS  # re-exported below via __all__


# ---------------------------------------------------------------------------
# OCC (optimistic concurrency control)
# ---------------------------------------------------------------------------

OP_OCC_RETRY_COUNT = Histogram(
    'op_occ_retry_count',
    'Distribution of OCC retry attempts per write (0 = first-try success).',
    labelnames=('endpoint',),
    buckets=(0, 1, 2, 3),
)

OP_OCC_FINAL_CONFLICT = Counter(
    'op_occ_final_conflict',
    'OCC writes that exhausted retries and returned 409 to the caller.',
    labelnames=('endpoint',),
)

# ---------------------------------------------------------------------------
# Worker conflict skips
# ---------------------------------------------------------------------------

OP_WORKER_SKIP_HUMAN_WON = Counter(
    'op_worker_skip_human_won',
    'Worker writes skipped because a human label (or higher-precedence writer) won.',
    labelnames=('writer_id',),
)

# ---------------------------------------------------------------------------
# VLM combined vs separate calls
# ---------------------------------------------------------------------------

OP_VLM_CALL_COMBINED_COUNT = Counter(
    'op_vlm_call_combined_count',
    'VLM calls made via the combined verify+OCR prompt path.',
)

OP_VLM_CALL_SEPARATE_COUNT = Counter(
    'op_vlm_call_separate_count',
    'VLM calls made via the legacy separate verify / OCR prompts.',
)

OP_VLM_COMBINED_PARSE_FAILURE = Counter(
    'op_vlm_combined_parse_failure',
    'Combined-call responses that failed to parse (fell back to separate calls).',
)

# ---------------------------------------------------------------------------
# Shared-memory crop cache
# ---------------------------------------------------------------------------

OP_SHM_CROP_CACHE_HITS = Counter(
    'op_shm_crop_cache_hits',
    'Crop lookups served from the shared-memory crop cache.',
)

OP_SHM_CROP_CACHE_MISSES = Counter(
    'op_shm_crop_cache_misses',
    'Crop lookups that missed the shared-memory crop cache.',
)

OP_SHM_CROP_CACHE_EVICTIONS = Counter(
    'op_shm_crop_cache_evictions',
    'Entries evicted from the shared-memory crop cache.',
)

# ---------------------------------------------------------------------------
# Source-image prefetch + decode counters
# ---------------------------------------------------------------------------

OP_SOURCE_IMAGE_PREFETCH_HITS = Counter(
    'op_source_image_prefetch_hits',
    'Source-image opens that found the file already in the page cache (prefetch landed).',
)

OP_SOURCE_IMAGE_PREFETCH_MISSES = Counter(
    'op_source_image_prefetch_misses',
    'Source-image opens that had to fault from disk despite prefetch.',
)

OP_SOURCE_IMAGE_DECODE_COUNT = Counter(
    'op_source_image_decode_count',
    'Total source-image decodes performed.',
)

# ---------------------------------------------------------------------------
# Thumbnail cache
# ---------------------------------------------------------------------------

OP_THUMBNAIL_CACHE_HITS = Counter(
    'op_thumbnail_cache_hits',
    'Thumbnail requests served from the in-memory cache.',
)

OP_THUMBNAIL_CACHE_MISSES = Counter(
    'op_thumbnail_cache_misses',
    'Thumbnail requests that missed the cache and were regenerated.',
)

# ---------------------------------------------------------------------------
# HTTP request latency. Route label is the FastAPI route template
# (e.g. ``/curation/crops/{crop_id}/label``), never the raw path, to
# avoid cardinality blow-up from id path-params.
#
# The Histogram itself lives in src.core.metrics (generic, shared by the
# FastAPI middleware in src.main) — defining it here too would register a
# second ``http_request_duration_seconds`` series under the same name and
# crash at import time with "Duplicated timeseries in CollectorRegistry".
# Re-imported above so existing callers that do
# ``from src.services.curation.metrics import HTTP_REQUEST_DURATION_SECONDS``
# keep working unchanged.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Per-stage Prometheus Histograms. Replaces structlog-only ``*_took_ms``
# events as the source of truth for Grafana p50/p99 alerting. The
# structlog logs remain in place for per-crop tail debugging — they are
# not redundant, they carry crop_id + request_id context that aggregate
# histograms cannot.
# ---------------------------------------------------------------------------

OP_STAGE_A_SEGMENTER_DURATION_SECONDS = Histogram(
    'op_stage_a_segmenter_duration_seconds',
    'Stage A segmenter segment call duration in seconds.',
    labelnames=('outcome',),  # hit / miss / error
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
)

OP_STAGE_A_VLM_VISIBLE_DURATION_SECONDS = Histogram(
    'op_stage_a_vlm_visible_duration_seconds',
    'Stage A.vlm_visible region_visible_batch call duration in seconds.',
    labelnames=('outcome',),  # ok / parse_failed / error
    buckets=(0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0),
)

OP_STAGE_B_VLM_VERIFY_DURATION_SECONDS = Histogram(
    'op_stage_b_vlm_verify_duration_seconds',
    'Stage B VLM verify_region_batch call duration in seconds.',
    labelnames=('outcome',),  # ok / parse_failed / error
    buckets=(0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0),
)

OP_STAGE_REGION_DETECTOR_DURATION_SECONDS = Histogram(
    'op_stage_region_detector_duration_seconds',
    'Stage A region detector (single-class detector) detect_batch call duration in seconds.',
    labelnames=('outcome',),  # hit / miss / error
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
)

# ---------------------------------------------------------------------------
# Segmenter circuit breaker + bounded retry.
# ---------------------------------------------------------------------------

OP_SEGMENTER_CIRCUIT_OPEN_TOTAL = Counter(
    'op_segmenter_circuit_open_total',
    'Number of times the segmenter per-host circuit breaker transitioned to UNHEALTHY.',
    labelnames=('host',),
)

OP_SEGMENTER_REQUEST_RETRIES_TOTAL = Counter(
    'op_segmenter_request_retries_total',
    'Segmenter request retry outcomes per host (success_after_retry / failed_after_all_retries).',
    labelnames=('host', 'outcome'),
)

# ---------------------------------------------------------------------------
# Segmenter fan-out decomposition. Three histograms split the total per-call
# wall time into client-side queue wait, HTTP round-trip (request sent →
# full body received), and JSON decode/parse. This answers the
# "oversubscription vs server saturation" question without needing
# per-host queue introspection.
#
# Approximation note: httpx.AsyncClient does NOT expose pre-connection
# wait time as an event. We approximate:
#   * ``wait``      = (time just-before .post() returns its awaitable
#                     starts executing) - (time the call site recorded
#                     its "request created" bookmark, i.e. before
#                     entering the connection pool semaphore)
#   * ``inflight``  = (time .post() returned) - (time .post() started).
#                     This includes connection-pool acquisition,
#                     network RTT, server-side queueing, and decode on
#                     the segmenter side.
#   * ``response``  = (time json() completed) - (time .post() returned).
#                     Pure client-side parse cost.
# These bookmarks are taken from ``time.monotonic()``. The ``wait``
# bucket is currently 0 by construction because the httpx pool is
# unbounded for our deployment (max_connections=512 default); it is
# retained so a future bounded-pool experiment populates it. The
# ``inflight`` histogram is the load-bearing signal in this phase.
# ---------------------------------------------------------------------------

_SEGMENTER_LATENCY_BUCKETS = (
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
    30.0,
)

OP_SEGMENTER_REQUEST_WAIT_SECONDS = Histogram(
    'op_segmenter_request_wait_seconds',
    'Segmenter client-side queue wait before the HTTP request is sent.',
    labelnames=('host', 'outcome'),
    buckets=_SEGMENTER_LATENCY_BUCKETS,
)

OP_SEGMENTER_REQUEST_INFLIGHT_SECONDS = Histogram(
    'op_segmenter_request_inflight_seconds',
    'Segmenter HTTP round-trip latency (request issued → full response body received).',
    labelnames=('host', 'outcome'),
    buckets=_SEGMENTER_LATENCY_BUCKETS,
)

OP_SEGMENTER_REQUEST_RESPONSE_SECONDS = Histogram(
    'op_segmenter_request_response_seconds',
    'Segmenter client-side JSON decode + parse latency after HTTP body received.',
    labelnames=('host', 'outcome'),
    buckets=_SEGMENTER_LATENCY_BUCKETS,
)

# ---------------------------------------------------------------------------
# Ingest OCC upsert. A blind bulk upsert can silently clobber
# human-applied labels whenever a deterministic ``crop_id`` already
# exists. The OCC-aware upsert helper preserves human-set guard fields
# and surfaces these two counters so Grafana can show the fix firing in
# production.
# ---------------------------------------------------------------------------

OP_INGEST_OCC_FINAL_CONFLICT = Counter(
    'op_ingest_occ_final_conflict',
    'Ingest upsert OCC writes that exhausted retries (skipped without clobber).',
)

OP_INGEST_PRESERVED_HUMAN_LABEL = Counter(
    'op_ingest_preserved_human_label',
    'Ingest upserts that preserved a pre-existing human-applied label field.',
    labelnames=('field',),
)

__all__ = [
    'HTTP_REQUEST_DURATION_SECONDS',
    'OP_INGEST_OCC_FINAL_CONFLICT',
    'OP_INGEST_PRESERVED_HUMAN_LABEL',
    'OP_OCC_FINAL_CONFLICT',
    'OP_OCC_RETRY_COUNT',
    'OP_SEGMENTER_CIRCUIT_OPEN_TOTAL',
    'OP_SEGMENTER_REQUEST_INFLIGHT_SECONDS',
    'OP_SEGMENTER_REQUEST_RESPONSE_SECONDS',
    'OP_SEGMENTER_REQUEST_RETRIES_TOTAL',
    'OP_SEGMENTER_REQUEST_WAIT_SECONDS',
    'OP_SHM_CROP_CACHE_EVICTIONS',
    'OP_SHM_CROP_CACHE_HITS',
    'OP_SHM_CROP_CACHE_MISSES',
    'OP_SOURCE_IMAGE_DECODE_COUNT',
    'OP_SOURCE_IMAGE_PREFETCH_HITS',
    'OP_SOURCE_IMAGE_PREFETCH_MISSES',
    'OP_STAGE_A_SEGMENTER_DURATION_SECONDS',
    'OP_STAGE_A_VLM_VISIBLE_DURATION_SECONDS',
    'OP_STAGE_B_VLM_VERIFY_DURATION_SECONDS',
    'OP_STAGE_REGION_DETECTOR_DURATION_SECONDS',
    'OP_THUMBNAIL_CACHE_HITS',
    'OP_THUMBNAIL_CACHE_MISSES',
    'OP_VLM_CALL_COMBINED_COUNT',
    'OP_VLM_CALL_SEPARATE_COUNT',
    'OP_VLM_COMBINED_PARSE_FAILURE',
    'OP_WORKER_SKIP_HUMAN_WON',
]
