"""Prometheus metric registry for curation OCC + worker conflict + VLM combined-call paths.

This module is import-only: it defines the :class:`Counter` and
:class:`Histogram` objects for the curation subsystem and exposes them
at module scope. It deliberately does NOT wire any of these metrics
into the worker / OCC / VLM code paths — the owning services are
responsible for incrementing / observing them where appropriate.

Metric *names* (``kb_*``) are unchanged from the reference
implementation this module was ported from — renaming them is
deferred to a later phase since they are operationally visible
(dashboards, alert rules); only the *module path* moves here. See
``docs/design/curation_design_rationale.md`` §6 (known gaps).
"""

from __future__ import annotations

from prometheus_client import Counter, Histogram

from src.core.metrics import HTTP_REQUEST_DURATION_SECONDS  # re-exported below via __all__


# ---------------------------------------------------------------------------
# OCC (optimistic concurrency control)
# ---------------------------------------------------------------------------

KB_OCC_RETRY_COUNT = Histogram(
    'kb_occ_retry_count',
    'Distribution of OCC retry attempts per write (0 = first-try success).',
    labelnames=('endpoint',),
    buckets=(0, 1, 2, 3),
)

KB_OCC_FINAL_CONFLICT = Counter(
    'kb_occ_final_conflict',
    'OCC writes that exhausted retries and returned 409 to the caller.',
    labelnames=('endpoint',),
)

# ---------------------------------------------------------------------------
# Worker conflict skips
# ---------------------------------------------------------------------------

KB_WORKER_SKIP_HUMAN_WON = Counter(
    'kb_worker_skip_human_won',
    'Worker writes skipped because a human label (or higher-precedence writer) won.',
    labelnames=('writer_id',),
)

# ---------------------------------------------------------------------------
# VLM combined vs separate calls
# ---------------------------------------------------------------------------

KB_GEMMA_CALL_COMBINED_COUNT = Counter(
    'kb_gemma_call_combined_count',
    'VLM calls made via the combined verify+OCR prompt path.',
)

KB_GEMMA_CALL_SEPARATE_COUNT = Counter(
    'kb_gemma_call_separate_count',
    'VLM calls made via the legacy separate verify / OCR prompts.',
)

KB_GEMMA_COMBINED_PARSE_FAILURE = Counter(
    'kb_gemma_combined_parse_failure',
    'Combined-call responses that failed to parse (fell back to separate calls).',
)

# ---------------------------------------------------------------------------
# Shared-memory crop cache
# ---------------------------------------------------------------------------

KB_SHM_CROP_CACHE_HITS = Counter(
    'kb_shm_crop_cache_hits',
    'Crop lookups served from the shared-memory crop cache.',
)

KB_SHM_CROP_CACHE_MISSES = Counter(
    'kb_shm_crop_cache_misses',
    'Crop lookups that missed the shared-memory crop cache.',
)

KB_SHM_CROP_CACHE_EVICTIONS = Counter(
    'kb_shm_crop_cache_evictions',
    'Entries evicted from the shared-memory crop cache.',
)

# ---------------------------------------------------------------------------
# Source-image prefetch + decode counters
# ---------------------------------------------------------------------------

KB_SOURCE_IMAGE_PREFETCH_HITS = Counter(
    'kb_source_image_prefetch_hits',
    'Source-image opens that found the file already in the page cache (prefetch landed).',
)

KB_SOURCE_IMAGE_PREFETCH_MISSES = Counter(
    'kb_source_image_prefetch_misses',
    'Source-image opens that had to fault from disk despite prefetch.',
)

KB_SOURCE_IMAGE_DECODE_COUNT = Counter(
    'kb_source_image_decode_count',
    'Total source-image decodes performed.',
)

# ---------------------------------------------------------------------------
# Thumbnail cache
# ---------------------------------------------------------------------------

KB_THUMBNAIL_CACHE_HITS = Counter(
    'kb_thumbnail_cache_hits',
    'Thumbnail requests served from the in-memory cache.',
)

KB_THUMBNAIL_CACHE_MISSES = Counter(
    'kb_thumbnail_cache_misses',
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

KB_STAGE_A_SAM_DURATION_SECONDS = Histogram(
    'kb_stage_a_sam_duration_seconds',
    'Stage A SAM3 segment_plate call duration in seconds.',
    labelnames=('outcome',),  # hit / miss / error
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
)

KB_STAGE_A_GEMMA_VISIBLE_DURATION_SECONDS = Histogram(
    'kb_stage_a_gemma_visible_duration_seconds',
    'Stage A.gemma_visible plate_visible_batch call duration in seconds.',
    labelnames=('outcome',),  # ok / parse_failed / error
    buckets=(0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0),
)

KB_STAGE_B_GEMMA_VERIFY_DURATION_SECONDS = Histogram(
    'kb_stage_b_gemma_verify_duration_seconds',
    'Stage B VLM verify_plate_batch call duration in seconds.',
    labelnames=('outcome',),  # ok / parse_failed / error
    buckets=(0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0),
)

KB_STAGE_LPR_DURATION_SECONDS = Histogram(
    'kb_stage_lpr_duration_seconds',
    'Stage A LPR (lpr_nanov11_640) detect_batch call duration in seconds.',
    labelnames=('outcome',),  # hit / miss / error
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
)

# ---------------------------------------------------------------------------
# SAM3 circuit breaker + bounded retry.
# ---------------------------------------------------------------------------

KB_SAM3_CIRCUIT_OPEN_TOTAL = Counter(
    'kb_sam3_circuit_open_total',
    'Number of times the SAM3 per-host circuit breaker transitioned to UNHEALTHY.',
    labelnames=('host',),
)

KB_SAM3_REQUEST_RETRIES_TOTAL = Counter(
    'kb_sam3_request_retries_total',
    'SAM3 request retry outcomes per host (success_after_retry / failed_after_all_retries).',
    labelnames=('host', 'outcome'),
)

# ---------------------------------------------------------------------------
# SAM3 fan-out decomposition. Three histograms split the total per-call
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
#                     the SAM3 side.
#   * ``response``  = (time json() completed) - (time .post() returned).
#                     Pure client-side parse cost.
# These bookmarks are taken from ``time.monotonic()``. The ``wait``
# bucket is currently 0 by construction because the httpx pool is
# unbounded for our deployment (max_connections=512 default); it is
# retained so a future bounded-pool experiment populates it. The
# ``inflight`` histogram is the load-bearing signal in this phase.
# ---------------------------------------------------------------------------

_SAM3_LATENCY_BUCKETS = (
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

KB_SAM3_REQUEST_WAIT_SECONDS = Histogram(
    'kb_sam3_request_wait_seconds',
    'SAM3 client-side queue wait before the HTTP request is sent.',
    labelnames=('host', 'outcome'),
    buckets=_SAM3_LATENCY_BUCKETS,
)

KB_SAM3_REQUEST_INFLIGHT_SECONDS = Histogram(
    'kb_sam3_request_inflight_seconds',
    'SAM3 HTTP round-trip latency (request issued → full response body received).',
    labelnames=('host', 'outcome'),
    buckets=_SAM3_LATENCY_BUCKETS,
)

KB_SAM3_REQUEST_RESPONSE_SECONDS = Histogram(
    'kb_sam3_request_response_seconds',
    'SAM3 client-side JSON decode + parse latency after HTTP body received.',
    labelnames=('host', 'outcome'),
    buckets=_SAM3_LATENCY_BUCKETS,
)

# ---------------------------------------------------------------------------
# Ingest OCC upsert. A blind bulk upsert can silently clobber
# human-applied labels whenever a deterministic ``crop_id`` already
# exists. The OCC-aware upsert helper preserves human-set guard fields
# and surfaces these two counters so Grafana can show the fix firing in
# production.
# ---------------------------------------------------------------------------

KB_INGEST_OCC_FINAL_CONFLICT = Counter(
    'kb_ingest_occ_final_conflict',
    'Ingest upsert OCC writes that exhausted retries (skipped without clobber).',
)

KB_INGEST_PRESERVED_HUMAN_LABEL = Counter(
    'kb_ingest_preserved_human_label',
    'Ingest upserts that preserved a pre-existing human-applied label field.',
    labelnames=('field',),
)

__all__ = [
    'HTTP_REQUEST_DURATION_SECONDS',
    'KB_GEMMA_CALL_COMBINED_COUNT',
    'KB_GEMMA_CALL_SEPARATE_COUNT',
    'KB_GEMMA_COMBINED_PARSE_FAILURE',
    'KB_INGEST_OCC_FINAL_CONFLICT',
    'KB_INGEST_PRESERVED_HUMAN_LABEL',
    'KB_OCC_FINAL_CONFLICT',
    'KB_OCC_RETRY_COUNT',
    'KB_SAM3_CIRCUIT_OPEN_TOTAL',
    'KB_SAM3_REQUEST_INFLIGHT_SECONDS',
    'KB_SAM3_REQUEST_RESPONSE_SECONDS',
    'KB_SAM3_REQUEST_RETRIES_TOTAL',
    'KB_SAM3_REQUEST_WAIT_SECONDS',
    'KB_SHM_CROP_CACHE_EVICTIONS',
    'KB_SHM_CROP_CACHE_HITS',
    'KB_SHM_CROP_CACHE_MISSES',
    'KB_SOURCE_IMAGE_DECODE_COUNT',
    'KB_SOURCE_IMAGE_PREFETCH_HITS',
    'KB_SOURCE_IMAGE_PREFETCH_MISSES',
    'KB_STAGE_A_GEMMA_VISIBLE_DURATION_SECONDS',
    'KB_STAGE_A_SAM_DURATION_SECONDS',
    'KB_STAGE_B_GEMMA_VERIFY_DURATION_SECONDS',
    'KB_STAGE_LPR_DURATION_SECONDS',
    'KB_THUMBNAIL_CACHE_HITS',
    'KB_THUMBNAIL_CACHE_MISSES',
    'KB_WORKER_SKIP_HUMAN_WON',
]
