"""Smoke tests for the curation Prometheus metric registry.

Asserts that every counter / histogram defined in
:mod:`src.services.curation.metrics` can be incremented / observed
without exception, and that the default registry sees them under their
names.
"""

from __future__ import annotations

from prometheus_client import REGISTRY

from src.services.curation import metrics


def test_all_counters_increment_without_error() -> None:
    metrics.KB_OCC_FINAL_CONFLICT.labels(endpoint='/curation/test').inc()
    metrics.KB_WORKER_SKIP_HUMAN_WON.labels(writer_id='test-writer').inc()
    metrics.KB_GEMMA_CALL_COMBINED_COUNT.inc()
    metrics.KB_GEMMA_CALL_SEPARATE_COUNT.inc()
    metrics.KB_GEMMA_COMBINED_PARSE_FAILURE.inc()
    metrics.KB_SHM_CROP_CACHE_HITS.inc()
    metrics.KB_SHM_CROP_CACHE_MISSES.inc()
    metrics.KB_SHM_CROP_CACHE_EVICTIONS.inc()
    metrics.KB_SOURCE_IMAGE_PREFETCH_HITS.inc()
    metrics.KB_SOURCE_IMAGE_PREFETCH_MISSES.inc()
    metrics.KB_SOURCE_IMAGE_DECODE_COUNT.inc()
    metrics.KB_THUMBNAIL_CACHE_HITS.inc()
    metrics.KB_THUMBNAIL_CACHE_MISSES.inc()


def test_histogram_observes_without_error() -> None:
    for v in (0, 1, 2, 3):
        metrics.KB_OCC_RETRY_COUNT.labels(endpoint='/curation/test').observe(v)


def test_registry_sees_expected_metric_names() -> None:
    expected = {
        'kb_occ_retry_count',
        'kb_occ_final_conflict',
        'kb_worker_skip_human_won',
        'kb_gemma_call_combined_count',
        'kb_gemma_call_separate_count',
        'kb_gemma_combined_parse_failure',
        'kb_shm_crop_cache_hits',
        'kb_shm_crop_cache_misses',
        'kb_shm_crop_cache_evictions',
        'kb_source_image_prefetch_hits',
        'kb_source_image_prefetch_misses',
        'kb_source_image_decode_count',
        'kb_thumbnail_cache_hits',
        'kb_thumbnail_cache_misses',
    }
    collected: set[str] = set()
    for collector in list(REGISTRY._collector_to_names.values()):
        collected.update(collector)
    # Counter metrics expose both `<name>_total` and `<name>_created`; match by prefix.
    missing = {
        name
        for name in expected
        if not any(c == name or c.startswith(f'{name}_') for c in collected)
    }
    assert not missing, f'Missing from registry: {sorted(missing)}'
