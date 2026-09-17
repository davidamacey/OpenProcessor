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
    metrics.LEGACY_OCC_FINAL_CONFLICT.labels(endpoint='/curation/test').inc()
    metrics.LEGACY_WORKER_SKIP_HUMAN_WON.labels(writer_id='test-writer').inc()
    metrics.LEGACY_GEMMA_CALL_COMBINED_COUNT.inc()
    metrics.LEGACY_GEMMA_CALL_SEPARATE_COUNT.inc()
    metrics.LEGACY_GEMMA_COMBINED_PARSE_FAILURE.inc()
    metrics.LEGACY_SHM_CROP_CACHE_HITS.inc()
    metrics.LEGACY_SHM_CROP_CACHE_MISSES.inc()
    metrics.LEGACY_SHM_CROP_CACHE_EVICTIONS.inc()
    metrics.LEGACY_SOURCE_IMAGE_PREFETCH_HITS.inc()
    metrics.LEGACY_SOURCE_IMAGE_PREFETCH_MISSES.inc()
    metrics.LEGACY_SOURCE_IMAGE_DECODE_COUNT.inc()
    metrics.LEGACY_THUMBNAIL_CACHE_HITS.inc()
    metrics.LEGACY_THUMBNAIL_CACHE_MISSES.inc()


def test_histogram_observes_without_error() -> None:
    for v in (0, 1, 2, 3):
        metrics.LEGACY_OCC_RETRY_COUNT.labels(endpoint='/curation/test').observe(v)


def test_registry_sees_expected_metric_names() -> None:
    expected = {
        'legacy_occ_retry_count',
        'legacy_occ_final_conflict',
        'legacy_worker_skip_human_won',
        'legacy_gemma_call_combined_count',
        'legacy_gemma_call_separate_count',
        'legacy_gemma_combined_parse_failure',
        'legacy_shm_crop_cache_hits',
        'legacy_shm_crop_cache_misses',
        'legacy_shm_crop_cache_evictions',
        'legacy_source_image_prefetch_hits',
        'legacy_source_image_prefetch_misses',
        'legacy_source_image_decode_count',
        'legacy_thumbnail_cache_hits',
        'legacy_thumbnail_cache_misses',
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
