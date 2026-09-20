"""Tests for the curation Prometheus metric registry.

Previously ``test_all_counters_increment_without_error`` and
``test_histogram_observes_without_error`` called ``.inc()``/``.observe()``
and asserted nothing about the effect — they'd pass identically if the
metric objects were no-ops. Replaced with real assertions on the
collected sample values, label sets, and the actual ``/metrics``
exposition text (plan Wave 5 W5.d).
"""

from __future__ import annotations

from fastapi.testclient import TestClient
from prometheus_client import REGISTRY

from src.services.curation import metrics


def _sample_value(collector, **labels) -> float:
    """Current value of one labeled (or unlabeled) collector."""
    child = collector.labels(**labels) if labels else collector
    return child._value.get()


def test_counters_increment_by_exactly_one() -> None:
    before = _sample_value(metrics.KB_OCC_FINAL_CONFLICT, endpoint='/curation/test-metrics')
    metrics.KB_OCC_FINAL_CONFLICT.labels(endpoint='/curation/test-metrics').inc()
    after = _sample_value(metrics.KB_OCC_FINAL_CONFLICT, endpoint='/curation/test-metrics')
    assert after == before + 1

    before_unlabeled = _sample_value(metrics.KB_GEMMA_CALL_COMBINED_COUNT)
    metrics.KB_GEMMA_CALL_COMBINED_COUNT.inc()
    after_unlabeled = _sample_value(metrics.KB_GEMMA_CALL_COMBINED_COUNT)
    assert after_unlabeled == before_unlabeled + 1


def test_worker_skip_human_won_tracks_per_writer_id() -> None:
    before_a = _sample_value(metrics.KB_WORKER_SKIP_HUMAN_WON, writer_id='test-writer-a')
    before_b = _sample_value(metrics.KB_WORKER_SKIP_HUMAN_WON, writer_id='test-writer-b')
    metrics.KB_WORKER_SKIP_HUMAN_WON.labels(writer_id='test-writer-a').inc()
    after_a = _sample_value(metrics.KB_WORKER_SKIP_HUMAN_WON, writer_id='test-writer-a')
    after_b = _sample_value(metrics.KB_WORKER_SKIP_HUMAN_WON, writer_id='test-writer-b')
    # Only the incremented label combination moved.
    assert after_a == before_a + 1
    assert after_b == before_b


def _histogram_count_and_sum(collector) -> tuple[float, float]:
    """Read a labeled Histogram child's ``_count``/``_sum`` samples via
    its own ``collect()`` (the public API), not the internal buckets."""
    (sample_family,) = collector.collect()
    count = next(s.value for s in sample_family.samples if s.name.endswith('_count'))
    total = next(s.value for s in sample_family.samples if s.name.endswith('_sum'))
    return count, total


def test_histogram_observe_updates_count_and_sum() -> None:
    hist = metrics.KB_OCC_RETRY_COUNT.labels(endpoint='/curation/test-histogram')
    before_count, before_sum = _histogram_count_and_sum(hist)
    for v in (0, 1, 2, 3):
        hist.observe(v)
    after_count, after_sum = _histogram_count_and_sum(hist)
    assert after_count == before_count + 4
    assert after_sum == before_sum + (0 + 1 + 2 + 3)


def test_registry_sees_expected_metric_names_and_label_sets() -> None:
    expected_labeled = {
        'kb_occ_retry_count': ('endpoint',),
        'kb_occ_final_conflict': ('endpoint',),
        'kb_worker_skip_human_won': ('writer_id',),
    }
    expected_unlabeled = {
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
    all_expected = set(expected_labeled) | expected_unlabeled
    missing = {
        name
        for name in all_expected
        if not any(c == name or c.startswith(f'{name}_') for c in collected)
    }
    assert not missing, f'Missing from registry: {sorted(missing)}'

    # Label sets: each metric's own `_labelnames` attribute is the real
    # source of truth, checked against a literal expected tuple (not
    # re-derived from the metric itself).
    for name, expected_labelnames in expected_labeled.items():
        metric_attr = name.upper()
        collector = getattr(metrics, metric_attr)
        assert collector._labelnames == expected_labelnames, (
            f'{name} label set changed: {collector._labelnames!r}'
        )


def test_metrics_exposition_contains_curation_counters_after_increment() -> None:
    """The actual /metrics HTTP exposition — not just the in-process
    registry — contains a curation metric and its label after a real
    increment, proving the whole collection -> render path works."""
    from src.main import app

    metrics.KB_SHM_CROP_CACHE_HITS.inc()
    metrics.KB_OCC_FINAL_CONFLICT.labels(endpoint='/curation/exposition-test').inc()

    client = TestClient(app)
    body = client.get('/metrics').text
    assert 'kb_shm_crop_cache_hits' in body
    assert 'kb_occ_final_conflict' in body
    assert 'endpoint="/curation/exposition-test"' in body
