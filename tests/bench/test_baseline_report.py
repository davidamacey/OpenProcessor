"""Report math, Prometheus deltas, GPU summary and the before/after compare."""

from __future__ import annotations

import pytest

from scripts.bench import baseline_report as rep


PROM_BEFORE = """\
# TYPE op_pipeline_stage_seconds histogram
op_pipeline_stage_seconds_bucket{stage="decode",le="1.0"} 4
op_pipeline_stage_seconds_sum{stage="decode"} 2.0
op_pipeline_stage_seconds_count{stage="decode"} 4
# TYPE op_pipeline_stage_bytes_total counter
op_pipeline_stage_bytes_total{stage="decode"} 1000
# TYPE http_other gauge
http_other 5
"""
PROM_AFTER = (
    PROM_BEFORE.replace('sum{stage="decode"} 2.0', 'sum{stage="decode"} 5.0')
    .replace('count{stage="decode"} 4', 'count{stage="decode"} 10')
    .replace('total{stage="decode"} 1000', 'total{stage="decode"} 7000')
)


def test_metrics_delta_histograms_and_counters() -> None:
    delta = rep.metrics_delta(
        rep.parse_prometheus(PROM_BEFORE), rep.parse_prometheus(PROM_AFTER), ('op_pipeline',)
    )
    hist = delta['histograms']['op_pipeline_stage_seconds{stage=decode}']
    assert hist == {'sum': 3.0, 'count': 6.0, 'mean': 0.5}
    assert delta['counters']['op_pipeline_stage_bytes_total{stage=decode}'] == 6000
    assert not any('http_other' in k for k in delta['counters'])


def test_metrics_delta_treats_a_series_new_after_the_run_as_from_zero() -> None:
    delta = rep.metrics_delta({}, rep.parse_prometheus(PROM_AFTER), ('op_pipeline',))
    assert delta['histograms']['op_pipeline_stage_seconds{stage=decode}']['count'] == 10


def test_parse_nvidia_smi_and_summary() -> None:
    samples = [
        rep.parse_nvidia_smi('0, 40, 1000\n1, 10, 200\n'),
        rep.parse_nvidia_smi('0, 80, 3000\n1, 30, 250\n'),
    ]
    summary = rep.summarize_gpu(samples)
    assert summary['0'] == {'util_mean_pct': 60.0, 'mem_peak_mb': 3000.0, 'samples': 2}
    assert summary['1']['util_mean_pct'] == 20.0
    assert rep.summarize_gpu([]) == {}


def test_percentile() -> None:
    assert rep.percentile([1.0, 2.0, 3.0, 4.0, 5.0], 50) == 3.0
    assert rep.percentile([1.0, 2.0, 3.0, 4.0, 5.0], 99) == pytest.approx(4.96)
    assert rep.percentile([], 50) == 0.0


def test_ingest_summary_rates() -> None:
    batches = [
        rep.BatchOutcome(images=4, ok=3, duplicates=1, failed=0, items=6, embedded=5, seconds=2.0),
        rep.BatchOutcome(images=4, ok=4, duplicates=0, failed=0, items=8, embedded=8, seconds=2.0),
    ]
    out = rep.ingest_summary(batches, wall_s=4.0, input_bytes=8_000_000)
    assert out['images_per_s'] == 2.0
    assert out['items_per_s'] == 3.5
    assert out['embedded'] == 13
    assert out['failed'] == 0
    assert out['input_mb_per_s'] == 2.0
    assert out['batch_s_p50'] == 2.0


def test_ingest_summary_zero_wall_does_not_divide_by_zero() -> None:
    out = rep.ingest_summary([], wall_s=0.0, input_bytes=0)
    assert out['images_per_s'] == 0.0


def test_storage_summary_per_image() -> None:
    out = rep.storage_summary(images=100, store_bytes=2_000_000, crop_cache_bytes=500_000)
    assert out['store_bytes_per_image'] == 20_000
    assert out['crop_cache_bytes_per_image'] == 5_000
    assert (
        rep.storage_summary(images=0, store_bytes=5, crop_cache_bytes=0)['store_bytes_per_image']
        == 0.0
    )


def _report(rate: float, decode_mean: float) -> dict:
    return {
        'ingest': {'images_per_s': rate, 'failed': 0},
        'metrics': {'api': {'histograms': {'op_x{stage=decode}': {'mean': decode_mean}}}},
        'gpu': {'0': {'util_mean_pct': 50.0}},
    }


def test_compare_prints_delta_and_percent() -> None:
    table = rep.compare_reports(_report(10.0, 0.2), _report(15.0, 0.1))
    assert '| ingest.images_per_s | 10 | 15 | +5 | +50.0% |' in table
    assert '| metrics.api.histograms.op_x{stage=decode}.mean | 0.2 | 0.1 | -0.1 | -50.0% |' in table


def test_compare_zero_baseline_has_no_percent() -> None:
    table = rep.compare_reports(_report(10.0, 0.2), _report(10.0, 0.2))
    assert '| ingest.failed | 0 | 0 | 0 | n/a |' in table


def test_compare_only_lists_keys_in_both() -> None:
    before = {'ingest': {'a': 1.0, 'only_before': 3.0}}
    after = {'ingest': {'a': 2.0, 'only_after': 4.0}}
    table = rep.compare_reports(before, after)
    assert 'ingest.a' in table
    assert 'only_before' not in table
    assert 'only_after' not in table


def test_markdown_report_has_tables_and_no_paths() -> None:
    report = {
        'schema': 1,
        'manifest': {'name': 'm.txt', 'sha256': 'abc', 'count': 4, 'bytes': 8},
        'config': {'policy': 'all', 'batch_size': 32},
        'ingest': rep.ingest_summary([], wall_s=1.0, input_bytes=0),
        'stages': {'region_drain': {'wall_s': 3.5}},
        'storage': rep.storage_summary(images=0, store_bytes=0, crop_cache_bytes=0),
        'gpu': {'0': {'util_mean_pct': 50.0, 'mem_peak_mb': 10.0, 'samples': 3}},
        'metrics': {
            'api': {
                'histograms': {
                    'op_pipeline_stage_seconds{stage=decode}': {
                        'sum': 3.0,
                        'count': 6.0,
                        'mean': 0.5,
                    }
                },
                'counters': {},
            }
        },
    }
    md = rep.to_markdown(report)
    assert '| images/s |' in md
    assert 'region_drain' in md
    assert 'op_pipeline_stage_seconds{stage=decode}' in md
    assert '/mnt/' not in md
