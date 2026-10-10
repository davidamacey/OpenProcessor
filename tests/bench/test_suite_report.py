"""Tests for scripts/bench/suite_report.py on a small synthetic result document."""

from __future__ import annotations

from typing import Any

from scripts.bench import suite_report as r
from scripts.bench.suite_lib import summarize, summarize_runs


def _run(images_per_s: float) -> dict[str, Any]:
    flat = {
        'images_per_s': images_per_s,
        'wall_s': 2000 / images_per_s,
        'failed': 0.0,
        'items_per_image': 6.4,
        'request_p50_s': 14.0,
        'api_cpu_s_per_image': 0.5,
        'triton_cpu_s_per_image': 0.03,
        'cluster_train_wall_s': 100.0,
        'bytes_per_image': 70000.0,
        'bytes_per_vector': 8800.0,
    }
    return {
        'flat': flat,
        'stages': {
            'decode': {'calls': 2000.0, 'seconds': 10.0, 'mean_ms': 5.0, 'bytes': 2000 * 1024.0}
        },
        'triton': {
            'pe_image_encoder': {
                'inferences': 14000,
                'mean_batch': 20.0,
                'queue_ms_per_request': 300.0,
                'exec_input_ms': 0.5,
                'exec_infer_ms': 250.0,
                'exec_output_ms': 2.0,
                'executions': 700,
            }
        },
        'triton_wire_bytes': {'pe_image_encoder': {'request_bytes': 2000 * 9 * 1024}},
        'gpu': {'0': {'util_mean_pct': 90.0, 'mem_peak_mb': 33000.0}},
        'attribution': {'own_sm_mean_pct': 88.0, 'foreign_sm_mean_pct': 0.0},
    }


def _doc() -> dict[str, Any]:
    runs = [_run(8.9), _run(9.1), _run(8.8)]
    block = {'summary': summarize_runs([x['flat'] for x in runs]), 'runs': runs}
    return {
        'ingest': {'config': {'images': 2000}, 'upload': block, 'batch': block},
        'triton': {
            'pe_image_encoder': {
                '8': {
                    'points': [{'concurrency': 1.0, 'infer_per_sec': 62.0, 'p50_ms': 124.0}],
                    'best': {'concurrency': 1.0, 'infer_per_sec': 62.0},
                }
            }
        },
        'endpoints': {
            'images': 200,
            'detect': {
                'threads_1': {
                    'summary': summarize_runs(
                        [{'images_per_s': 10.0, 'p50_ms': 90.0, 'p95_ms': 170.0}]
                    )
                }
            },
        },
    }


def test_fmt_shows_median_and_range() -> None:
    assert r.fmt(summarize([8.7, 8.9, 9.1]), 1) == '8.9 (8.7-9.1)'
    assert r.fmt(None) == 'n/a'


def test_ingest_headline_has_both_routes_and_the_median() -> None:
    out = r.ingest_headline(_doc())
    assert '`/ingest/upload` | 8.90 (8.80-9.10)' in out
    assert '`/ingest/batch`' in out


def test_stage_table_is_per_image() -> None:
    out = r.ingest_stages(_doc())
    assert '| decode | 1.00 (1.00-1.00) | 5.0 (5.0-5.0) | 5.0 (5.0-5.0) | 1 (1-1) |' in out


def test_models_table_gpu_busy_per_image() -> None:
    out = r.triton_models(_doc())
    assert '`pe_image_encoder` | 7.00 (7.00-7.00) | 20.0 (20.0-20.0)' in out
    assert '87.5 (87.5-87.5)' in out  # 250 ms x 700 executions / 2000 images


def test_storage_ratio_to_reference() -> None:
    assert '1.05x' in r.storage(_doc())


def test_perf_and_endpoint_tables_render() -> None:
    assert '| `pe_image_encoder` | 8 | 62 | 124.0 | 62 | 1 |' in r.perf_points(_doc())
    assert '| `detect` | 1 | 10.0 (10.0-10.0) |' in r.endpoints(_doc())
