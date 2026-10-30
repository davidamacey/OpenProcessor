"""The per-stage table of an ingest profile row (``scripts/bench/ingest_profile_report.py``)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

from scripts.bench import ingest_profile_report as ipr


if TYPE_CHECKING:
    from pathlib import Path


def _report(rate: float, *, jpeg_s: float = 2.0, noisy: bool = False) -> dict[str, Any]:
    return {
        'ingest': {'images': 100, 'items': 235, 'images_per_s': rate},
        'gpu': {'0': {'util_mean_pct': 80.0, 'util_p50_pct': 84.0, 'mem_peak_mb': 1.0}},
        'host': {
            'loadavg_start': [3.0, 2.0, 1.0],
            'loadavg_end': [4.0, 2.0, 1.0],
            'noisy': noisy,
            'api_cpu_s': 5.0,
            'triton_cpu_pct_mean': 250.0,
        },
        'gpu_prep_stages': {
            'decode': {'seconds': 6.0, 'observations': 100.0, 'mean_s': 0.06, 'share': 0.75},
            'crop_jpeg': {'seconds': jpeg_s, 'observations': 100.0, 'mean_s': 0.02, 'share': 0.25},
        },
        'metrics': {
            'api': {
                'histograms': {
                    'op_pipeline_stage_seconds{stage=gpu_decode}': {
                        'sum': 30.0,
                        'count': 100,
                        'mean': 0.3,
                    },
                    'op_pipeline_stage_seconds{stage=index}': {
                        'sum': 10.0,
                        'count': 100,
                        'mean': 0.1,
                    },
                },
                'counters': {},
            },
            'triton': {
                'histograms': {},
                'counters': {
                    'nv_inference_count{gpu_uuid=g,model=pe_image_encoder,version=1}': 400.0,
                    'nv_inference_queue_duration_us{gpu_uuid=g,model=pe_image_encoder,version=1}': 800_000.0,
                    'nv_inference_compute_infer_duration_us{gpu_uuid=g,model=pe_image_encoder,version=1}': 2_000_000.0,
                },
            },
        },
    }


def test_api_stage_rows_are_per_image_with_shares() -> None:
    rows = ipr.api_stage_rows([_report(30.0), _report(31.0)])
    assert rows[0][0] == 'gpu_decode'
    assert rows[0][1] == pytest.approx(0.3)
    assert rows[0][2] == pytest.approx(0.75)
    assert sum(r[2] for r in rows) == pytest.approx(1.0)


def test_triton_rows_are_ms_per_request() -> None:
    ((model, queue, compute),) = ipr.triton_model_rows([_report(30.0)])
    assert model == 'pe_image_encoder'
    assert queue == pytest.approx(2.0)  # 800 ms over 400 requests
    assert compute == pytest.approx(5.0)


def test_prep_rows_take_the_median_over_repetitions() -> None:
    rows = {s: (ms, sh) for s, ms, sh in ipr.prep_stage_rows([_report(1), _report(2), _report(3)])}
    assert rows['decode'] == (pytest.approx(60.0), 0.75)


def test_render_names_every_stage_and_the_headline() -> None:
    text = ipr.render_row('set B', [_report(30.0), _report(32.0, noisy=True), _report(31.0)])
    for needle in ('### set B', 'gpu_decode', 'pe_image_encoder', 'crop_jpeg', 'decode'):
        assert needle in text
    assert '31.00 (30.00 to 32.00)' in text
    assert 'crops per image | 2.35' in text
    assert 'noisy repetitions | 2' in text
    assert 'Triton cores (docker stats) | 2.50' in text
    assert 'API CPU-s per image | 0.0500' in text


def test_load_reps_orders_by_repetition_number(tmp_path: Path) -> None:
    for n in (10, 2, 1):
        (tmp_path / f'row-rep{n}.json').write_text(json.dumps({'n': n}))
    (tmp_path / 'row-summary.json').write_text('{}')
    assert [r['n'] for r in ipr.load_reps(tmp_path)] == [1, 2, 10]


def test_an_empty_row_directory_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        ipr.load_reps(tmp_path)
