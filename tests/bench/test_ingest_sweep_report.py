"""Tests for scripts/bench/ingest_sweep_report.py."""

from __future__ import annotations

from typing import Any

from scripts.bench import ingest_sweep_report as r
from scripts.bench.suite_lib import summarize


def summary_of(values: dict[str, list[float]]) -> dict[str, Any]:
    return {k: summarize(v) for k, v in values.items()}


def run(images_per_s: float, api_cpu: float) -> dict[str, Any]:
    return {'flat': {'images_per_s': images_per_s, 'api_cpu_s_per_image': api_cpu}}


def test_point_key_orders_numerically() -> None:
    points = ['b32_c4', 'b8_c16', 'b32_c16', 'b32_c1', 'b8_c4']
    assert sorted(points, key=r.point_key) == ['b8_c4', 'b8_c16', 'b32_c1', 'b32_c4', 'b32_c16']


def test_sweep_table_reports_median_range_and_api_cores() -> None:
    runs = [run(20.0, 0.2), run(22.0, 0.2), run(21.0, 0.2)]
    flat = {
        'images_per_s': [20.0, 22.0, 21.0],
        'api_cpu_s_per_image': [0.2, 0.2, 0.2],
        'gpu2_util_mean_pct': [90.0, 92.0, 91.0],
    }
    doc = {
        'sweep': {
            'arms': {
                'shipped': {
                    'points': {'b32_c16': {'upload': {'summary': summary_of(flat), 'runs': runs}}}
                }
            }
        }
    }
    out = r.sweep_table(doc, '2')
    row = out.splitlines()[2]
    assert row.startswith('| 32 x 16 | 21.00 (20.00-22.00) |')  # one arm: no arm column
    assert '| 91 (90-92) |' in row
    assert '| 4.2 |' in row  # 0.2 CPU-s x 21 images/s
    assert 'n/a' in row  # columns the run did not record
    doc['sweep']['arms']['pe=2'] = doc['sweep']['arms']['shipped']
    two = r.sweep_table(doc, '2').splitlines()
    assert two[0].startswith('| Triton arm | batch x clients |')
    assert two[3].startswith('| pe=2 | 32 x 16 |')


def test_opensearch_table_renders_one_row_per_arm() -> None:
    flat = {
        k: [1.0, 2.0, 3.0]
        for k in (
            'docs_per_s',
            'images_per_s_equivalent',
            'searchable_s',
            'bulk_p95_s',
            'bytes_per_doc',
            'node_cpu_cores_during_load',
            'retries_429',
        )
    }
    doc = {
        'arms': {
            'ef128': {
                'config': {
                    'bulk': 1000,
                    'writers': 4,
                    'refresh': 'shipped',
                    'ef': 128,
                    'vectors': 'real',
                },
                'summary': summary_of(flat),
                'runs': [{'flat': {'node_cpu_cores_during_load': 3.0, 'docs_per_s': 1500.0}}],
            }
        }
    }
    row = r.opensearch_table(doc).splitlines()[2]
    assert row.startswith('| ef128 | 1000 x 4 | shipped, ef 128, real |')
    assert '| 2.00 (2.00-2.00) |' in row  # 3 cores for 1,500 docs/s = 2 CPU-s per 1k docs
