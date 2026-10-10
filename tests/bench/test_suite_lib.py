"""Tests for scripts/bench/suite_lib.py: the arithmetic behind every baseline number."""

from __future__ import annotations

import pytest

from scripts.bench import suite_lib as s


def test_summarize_median_and_spread() -> None:
    out = s.summarize([10.0, 12.0, 11.0])
    assert out['median'] == 11.0
    assert out['spread_pct'] == pytest.approx(2 / 11 * 100)
    assert s.summarize([]) == {'n': 0}


def test_summarize_runs_handles_missing_keys() -> None:
    out = s.summarize_runs([{'a': 1.0, 'b': 2.0}, {'a': 3.0}])
    assert out['a']['median'] == 2.0
    assert out['b']['n'] == 1


def _stats(
    inf: int, exe: int, reqs: int, queue_ns: int, batches: dict[int, tuple[int, int]]
) -> dict:
    """Triton stats doc; ``batches`` maps batch size -> (executions, infer ns per execution)."""
    return {
        'model_stats': [
            {
                'name': 'm',
                'inference_count': str(inf),
                'execution_count': str(exe),
                'inference_stats': {
                    'success': {'count': reqs, 'ns': 0},
                    'queue': {'count': reqs, 'ns': queue_ns},
                    'compute_infer': {'count': reqs, 'ns': 999},
                },
                'batch_stats': [
                    {
                        'batch_size': str(k),
                        'compute_input': {'count': n, 'ns': n * 1_000_000},
                        'compute_infer': {'count': n, 'ns': n * ns},
                        'compute_output': {'count': n, 'ns': n * 500_000},
                    }
                    for k, (n, ns) in batches.items()
                ],
            }
        ]
    }


def test_triton_delta_uses_per_execution_batch_stats() -> None:
    before = s.parse_triton_stats(_stats(100, 50, 60, 1_000_000, {2: (50, 4_000_000)}))
    after = s.parse_triton_stats(
        _stats(300, 90, 100, 9_000_000, {2: (50, 4_000_000), 8: (40, 10_000_000)})
    )
    d = s.triton_stats_delta(before, after)['m']
    assert d['inferences'] == 200
    assert d['executions'] == 40
    assert d['requests'] == 40
    assert d['mean_batch'] == 5.0
    assert d['queue_ms_per_request'] == pytest.approx(8_000_000 / 1e6 / 40)
    assert d['exec_infer_ms'] == pytest.approx(10.0)
    assert d['exec_input_ms'] == pytest.approx(1.0)
    assert d['batch_histogram'] == {'8': 40}


def test_triton_delta_drops_idle_models() -> None:
    snap = s.parse_triton_stats(_stats(5, 5, 5, 0, {1: (5, 10)}))
    assert s.triton_stats_delta(snap, snap) == {}


def test_wire_bytes_uses_dtype_and_counts() -> None:
    delta = {'m': {'inferences': 10}}
    cfg = {
        'm': {
            'input': [{'dims': [3, 640, 640], 'data_type': 'TYPE_FP32'}],
            'output': [
                {'dims': [300, 4], 'data_type': 'TYPE_FP16'},
                {'dims': [-1], 'data_type': 'TYPE_INT32'},
            ],
        }
    }
    out = s.triton_wire_bytes(delta, cfg)['m']
    assert out['request_bytes'] == 3 * 640 * 640 * 4 * 10
    assert out['response_bytes'] == (300 * 4 * 2 + 4) * 10


def test_parse_perf_csv_converts_microseconds_to_ms() -> None:
    text = (
        'Concurrency,Inferences/Second,Client Send,Network+Server Send/Recv,Server Queue,'
        'Server Compute Input,Server Compute Infer,Server Compute Output,Client Recv,'
        'p50 latency,p90 latency,p95 latency,p99 latency\n'
        '1,100.5,1,2,500,200,3000,100,1,4000,5000,6000,7000\n'
        '6,300.0,1,2,900,200,6000,100,1,9000,9500,10000,11000\n'
    )
    rows = s.parse_perf_csv(text)
    assert rows[0]['p50_ms'] == 4.0
    assert rows[0]['queue_ms'] == 0.5
    assert s.best_point(rows)['concurrency'] == 6.0


def test_storage_summary_bytes_per_image_and_vector() -> None:
    cat = [
        {'index': 'a__images', 'docs.count': '10', 'docs.deleted': '0', 'pri.store.size': '1000'},
        {'index': 'a__items', 'docs.count': '25', 'docs.deleted': '1', 'pri.store.size': '9000'},
    ]
    out = s.storage_summary(cat, images=10, vectors=40)
    assert out['total_bytes'] == 10000
    assert out['bytes_per_image'] == 1000.0
    assert out['bytes_per_vector'] == 250.0
    assert s.compare_storage({'bytes_per_vector': 250.0}, {'bytes_per_vector': 500.0}) == {
        'bytes_per_vector': 0.5
    }


PMON = """# gpu pid type sm mem enc dec jpg ofa fb ccpm command
# Idx # C/G % % % % % % MB MB name
    0       10     C     40      -      -      -      -      -   100      0    tritonserver
    0       11     C     25      -      -      -      -      -   100      0    tritonserver
    0       12     G      -      -      -      -      -      -     4      0    Xorg
    1       13     C     90      -      -      -      -      -   100      0    python3
"""


def test_pmon_parse_and_foreign_attribution() -> None:
    assert s.parse_pmon(PMON, '0') == {10: 40.0, 11: 25.0}
    out = s.foreign_gpu_summary([{10: 40.0, 11: 25.0}, {10: 10.0}], own_pids={10})
    assert out['foreign_sm_peak_pct'] == 25.0
    assert out['foreign_active_share'] == 0.5
    assert out['own_sm_mean_pct'] == 25.0
    assert s.foreign_gpu_summary([], set()) == {}
