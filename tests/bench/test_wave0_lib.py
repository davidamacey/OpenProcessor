"""Tests for scripts/bench/wave0_lib.py: the arithmetic behind the Wave 0 tables."""

from __future__ import annotations

import pytest

from scripts.bench import wave0_lib as w


PE_CONFIG = """name: "pe_image_encoder"
max_batch_size: 32
dynamic_batching {
  preferred_batch_size: [ 4, 8, 16, 32 ]
  max_queue_delay_microseconds: 15000
}

instance_group [
  {
    count: 1
    kind: KIND_GPU
    gpus: [ 0 ]
  }
]
"""


def test_rewrite_instance_group_changes_only_that_block() -> None:
    out = w.rewrite_instance_group(PE_CONFIG, count=3, gpus=[0, 1])
    assert 'count: 3' in out
    assert 'gpus: [ 0, 1 ]' in out
    assert 'max_batch_size: 32' in out
    assert 'max_queue_delay_microseconds: 15000' in out


def test_rewrite_instance_group_leaves_a_count_elsewhere_alone() -> None:
    text = 'input [ { dims: [ 3 ] count: 9 } ]\n' + PE_CONFIG
    out = w.rewrite_instance_group(text, count=2)
    assert 'count: 9' in out
    assert out.count('count: 2') == 1


def test_rewrite_instance_group_requires_a_block() -> None:
    with pytest.raises(ValueError, match='no instance_group'):
        w.rewrite_instance_group('name: "m"\n', count=2)


def test_rewrite_batching_delay_and_preferred() -> None:
    out = w.rewrite_batching(PE_CONFIG, queue_delay_us=2000, drop_preferred=True)
    assert 'max_queue_delay_microseconds: 2000' in out
    assert 'preferred_batch_size' not in out
    assert 'dynamic_batching {' in out


def test_parse_arm_and_points() -> None:
    assert w.parse_arm('pe=2, det=3') == {'pe': 2, 'det': 3}
    assert w.parse_arm('') == {}
    with pytest.raises(ValueError, match='name=count'):
        w.parse_arm('pe')
    assert w.parse_points('32x4,8x16') == [(32, 4), (8, 16)]
    with pytest.raises(ValueError, match='<batch>x<concurrency>'):
        w.parse_points('32')


def test_host_busy_pct_counts_idle_and_iowait_as_not_busy() -> None:
    before = w.parse_proc_stat('cpu  100 0 100 700 100 0 0 0 0 0\n')
    after = w.parse_proc_stat('cpu  200 0 200 800 100 0 0 0 0 0\n')
    assert before == (200, 1000)
    assert w.host_busy_pct(before, after) == pytest.approx(200 / 300 * 100)
    assert w.host_busy_pct(before, before) == 0.0


def test_triton_gpu_busy_is_execution_time_over_wall() -> None:
    delta = {
        'pe': {'exec_infer_ms': 100.0, 'executions': 50},
        'det': {'exec_infer_ms': 40.0, 'executions': 25},
    }
    out = w.triton_gpu_busy(delta, wall_s=10.0)
    assert out['busy_pe'] == pytest.approx(0.5)
    assert out['busy_det'] == pytest.approx(0.1)
    assert out['busy_total'] == pytest.approx(0.6)


def test_parse_clock_line_and_summary() -> None:
    row = w.parse_clock_line('2, 1650, 281.5, 0x0000000000000004')
    assert row == {'gpu': 2.0, 'sm_mhz': 1650.0, 'power_w': 281.5, 'throttled': 1.0}
    idle = w.parse_clock_line('0, 210, 20.1, 0x0000000000000001')
    assert idle is not None
    assert idle['throttled'] == 0.0
    assert w.parse_clock_line('garbage') is None
    summary = w.clock_summary([row, idle])  # type: ignore[list-item]
    assert summary['sm_mhz_min'] == 210.0
    assert summary['throttled_share'] == 0.5


def test_aggregate_collapsed_buckets_and_inclusive_share() -> None:
    lines = [
        'main (a.py:1);handler (b.py:2);resize (cv2:0) 6',
        'main (a.py:1);handler (b.py:2);send (tritonclient/grpc/_client.py:9) 3',
        'main (a.py:1);wait (asyncio/base_events.py:5) 1',
        'bad line without a count',
    ]
    out = w.aggregate_collapsed(lines)
    assert out['samples'] == 10
    assert out['leaf_bucket_share']['image_decode_encode'] == pytest.approx(0.6)
    assert out['leaf_bucket_share']['triton_client_grpc'] == pytest.approx(0.3)
    assert out['leaf_bucket_share']['event_loop_wait'] == pytest.approx(0.1)
    inclusive = dict(out['top_inclusive'])
    assert inclusive['handler (b.py:2)'] == pytest.approx(0.9)
    assert inclusive['main (a.py:1)'] == pytest.approx(1.0)


def test_aggregate_collapsed_empty() -> None:
    assert w.aggregate_collapsed([])['samples'] == 0


def test_apply_arm_maps_keys_to_model_knobs() -> None:
    from scripts.bench import wave0_suite as suite
    from scripts.bench.wave0_probes import MODEL_KEYS

    calls: list[tuple[str, dict]] = []

    class Fake:
        def apply(self, model: str, **opts: object) -> None:
            calls.append((model, opts))

    suite.apply_arm(Fake(), {'pe': 2, 'pe_delay': 2000, 'det_nopref': 1})  # type: ignore[arg-type]
    by_model = dict(calls)
    assert by_model[MODEL_KEYS['pe']] == {'count': 2, 'queue_delay_us': 2000}
    assert by_model[MODEL_KEYS['det']] == {'drop_preferred': True}
    with pytest.raises(ValueError, match='unknown arm knob'):
        suite.apply_arm(Fake(), {'pe_bogus': 1})  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ('gpu', 'host', 'api', 'expected'),
    [
        (95.0, 20.0, 1.0, 'gpu'),
        (60.0, 20.0, 31.0, 'api-cpu'),
        (60.0, 95.0, 5.0, 'host-cpu'),
        (60.0, 30.0, 5.0, 'latency/feeding'),
    ],
)
def test_binding_resource(gpu: float, host: float, api: float, expected: str) -> None:
    got = w.binding_resource(
        gpu_busy_pct=gpu, host_busy_pct_=host, api_cores=api, api_cores_available=32.0
    )
    assert got == expected
