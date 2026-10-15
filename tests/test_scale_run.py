"""Pure logic of the scale benchmark driver (scripts/bench/scale_run.py)."""

from __future__ import annotations

import pytest

from scripts.bench import scale_run as sr


def test_stop_count_is_absolute_from_final_total() -> None:
    assert sr.stop_count(100_000, 0.5) == 50_000
    assert sr.stop_count(10_001, 0.5) == 5_001


def test_stop_count_stays_inside_the_run() -> None:
    assert sr.stop_count(10, 0.0) == 1
    assert sr.stop_count(10, 1.0) == 9
    with pytest.raises(ValueError, match='at least 2'):
        sr.stop_count(0, 0.5)
    with pytest.raises(ValueError, match='within'):
        sr.stop_count(100, 1.5)


def test_should_cancel_ignores_the_growing_discovery_count() -> None:
    # 7 % of a growing discovery count must not trigger a 50 % stop of a 10k run.
    assert not sr.should_cancel(processed=700, stop_at=5_000, already_cancelled=False)
    assert sr.should_cancel(processed=5_000, stop_at=5_000, already_cancelled=False)
    assert not sr.should_cancel(processed=9_000, stop_at=5_000, already_cancelled=True)


def test_rates() -> None:
    assert sr.cumulative_rate(300, 10.0) == 30.0
    assert sr.cumulative_rate(300, 0.0) == 0.0
    window = [(0.0, 0), (15.0, 450), (30.0, 900)]
    assert sr.recent_rate(window) == 30.0
    assert sr.recent_rate(window[:1]) == 0.0


def test_summarize_combines_legs_and_peaks() -> None:
    legs = [
        {'ingested': 5_000, 'failed': 1, 'wall_s': 100.0},
        {'ingested': 5_000, 'failed': 0, 'wall_s': 120.0},
    ]
    samples = [
        {
            'gpu_util': [10, 80],
            'gpu_mem_mib': [100, 9000],
            'host_cpu_pct': 40.0,
            'host_mem_used_gb': 20.0,
            'os_store_bytes': 1_000,
            'os_segments': 10,
            'os_heap_pct': 30,
        },
        {
            'gpu_util': [20, 95],
            'gpu_mem_mib': [100, 9500],
            'host_cpu_pct': 90.0,
            'host_mem_used_gb': 25.0,
            'os_store_bytes': 4_000,
            'os_segments': 40,
            'os_heap_pct': 55,
        },
    ]
    s = sr.summarize(legs, samples, total=10_000, baseline_img_s=None)
    assert s['wall_s'] == 220.0
    assert s['failed'] == 1
    assert s['images_per_s'] == pytest.approx(10_000 / 220.0)
    assert s['peak_gpu_util'] == 95
    assert s['peak_gpu_mem_mib'] == 9500
    assert s['peak_host_cpu_pct'] == 90.0
    assert s['peak_os_heap_pct'] == 55
    assert s['os_store_bytes'] == 4_000


def test_markdown_table_has_the_run_numbers() -> None:
    s = sr.summarize(
        [{'ingested': 10, 'failed': 0, 'wall_s': 5.0}], [], total=10, baseline_img_s=2.0
    )
    md = sr.render_markdown('lbl', s, cluster_pass_s=42.0, db_bytes=2 * 1024**3)
    assert '| lbl |' in md or 'lbl' in md
    assert '2.00 img/s' in md
    assert '100 % of baseline' in md
    assert '42' in md
    assert '2.00 GiB' in md
