"""Pure helpers of the text-search latency bench (scripts/bench/retrieval_eval_latency.py)."""

from __future__ import annotations

import pytest

from scripts.bench import retrieval_eval_latency as lat


def test_parse_server_timing_reads_stage_durations() -> None:
    header = 'bootstrap;dur=0.01, encode;dur=12.50, knn;dur=3.25, encode_cache_hit;dur=0'
    assert lat.parse_server_timing(header) == {
        'bootstrap': 0.01,
        'encode': 12.5,
        'knn': 3.25,
        'encode_cache_hit': 0.0,
    }
    assert lat.parse_server_timing('') == {}


def test_pick_bands_takes_the_nearest_share_once_per_class() -> None:
    counts = {'person': 450, 'car': 150, 'dog': 30, 'kite': 5, 'cup': 140}
    bands = lat.pick_bands(counts, total=1000)
    by_band = {b['band']: b['class'] for b in bands}
    assert by_band == {
        'rare': 'kite',
        'about_3pct': 'dog',
        'about_15pct': 'car',
        'about_45pct': 'person',
    }
    assert bands[0]['share'] == pytest.approx(0.005)


def test_pick_bands_refuses_an_empty_project() -> None:
    with pytest.raises(ValueError, match='no embedded items'):
        lat.pick_bands({}, total=0)


def test_summarize_reports_each_stage_and_the_totals() -> None:
    rows = [
        {'wall_ms': 10.0 + i, 'total_ms': 8.0 + i, 'stages': {'knn': float(i), 'encode': 2.0}}
        for i in range(100)
    ]
    out = lat.summarize(rows)
    assert out['knn']['n'] == 100
    assert out['knn']['p99'] >= out['knn']['p50']
    assert out['encode']['p50'] == 2.0
    assert out['client_wall_ms']['max'] == 109.0
