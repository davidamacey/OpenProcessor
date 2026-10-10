"""Tests for the pure helpers in scripts/bench/baseline_suite.py."""

from __future__ import annotations

from scripts.bench import baseline_suite as bs
from scripts.bench.baseline_report import metrics_delta


def test_stage_table_joins_seconds_and_bytes() -> None:
    before = {}
    after = {
        'op_pipeline_stage_seconds_sum{stage=decode}': 4.0,
        'op_pipeline_stage_seconds_count{stage=decode}': 8.0,
        'op_pipeline_stage_bytes_total{stage=decode}': 800.0,
        'op_pipeline_stage_seconds_sum{stage=embed}': 3.0,
        'op_pipeline_stage_seconds_count{stage=embed}': 2.0,
    }
    table = bs.stage_table(metrics_delta(before, after, ('op_',)))
    assert table['decode'] == {'calls': 8.0, 'seconds': 4.0, 'mean_ms': 500.0, 'bytes': 800.0}
    assert table['embed']['mean_ms'] == 1500.0


def test_knn_fields_finds_top_level_and_nested_vectors() -> None:
    props = {
        'pe_embedding': {'type': 'knn_vector'},
        'meta': {'type': 'keyword'},
        'region_box_embeddings': {
            'type': 'nested',
            'properties': {'embedding': {'type': 'knn_vector'}, 'box_id': {'type': 'keyword'}},
        },
    }
    assert bs._knn_fields(props) == [
        ('pe_embedding', {'nested': ''}),
        ('region_box_embeddings.embedding', {'nested': 'region_box_embeddings'}),
    ]


def test_chunks() -> None:
    assert [len(c) for c in bs.chunks(list(range(10)), 4)] == [4, 4, 2]
