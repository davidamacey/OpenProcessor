"""The serialization micro-benchmark runs offline and its new path matches the old bytes."""

from __future__ import annotations

from scripts.bench import serialization_bench as sb


def test_bench_runs_and_new_path_is_byte_identical() -> None:
    # run() asserts new == old for every case; a small shape keeps this fast.
    rows = sb.run(iterations=2, review_items=5, boxes=20, dim=8)
    cases = {r.case for r in rows}
    assert cases == {'review_page_5', 'box_embeddings_20', 'batch_ingest_20'}
    assert all(r.ms_median > 0 and r.body_bytes > 0 for r in rows)
    table = sb.format_rows(rows)
    assert 'new (WireRoute)' in table
