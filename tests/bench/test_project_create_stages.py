"""The 4D-0 stage table: medians, p95 and the wall rows, from the stage seconds of each run."""

from __future__ import annotations

from scripts.bench.project_create_stages import render_table


def _run(indexes: float, total: float) -> dict:
    return {
        'post_s': 0.05,
        'total_s': total,
        'stages': {'project_create_indexes': indexes, 'project_create_dirs': 0.01},
    }


def test_the_table_has_one_row_per_stage_plus_the_wall_rows() -> None:
    table = render_table([_run(1.0, 2.0), _run(3.0, 4.0), _run(2.0, 3.0)])
    rows = {line.split('|')[1].strip(): line for line in table.splitlines()[2:]}
    assert list(rows) == [
        'project_create_indexes',
        'project_create_dirs',
        'route_202 (POST until 202)',
        'total (POST until job done)',
    ]
    assert '| 2.000 | 3.000 | 3.000 |' in rows['project_create_indexes']
    assert '| 3.000 | 4.000 | 4.000 |' in rows['total (POST until job done)']
