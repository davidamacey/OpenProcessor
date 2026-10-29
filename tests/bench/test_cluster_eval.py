"""cluster_eval.py score: JSONL in, JSON out, on a planted 20k-item partition."""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from scripts.bench import cluster_eval


if TYPE_CHECKING:
    from pathlib import Path


def _write(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    return path


def _run(tmp_path: Path, *extra: str, labels: Path, partition: Path) -> dict[str, Any]:
    out = tmp_path / 'score.json'
    cluster_eval.main(
        ['score', '--labels', str(labels), '--partition', str(partition), '--out', str(out), *extra]
    )
    return json.loads(out.read_text())


def test_score_reports_merge_and_split_separately_and_ignores_unlabelled_rows(
    tmp_path: Path,
) -> None:
    labels = _write(
        tmp_path / 'l.jsonl',
        [{'item_id': f'i{k}', 'label': 'car' if k < 3 else 'bus'} for k in range(6)],
    )
    part = _write(
        tmp_path / 'p.jsonl',
        [
            {'item_id': 'i0', 'cluster_id': 'a'},
            {'item_id': 'i1', 'cluster_id': 'a'},
            {'item_id': 'i2', 'cluster_id': 'b'},
            {'item_id': 'i3', 'cluster_id': 'a'},
            {'item_id': 'i4', 'cluster_id': None},
            {'item_id': 'stray', 'cluster_id': 'a'},
        ],
    )
    s = _run(tmp_path, labels=labels, partition=part)
    assert s['partition_rows_without_label'] == 1
    assert s['unassigned_fraction'] == pytest.approx(2 / 6)  # i4 null, i5 absent
    assert s['false_merge_rate'] == pytest.approx(2 / 3)  # a = {car, car, bus}
    assert s['false_split_rate'] == pytest.approx(5 / 6)
    assert 'strata' not in s
    assert 'stability' not in s


def test_cli_scores_a_planted_20k_partition_with_stability_and_moved_items(
    tmp_path: Path,
) -> None:
    n = 20_000
    rng = np.random.default_rng(3)
    truth = rng.integers(0, 100, n)
    strata = np.where(np.arange(n) < n // 2, 'similar', 'rare')
    protected = np.arange(n) % 10 == 0
    labels = _write(
        tmp_path / 'l.jsonl',
        [
            {
                'item_id': str(i),
                'label': f'c{truth[i]}',
                'stratum': strata[i],
                'protected': bool(protected[i]),
            }
            for i in range(n)
        ],
    )

    def partition(name: str, flip: int) -> Path:
        cl = truth.copy()
        cl[:flip] = (cl[:flip] + 1) % 100  # these items land in the wrong class's cluster
        return _write(
            tmp_path / name,
            [{'item_id': str(i), 'cluster_id': f'k{cl[i]}'} for i in range(n)],
        )

    first, second = partition('a.jsonl', 40), partition('b.jsonl', 0)
    started = time.perf_counter()
    s = _run(
        tmp_path,
        '--second-partition', str(second),
        '--reference', str(second),
        '--runtime-s', '12.5',
        labels=labels,
        partition=first,
    )  # fmt: skip
    assert time.perf_counter() - started < 10
    assert 0 < s['false_split_rate'] < 0.01
    assert 0 < s['false_merge_rate'] < 0.01
    assert s['stability']['identical_grouping'] is False
    assert s['stability']['ari'] < 1.0
    assert set(s['strata']) == {'similar', 'rare'}
    assert s['strata']['rare']['false_merge_rate'] == 0.0
    assert s['protected_items'] == n // 10
    assert s['protected_items_moved'] == 4  # items 0, 10, 20, 30 of the 40 flipped
    assert s['producer_runtime_s'] == 12.5
    assert s['scoring_s'] > 0


def test_identical_reruns_are_reported_identical(tmp_path: Path) -> None:
    rows = [{'item_id': str(i), 'cluster_id': f'k{i % 5}'} for i in range(50)]
    rerun = [{'item_id': r['item_id'], 'cluster_id': f'z{r["cluster_id"]}'} for r in rows]
    labels = _write(
        tmp_path / 'l.jsonl', [{'item_id': str(i), 'label': f'c{i % 5}'} for i in range(50)]
    )
    s = _run(
        tmp_path,
        '--second-partition', str(_write(tmp_path / 'r.jsonl', rerun)),
        labels=labels,
        partition=_write(tmp_path / 'p.jsonl', rows),
    )  # fmt: skip
    assert s['stability'] == {'compared_items': 50, 'ari': 1.0, 'identical_grouping': True}
    assert s['false_merge_rate'] == 0.0


@pytest.mark.parametrize(
    ('rows', 'msg'),
    [
        ([{'item_id': 'a', 'label': 'x'}, {'item_id': 'a', 'label': 'y'}], 'duplicate item_id'),
        ([{'item_id': 'a'}], 'needs a label'),
    ],
)
def test_malformed_labels_fail_closed(tmp_path: Path, rows: list[dict[str, Any]], msg: str) -> None:
    with pytest.raises(SystemExit, match=msg):
        cluster_eval.read_labels(_write(tmp_path / 'l.jsonl', rows))


def test_duplicate_partition_item_fails_closed(tmp_path: Path) -> None:
    part = _write(tmp_path / 'p.jsonl', [{'item_id': 'a', 'cluster_id': 1}] * 2)
    with pytest.raises(SystemExit, match='duplicate item_id'):
        cluster_eval.read_partition(part, ['a'])
