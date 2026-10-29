"""cluster_eval.py score: JSONL in, JSON out, on a planted 20k-item partition."""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from scripts.bench import cluster_eval, eval_split, scorecard


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


# --- BCubed/AMI rows, view and size pair strata, identity labels, splits, scorecard ---


def _two_class_set(tmp_path: Path, **extra: Any) -> tuple[Path, Path]:
    """Six items from three source images; cluster 'a' mixes one bus into the cars."""
    groups = ['g0', 'g0', 'g1', 'g1', 'g2', 'g2']
    labels = _write(
        tmp_path / 'l.jsonl',
        [
            {
                'item_id': f'i{k}',
                'label': 'car' if k < 3 else 'bus',
                'group': groups[k],
                'view': 'front' if k % 2 else 'side',
                'size_bin': 'big' if k < 4 else 'small',
                'identity': f'veh{k // 2}',
                **{key: val[k] for key, val in extra.items()},
            }
            for k in range(6)
        ],
    )
    part = _write(
        tmp_path / 'p.jsonl',
        [
            {'item_id': f'i{k}', 'cluster_id': c}
            for k, c in enumerate(['a', 'a', 'a', 'a', 'b', None])
        ],
    )
    return labels, part


def test_score_adds_bcubed_ami_risk_and_label_field_blocks(tmp_path: Path) -> None:
    labels, part = _two_class_set(tmp_path)
    s = _run(tmp_path, labels=labels, partition=part)
    # cluster a = {car, car, car, bus}; b = {bus}; i5 abstained (a singleton for BCubed)
    assert s['bcubed_precision'] == pytest.approx((3 * 3 / 4 + 1 / 4 + 1 + 1) / 6)
    assert s['bcubed_recall'] == pytest.approx((3 * 3 / 3 + 1 / 3 + 1 / 3 + 1 / 3) / 6)
    assert 0.0 < s['ami'] < 1.0
    assert s['risk_coverage']['coverage'] == pytest.approx(5 / 6)
    assert s['risk_coverage']['risk'] == pytest.approx(1 / 5)
    assert set(s['view_pairs']) == {'same', 'different'}
    assert set(s['size_pairs']) == {'same', 'different'}
    # identity labels are scored on their own, never folded into the class metrics
    assert s['identity']['pair_recall'] == pytest.approx(2 / 3)  # (i4,i5) lost to the abstention
    assert s['identity']['pair_precision'] == pytest.approx(2 / 6)
    assert 'bcubed_f' in s['identity']


def test_score_without_optional_label_fields_has_no_such_blocks(tmp_path: Path) -> None:
    labels = _write(
        tmp_path / 'l.jsonl', [{'item_id': 'a', 'label': 'x'}, {'item_id': 'b', 'label': 'x'}]
    )
    part = _write(
        tmp_path / 'p.jsonl', [{'item_id': 'a', 'cluster_id': 1}, {'item_id': 'b', 'cluster_id': 1}]
    )
    s = _run(tmp_path, labels=labels, partition=part)
    for absent in ('view_pairs', 'size_pairs', 'identity'):
        assert absent not in s
    assert s['bcubed_f'] == 1.0


def test_confidence_in_the_partition_gives_a_risk_coverage_curve(tmp_path: Path) -> None:
    labels, _ = _two_class_set(tmp_path)
    part = _write(
        tmp_path / 'pc.jsonl',
        [
            {'item_id': f'i{k}', 'cluster_id': c, 'confidence': conf}
            for k, (c, conf) in enumerate(
                [('a', 0.9), ('a', 0.8), ('a', 0.7), ('a', 0.1), ('b', 0.6), (None, None)]
            )
        ],
    )
    s = _run(tmp_path, labels=labels, partition=part)
    curve = s['risk_coverage']['curve']
    assert curve[-1] == [pytest.approx(5 / 6), pytest.approx(1 / 5)]
    assert curve[0] == [pytest.approx(1 / 6), 0.0]
    assert s['risk_coverage']['aurc'] is not None


def _split(tmp_path: Path, groups: list[str], seed: int = 1) -> tuple[Path, dict[str, Any]]:
    manifest = eval_split.build_manifest('toy', seed, 'source_image', groups)
    path = tmp_path / 'split.json'
    eval_split.write_manifest(path, manifest)
    return path, manifest


def test_split_flags_score_only_the_chosen_half(tmp_path: Path) -> None:
    labels, part = _two_class_set(tmp_path)
    path, manifest = _split(tmp_path, ['g0', 'g1', 'g2'])
    half = set(manifest['heldout'])
    s = _run(
        tmp_path, '--split-manifest', str(path), '--split', 'heldout', labels=labels, partition=part
    )
    assert s['items'] == 2 * len(half)
    assert s['split'] == {'name': 'heldout', 'manifest_sha256': manifest['manifest_sha256']}
    full = _run(tmp_path, labels=labels, partition=part)
    assert full['items'] == 6
    assert 'split' not in full


def test_split_needs_a_group_on_every_label(tmp_path: Path) -> None:
    labels = _write(tmp_path / 'l.jsonl', [{'item_id': 'a', 'label': 'x'}])
    part = _write(tmp_path / 'p.jsonl', [{'item_id': 'a', 'cluster_id': 1}])
    path, _ = _split(tmp_path, ['g0', 'g1'])
    with pytest.raises(SystemExit, match='group'):
        _run(
            tmp_path,
            '--split-manifest',
            str(path),
            '--split',
            'heldout',
            labels=labels,
            partition=part,
        )


def test_scoring_on_what_was_tuned_on_is_refused(tmp_path: Path) -> None:
    labels, part = _two_class_set(tmp_path)
    path, _ = _split(tmp_path, ['g0', 'g1', 'g2'])
    args = ['--split-manifest', str(path), '--split', 'heldout', '--tuning-manifest', str(path)]
    with pytest.raises(SystemExit, match='overlap'):
        _run(tmp_path, *args, '--tuning-split', 'heldout', labels=labels, partition=part)
    with pytest.raises(SystemExit, match='overlap'):
        _run(tmp_path, *args, '--tuning-split', 'full', labels=labels, partition=part)
    ok = _run(tmp_path, *args, '--tuning-split', 'calibration', labels=labels, partition=part)
    assert ok['tuning'] == {
        'split': 'calibration',
        'manifest_sha256': _split(tmp_path, ['g0', 'g1', 'g2'])[1]['manifest_sha256'],
    }


def test_scorecard_out_writes_the_envelope_with_acceptance(tmp_path: Path) -> None:
    labels, part = _two_class_set(tmp_path)
    path, manifest = _split(tmp_path, ['g0', 'g1', 'g2'])
    card_path = tmp_path / 'card.json'
    _run(
        tmp_path,
        '--split-manifest', str(path), '--split', 'heldout',
        '--scorecard-out', str(card_path), '--kind', 'K2', '--level', 'items',
        '--dataset-name', 'toy', '--seed', '5', '--git-sha', 'abc123',
        '--bar', 'bcubed_precision=0.01',
        labels=labels, partition=part,
    )  # fmt: skip
    card = json.loads(card_path.read_text())
    assert card['schema_version'] == scorecard.SCHEMA_VERSION
    assert card['dataset'] == {
        'name': 'toy',
        'synthetic': False,
        'split': 'heldout',
        'manifest_sha256': manifest['manifest_sha256'],
    }
    assert card['run_meta']['git_sha'] == 'abc123'
    assert card['run_meta']['seed'] == 5
    assert card['metrics']['bcubed_precision'] is not None
    assert card['acceptance']['pass'] is True
    assert card['acceptance']['value'] == card['metrics']['bcubed_precision']


def test_a_bar_on_a_calibration_run_is_refused(tmp_path: Path) -> None:
    labels, part = _two_class_set(tmp_path)
    path, _ = _split(tmp_path, ['g0', 'g1', 'g2'])
    with pytest.raises(scorecard.ScorecardError, match='heldout'):
        _run(
            tmp_path,
            '--split-manifest', str(path), '--split', 'calibration',
            '--scorecard-out', str(tmp_path / 'c.json'), '--kind', 'K2', '--level', 'items',
            '--dataset-name', 'toy', '--seed', '1', '--git-sha', 'x',
            '--bar', 'bcubed_precision=0.5',
            labels=labels, partition=part,
        )  # fmt: skip
