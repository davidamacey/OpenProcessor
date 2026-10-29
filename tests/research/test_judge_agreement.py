"""judge_agreement: per-judge metrics against truth, judge-vs-judge kappa, layout calibration."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

from scripts.research import judge_agreement as ja


if TYPE_CHECKING:
    from pathlib import Path


def _truth(
    sid: str, same: bool, odd: list[int], majority: str = 'car', **extra: Any
) -> dict[str, Any]:
    return {
        'sheet_id': sid,
        'layout': 'cluster',
        'case': sid,
        'truth': {'same': same, 'odd_cells': odd, 'majority': majority},
        **extra,
    }


def _rec(
    sid: str, model: str, same: bool, odd: list[int], name: str = 'car', **extra: Any
) -> dict[str, Any]:
    return {
        'sheet_id': sid,
        'model': model,
        'unit': 'cluster',
        'cached': False,
        'error': None,
        'verdict': {'same': same, 'odd_cells': odd, 'name': name, 'confidence': 'high'},
        'input_tokens': 100,
        'output_tokens': 10,
        'latency_ms': 50.0,
        **extra,
    }


def _manifest(sid: str, n: int = 6) -> dict[str, Any]:
    return {
        'sheet_id': sid,
        'layout': 'cluster',
        'cells': [{'cell': k, 'item_id': f'{sid}-{k}', 'box_id': ''} for k in range(1, n + 1)],
    }


def test_cluster_metrics_flag_precision_recall_false_same_and_names() -> None:
    truth = [
        _truth('a', True, []),
        _truth('b', False, [2]),
        _truth('c', False, [1, 3]),
        _truth('d', True, []),
        _truth('e', False, [4]),
    ]
    recs = [
        _rec('a', 'm', True, []),
        _rec('b', 'm', False, [2]),
        _rec('c', 'm', True, [], name='truck'),
        _rec('d', 'm', False, [5], name='cat'),
        _rec('e', 'm', False, [4]),
    ]
    m = ja.cluster_metrics(truth, recs, {s['sheet_id']: _manifest(s['sheet_id']) for s in truth})
    assert m['n'] == 5
    assert m['different']['accuracy'] == pytest.approx(3 / 5)
    assert m['different']['precision'] == pytest.approx(2 / 3)
    assert m['different']['recall'] == pytest.approx(2 / 3)
    assert m['false_same_rate'] == pytest.approx(1 / 3)
    assert m['cell_flags']['precision'] == pytest.approx(2 / 3)  # flagged 2,5,4; true 2,1,3,4
    assert m['cell_flags']['recall'] == pytest.approx(2 / 4)
    assert m['name_match_rate'] == pytest.approx(3 / 5)
    assert len(m['accuracy_wilson95']) == 2


def test_purity_estimate_comes_with_a_cluster_bootstrap_interval() -> None:
    truth = [_truth(f's{k}', k % 2 == 0, [] if k % 2 == 0 else [1]) for k in range(20)]
    recs = [_rec(f's{k}', 'm', k % 2 == 0, [] if k % 2 == 0 else [1]) for k in range(20)]
    mf = {s['sheet_id']: _manifest(s['sheet_id']) for s in truth}
    p = ja.cluster_metrics(truth, recs, mf)['purity']
    assert p['judge'] == pytest.approx(p['truth'])
    assert p['judge_ci95'][0] <= p['judge'] <= p['judge_ci95'][1]


def test_invalid_replies_are_counted_not_scored() -> None:
    truth = [_truth('a', True, []), _truth('b', False, [2])]
    bad = {
        'sheet_id': 'b',
        'model': 'm',
        'unit': 'cluster',
        'verdict': None,
        'error': 'x',
        'cached': False,
    }
    m = ja.cluster_metrics(
        truth, [_rec('a', 'm', True, []), bad], {'a': _manifest('a'), 'b': _manifest('b')}
    )
    assert m['n'] == 1
    assert m['invalid'] == 1
    assert m['missing'] == 0


def test_pair_metrics_exclude_unsure_but_report_it() -> None:
    truth = [
        {
            'sheet_id': 'p1',
            'layout': 'pair',
            'truth': {'rows': {'1': True, '2': False, '3': True, '4': False}},
        }
    ]
    rows = [('yes', 1), ('yes', 2), ('unsure', 3), ('no', 4)]
    rec = {
        'sheet_id': 'p1',
        'model': 'm',
        'unit': 'pair',
        'error': None,
        'cached': False,
        'verdict': {'rows': [{'row': r, 'answer': a} for a, r in rows]},
    }
    m = ja.pair_metrics(truth, [rec])
    assert m['rows_total'] == 4
    assert m['rows_decided'] == 3
    assert m['unsure_rate'] == pytest.approx(0.25)
    assert m['accuracy'] == pytest.approx(2 / 3)
    assert m['false_same_rate'] == pytest.approx(1 / 2)


def test_judge_vs_judge_kappa_and_fleiss_on_common_sheets() -> None:
    truth = [_truth(f's{k}', k % 2 == 0, []) for k in range(8)]
    by_judge = {
        'a': {f's{k}': _rec(f's{k}', 'a', k % 2 == 0, []) for k in range(8)},
        'b': {f's{k}': _rec(f's{k}', 'b', k % 2 == 0, []) for k in range(8)},
        'c': {f's{k}': _rec(f's{k}', 'c', k % 4 == 0, []) for k in range(8)},
    }
    out = ja.judge_agreement(truth, by_judge)
    assert out['pairwise_kappa']['a|b'] == 1.0
    assert out['pairwise_kappa']['a|c'] < 1.0
    assert out['fleiss_kappa'] is not None
    assert out['fleiss_kappa'] < 1.0
    assert out['kappa_with_truth']['a'] == 1.0


def test_layout_recommendation_is_the_smallest_within_005_of_the_best() -> None:
    kappas = {(9, 160): 0.80, (16, 160): 0.86, (24, 160): 0.88, (24, 128): 0.83, (24, 192): 0.87}
    assert ja.recommend_layout(kappas) == (16, 160)  # 0.86 is within 0.05 of 0.88, smaller than 24
    assert ja.recommend_layout({(24, 160): 0.5}) == (24, 160)


def test_calibration_groups_layouts_and_measures_flag_overlap() -> None:
    def sheet(cid: str, grid: str, thumb: int, same: bool) -> dict[str, Any]:
        t = _truth(f'{cid}-{grid}-{thumb}', same, [] if same else [1], grid=grid, thumb_px=thumb)
        return {**t, 'case': cid}

    truth = [sheet(c, g, t, c == 'x') for c in ('x', 'y') for g, t in (('3x3', 160), ('6x4', 160))]
    recs = {
        r['sheet_id']: r
        for r in [
            _rec(f'{c}-{g}-{t}', 'm', c == 'x', [] if c == 'x' else [1])
            for c in ('x', 'y')
            for g, t in (('3x3', 160), ('6x4', 160))
        ]
    }
    mf = {
        s['sheet_id']: {
            **_manifest(s['sheet_id'], 4),
            'cells': [
                {'cell': k, 'item_id': f'{s["case"]}-{k}', 'box_id': ''} for k in range(1, 5)
            ],
        }
        for s in truth
    }
    out = ja.calibration(truth, {'m': recs}, mf)['m']
    assert out['layouts']['3x3@160']['accuracy'] == 1.0
    assert out['layouts']['3x3@160']['kappa_truth'] == 1.0
    assert out['stability']['3x3@160']['same_kappa_vs_reference'] == 1.0
    assert out['reference'] == '6x4@160'
    assert out['stability']['3x3@160']['flag_overlap'] == 1.0


def test_controls_report_flags_a_failing_batch() -> None:
    truth = [
        _truth(f'c{k}', k < 3, [] if k < 3 else [1], control='same' if k < 3 else 'different')
        for k in range(6)
    ]
    good = {f'c{k}': _rec(f'c{k}', 'm', k < 3, [] if k < 3 else [1]) for k in range(6)}
    assert ja.controls_report(truth, good)['batch_ok'] is True
    bad = {f'c{k}': _rec(f'c{k}', 'm', True, []) for k in range(6)}
    rep = ja.controls_report(truth, bad)
    assert rep['false_accept_rate'] == 1.0
    assert rep['batch_ok'] is False


def test_load_runs_keeps_the_last_valid_record_per_sheet(tmp_path: Path) -> None:
    p = tmp_path / 'r.jsonl'
    rows = [
        {'run_id': 'r', 'budget': {}},
        {'sheet_id': 's', 'model': 'm', 'verdict': None, 'error': 'bad', 'unit': 'cluster'},
        _rec('s', 'm', True, []),
        _rec('t', 'm', True, []),
    ]
    p.write_text('\n'.join(json.dumps(r) for r in rows))
    got = ja.load_runs([p])
    assert set(got['m']) == {'s', 't'}
    assert got['m']['s']['error'] is None


def test_disagreements_map_flagged_cells_back_to_item_ids() -> None:
    truth = [_truth('a', False, [2]), _truth('b', True, [])]
    recs = [_rec('a', 'm', False, [3]), _rec('b', 'm', True, [])]
    mf = {s['sheet_id']: _manifest(s['sheet_id']) for s in truth}
    got = ja.disagreements(truth, recs, mf)
    assert got == [
        {
            'sheet_id': 'a',
            'truth_same': False,
            'judge_same': False,
            'missed_items': ['a-2'],
            'false_flags': ['a-3'],
        }
    ]
