"""visual_judge end to end on a synthetic pool: build, rebuild-verify, judge through files, report."""

from __future__ import annotations

import io
import json
from typing import TYPE_CHECKING

import pytest
from PIL import Image

from scripts.research import judge_agreement as ja, visual_judge as vj
from src.services.labeling import contact_sheet as cs


if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def pool(tmp_path: Path) -> Path:
    rows = []
    k = 0
    for label in ('car', 'truck', 'cat', 'book'):
        for _ in range(40):
            k += 1
            p = tmp_path / 'crops' / f'{k}.png'
            p.parent.mkdir(exist_ok=True)
            buf = io.BytesIO()
            Image.new('RGB', (50, 40), (k * 5 % 256, k * 9 % 256, 70)).save(buf, 'PNG')
            p.write_bytes(buf.getvalue())
            rows.append(
                {'item_id': f'{label}{k}', 'label': label, 'path': str(p), 'group': f'g{k}'}
            )
    out = tmp_path / 'pool.jsonl'
    out.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    return out


def _e0(tmp_path: Path, pool: Path, **extra: str) -> Path:
    out = tmp_path / 'e0'
    args = [
        'make-e0',
        '--pool',
        str(pool),
        '--kind',
        'object',
        '--level',
        'object class',
        '--out',
        str(out),
        '--n-cluster',
        '8',
        '--n-pair',
        '2',
        '--seed',
        '3',
    ]
    for k, v in extra.items():
        args += [f'--{k}', v]
    assert vj.main(args) == 0
    return out


def test_make_e0_writes_sheets_manifests_truth_and_is_reproducible(
    tmp_path: Path, pool: Path
) -> None:
    a = _e0(tmp_path, pool)
    other = tmp_path / 'again'
    vj.main(
        [
            'make-e0',
            '--pool',
            str(pool),
            '--kind',
            'object',
            '--level',
            'object class',
            '--out',
            str(other),
            '--n-cluster',
            '8',
            '--n-pair',
            '2',
            '--seed',
            '3',
        ]
    )
    names = sorted(p.name for p in (a / 'sheets').glob('*.png'))
    assert names == sorted(p.name for p in (other / 'sheets').glob('*.png'))
    assert len([n for n in names if n.startswith('cl')]) >= 8
    assert sum(n.startswith('pr') for n in names) == 2
    for n in names:
        assert (a / 'sheets' / n).read_bytes() == (other / 'sheets' / n).read_bytes()
    truth = [json.loads(x) for x in (a / 'truth.jsonl').read_text().splitlines()]
    assert {t['layout'] for t in truth} == {'cluster', 'pair'}
    assert all(t['truth'] is not None for t in truth)
    spec = json.loads((a / 'spec.json').read_text())
    assert spec['level'] == 'object class'
    assert spec['seed'] == 3


def test_a_reordered_twin_has_the_same_members_in_another_order(tmp_path: Path, pool: Path) -> None:
    out = _e0(tmp_path, pool, **{'n-cluster': '12', 'reorder-frac': '1.0'})
    m = {p.stem: json.loads(p.read_text()) for p in (out / 'manifests').glob('cl*.json')}
    twins = [k for k in m if k.endswith('-r')]
    assert twins
    for t in twins:
        base = m[t[:-2]]
        assert {c['item_id'] for c in m[t]['cells']} == {c['item_id'] for c in base['cells']}
        assert [c['item_id'] for c in m[t]['cells']] != [c['item_id'] for c in base['cells']]


def test_verify_rebuilds_every_sheet_and_catches_a_changed_crop(tmp_path: Path, pool: Path) -> None:
    out = _e0(tmp_path, pool)
    vj.main(['verify', '--sheets', str(out), '--pool', str(pool)])
    victim = json.loads(pool.read_text().splitlines()[0])
    Image.new('RGB', (50, 40), (1, 2, 3)).save(victim['path'], 'PNG')
    with pytest.raises(SystemExit):
        vj.main(['verify', '--sheets', str(out), '--pool', str(pool)])


def test_sheet_file_round_trip_then_report(
    tmp_path: Path, pool: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = _e0(tmp_path, pool)
    ex = tmp_path / 'exchange'
    common = [
        'run',
        '--sheets',
        str(out),
        '--backend',
        'sheet-file',
        '--model',
        'haiku-sub',
        '--exchange-dir',
        str(ex),
        '--max-calls',
        '100',
        '--max-input-tokens',
        '1000000',
        '--run-id',
        'h1',
    ]
    vj.main(common)
    pending = sorted((ex / 'pending').glob('*.png'))
    assert len(pending) == len(list((out / 'sheets').glob('*.png')))
    assert all((ex / 'pending' / f'{p.stem}.prompt.txt').is_file() for p in pending)
    truth = {
        json.loads(x)['sheet_id']: json.loads(x)
        for x in (out / 'truth.jsonl').read_text().splitlines()
    }
    for p in pending:  # an oracle judge answers from the truth: the report must show perfect scores
        t = truth[p.stem]['truth']
        reply = (
            {
                'rows': [
                    {'row': int(r), 'answer': 'yes' if s else 'no'} for r, s in t['rows'].items()
                ]
            }
            if 'rows' in t
            else {
                'same': t['same'],
                'odd_cells': t['odd_cells'],
                'name': t['majority'],
                'confidence': 'high',
            }
        )
        (ex / 'verdicts' / f'{p.stem}.json').write_text(json.dumps(reply))
    vj.main(common)
    runs = ja.load_runs([out / 'judge' / 'h1.jsonl'])
    report = ja.build_report(out, runs)
    j = report['judges']['haiku-sub']
    assert j['cluster']['different']['accuracy'] == 1.0
    assert j['cluster']['name_match_rate'] == 1.0
    assert j['pair']['accuracy'] == 1.0
    budget = json.loads((out / 'judge' / 'budget-sheet-file-haiku-sub.json').read_text())
    assert len(budget['reserved']) == len(pending)  # each sheet reserved once across both runs


def test_run_stops_at_the_call_cap(tmp_path: Path, pool: Path) -> None:
    out = _e0(tmp_path, pool)
    from scripts.research.judge_run import BudgetExceededError

    with pytest.raises(BudgetExceededError):
        vj.main(
            [
                'run',
                '--sheets',
                str(out),
                '--backend',
                'sheet-file',
                '--model',
                'h',
                '--exchange-dir',
                str(tmp_path / 'ex'),
                '--max-calls',
                '3',
                '--max-input-tokens',
                '10000000',
            ]
        )
    assert len(list((tmp_path / 'ex' / 'pending').glob('*.png'))) == 3


def test_calibration_sheets_share_clusters_across_five_layouts(tmp_path: Path, pool: Path) -> None:
    out = tmp_path / 'cal'
    vj.main(
        [
            'make-calibration',
            '--pool',
            str(pool),
            '--kind',
            'object',
            '--level',
            'object class',
            '--out',
            str(out),
            '--n-cluster',
            '3',
            '--seed',
            '1',
        ]
    )
    m = {p.stem: json.loads(p.read_text()) for p in (out / 'manifests').glob('*.json')}
    assert len(m) == 15
    one = {k: v for k, v in m.items() if k.startswith('cal000-')}
    assert {(v['grid'][0], v['thumb_px']) for v in one.values()} == {
        (3, 160),
        (4, 160),
        (6, 160),
        (6, 128),
        (6, 192),
    }
    shown = {k: {c['item_id'] for c in v['cells']} for k, v in one.items()}
    assert len(shown['cal000-3x3-160']) == 9
    assert shown['cal000-3x3-160'] <= shown['cal000-6x4-160'] or len(shown['cal000-6x4-160']) == 24


def test_make_clusters_splits_half_random_and_plants_controls(tmp_path: Path, pool: Path) -> None:
    rows = [json.loads(x) for x in pool.read_text().splitlines()]
    clusters = {
        f'c{n}': {
            'tags': ['borderline'] if n < 8 else [],
            'members': [{**r, 'distance': 0.1 * i} for i, r in enumerate(rows[n * 8 : n * 8 + 8])],
        }
        for n in range(10)
    }
    spec = tmp_path / 'clusters.json'
    spec.write_text(json.dumps({'clusters': clusters}))
    out = tmp_path / 'real'
    vj.main(
        [
            'make-clusters',
            '--clusters',
            str(spec),
            '--kind',
            'object',
            '--level',
            'object class',
            '--out',
            str(out),
            '--n-cluster',
            '6',
            '--controls-pool',
            str(pool),
            '--seed',
            '2',
        ]
    )
    truth = [json.loads(x) for x in (out / 'truth.jsonl').read_text().splitlines()]
    samples = [t['sample'] for t in truth]
    assert samples.count('random') == 3
    assert samples.count('targeted') == 3
    assert samples.count('control') >= 2
    assert {t['control'] for t in truth if t.get('control')} == {'same', 'different'}
    assert json.loads((out / 'spec.json').read_text())['builder_version'] == cs.BUILDER_VERSION


def test_make_search_sheets_put_the_query_first(tmp_path: Path, pool: Path) -> None:
    rows = [json.loads(x) for x in pool.read_text().splitlines()]
    spec = tmp_path / 's.json'
    spec.write_text(
        json.dumps({'queries': [{'id': 'q1', 'query': rows[0], 'results': rows[1:20]}]})
    )
    out = tmp_path / 'search'
    vj.main(
        [
            'make-search',
            '--spec',
            str(spec),
            '--kind',
            'object',
            '--level',
            'object class',
            '--out',
            str(out),
        ]
    )
    manifest = json.loads((out / 'manifests' / 'sq-q1.json').read_text())
    assert manifest['layout'] == 'search'
    assert manifest['cells'][0]['role'] == 'query'
