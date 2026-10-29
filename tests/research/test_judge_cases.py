"""judge_cases: synthetic labelled clusters and pairs, sheet truth and the targeted/random split."""

from __future__ import annotations

import io
import random
from typing import TYPE_CHECKING

from PIL import Image

from scripts.research import judge_cases as jc
from src.services.labeling import contact_sheet as cs


if TYPE_CHECKING:
    from pathlib import Path


def _pool(tmp_path: Path, labels: dict[str, int]) -> dict[str, list[jc.PoolItem]]:
    pool: dict[str, list[jc.PoolItem]] = {}
    k = 0
    for label, n in labels.items():
        for _ in range(n):
            k += 1
            p = tmp_path / f'{k}.png'
            buf = io.BytesIO()
            Image.new('RGB', (30, 20), (k * 5 % 256, k * 11 % 256, 90)).save(buf, 'PNG')
            p.write_bytes(buf.getvalue())
            pool.setdefault(label, []).append(jc.PoolItem(f'{label}{k}', label, str(p), f'g{k}'))
    return pool


LABELS = {'car': 40, 'truck': 40, 'cat': 40, 'book': 40}


def test_cluster_case_has_the_requested_contamination_and_far_contaminants(tmp_path: Path) -> None:
    pool = _pool(tmp_path, LABELS)
    case = jc.synthetic_cluster(pool, random.Random(1), case_id='c1', size=24, n_bad=3)
    assert len(case.members) == 24
    assert case.n_bad == 3
    labels = [case.labels[cs.member_key(m.item_id, m.box_id)] for m in case.members]
    assert sum(lab != case.majority for lab in labels) == 3
    assert sum(m.is_prototype for m in case.members) == 3
    bad = [m for m in case.members if case.labels[cs.member_key(m.item_id, '')] != case.majority]
    assert min(m.distance for m in bad if m.distance is not None) > 0.5


def test_clean_case_has_no_contaminants_and_hard_mode_picks_a_confusable_class(
    tmp_path: Path,
) -> None:
    pool = _pool(tmp_path, LABELS)
    clean = jc.synthetic_cluster(pool, random.Random(2), case_id='c', size=12, n_bad=0)
    assert clean.bad_kind == 'none'
    kinds = set()
    for seed in range(40):
        c = jc.synthetic_cluster(
            pool, random.Random(seed), case_id='c', size=12, n_bad=2, hard_fraction=1.0
        )
        other = {v for v in c.labels.values() if v != c.majority}
        if c.majority in ('car', 'truck'):
            assert c.bad_kind == 'hard'
            assert other <= {'car', 'truck'}
        kinds.add(c.bad_kind)
    assert 'hard' in kinds


def test_sheet_truth_uses_only_the_cells_shown(tmp_path: Path) -> None:
    pool = _pool(tmp_path, LABELS)
    case = jc.synthetic_cluster(pool, random.Random(3), case_id='c', size=30, n_bad=4)
    sheet = cs.build_cluster_sheet(case.members, sheet_id='c', seed=3, kind='object', grid=(3, 3))
    truth = jc.sheet_truth(sheet.manifest, case.labels)
    shown_bad = [
        c['cell']
        for c in sheet.manifest['cells']
        if case.labels[cs.member_key(c['item_id'], c['box_id'])] != case.majority
    ]
    assert truth['odd_cells'] == sorted(shown_bad)
    assert truth['same'] == (not shown_bad)
    assert truth['majority'] == case.majority


def test_pair_rows_are_half_same_and_truth_follows_the_swapped_sides(tmp_path: Path) -> None:
    pool = _pool(tmp_path, LABELS)
    pairs, labels = jc.pair_rows(pool, random.Random(4), n_pairs=12)
    assert len(pairs) == 12
    sheet = cs.build_pair_sheet(pairs, sheet_id='p', seed=4, kind='object')
    truth = jc.sheet_truth(sheet.manifest, labels)['rows']
    assert sorted(truth) == list(range(1, 13))
    assert sum(truth.values()) == 6
    for r, same in truth.items():
        a, b = (c for c in sheet.manifest['cells'] if c['row'] == r)
        assert (
            labels[cs.member_key(a['item_id'], '')] == labels[cs.member_key(b['item_id'], '')]
        ) == same


def test_targeted_random_split_is_seeded_disjoint_and_half_uniform() -> None:
    tags = {f'c{k}': ({'borderline'} if k % 3 == 0 else set()) for k in range(60)}
    a = jc.targeted_random_split(tags, 20, seed=1)
    assert a == jc.targeted_random_split(tags, 20, seed=1)
    assert a != jc.targeted_random_split(tags, 20, seed=2)
    rand = [i for i, s in a if s == 'random']
    targ = [i for i, s in a if s == 'targeted']
    assert len(rand) == 10
    assert len(targ) == 10
    assert set(rand).isdisjoint(targ)
    assert all(tags[i] for i in targ)  # targeted picks come from tagged ids only


def test_targeted_half_shrinks_when_few_ids_are_tagged() -> None:
    tags = {f'c{k}': ({'x'} if k == 0 else set()) for k in range(30)}
    got = jc.targeted_random_split(tags, 10, seed=3)
    assert sum(s == 'random' for _, s in got) == 5
    assert sum(s == 'targeted' for _, s in got) <= 1


def test_load_pool_round_trip(tmp_path: Path) -> None:
    (tmp_path / 'pool.jsonl').write_text(
        '{"item_id": "a", "label": "x", "path": "/p/a.png", "bbox_norm": [0, 0, 1, 1]}\n'
        '{"item_id": "b", "label": "x", "path": "/p/b.png"}\n'
    )
    pool = jc.load_pool(tmp_path / 'pool.jsonl')
    assert [i.item_id for i in pool['x']] == ['a', 'b']
    assert pool['x'][0].bbox_norm == (0, 0, 1, 1)
    assert pool['x'][1].bbox_norm is None


def test_load_pool_can_restrict_to_some_labels(tmp_path: Path) -> None:
    (tmp_path / 'pool.jsonl').write_text(
        '{"item_id": "a", "label": "x", "path": "/p/a.png"}\n'
        '{"item_id": "b", "label": "y", "path": "/p/b.png"}\n'
    )
    assert sorted(jc.load_pool(tmp_path / 'pool.jsonl', {'y'})) == ['y']
