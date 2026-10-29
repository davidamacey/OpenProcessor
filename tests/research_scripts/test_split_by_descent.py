"""Y4: descent purity on a hand-built dendrogram (arithmetic in the comments)."""

from __future__ import annotations

import numpy as np

from scripts.research.cluster_diag import descent as dn


def _z() -> np.ndarray:
    # 8 leaves 0..7. node 8=(0,1) h.1; 9=(2,3) h.1; 10=(4,5) h.1; 11=(6,7) h.1;
    # 12=(8,9) h.2 (leaves 0-3); 13=(10,11) h.2 (leaves 4-7); 14=(12,13) h.3 (all 8).
    return np.array(
        [
            [0, 1, 0.1, 2],
            [2, 3, 0.1, 2],
            [4, 5, 0.1, 2],
            [6, 7, 0.1, 2],
            [8, 9, 0.2, 4],
            [10, 11, 0.2, 4],
            [12, 13, 0.3, 8],
        ],
        dtype=float,
    )


def test_descend_replaces_internal_nodes_by_their_children_and_keeps_leaves() -> None:
    z = _z()
    assert dn.descend(z, 14, 1) == [12, 13]
    assert dn.descend(z, 14, 2) == [8, 9, 10, 11]
    assert dn.descend(z, 8, 2) == [0, 1]  # one level makes leaves; the second leaves them alone
    assert sorted(dn.leaves_of(z, 12)) == [0, 1, 2, 3]


def test_purity_is_item_weighted_over_labelled_members() -> None:
    labels = dict(enumerate('aabbbbbb'))
    # whole cluster: majority b = 6 of 8 = 0.75; level 1: {0-3: a,a,b,b -> 2}, {4-7: 4} -> 6/8
    z = _z()
    assert dn.weighted_purity([dn.leaves_of(z, 14)], labels) == 0.75
    assert dn.weighted_purity([dn.leaves_of(z, p) for p in dn.descend(z, 14, 1)], labels) == 0.75
    # level 2: {0,1}: a,a -> 2; {2,3}: b,b -> 2; {4,5}: 2; {6,7}: 2 -> 8/8
    assert dn.weighted_purity([dn.leaves_of(z, p) for p in dn.descend(z, 14, 2)], labels) == 1.0
    assert dn.weighted_purity([[0]], {}) is None  # nothing labelled


def test_judge_key_flags_impure_clusters_and_finds_the_level_that_separates() -> None:
    labels = dict(enumerate('aabbbbbb'))
    rows = dn.judge_key(_z(), 0.35, labels)  # cut above every merge: one cluster of 8
    assert len(rows) == 1
    row = rows[0]
    assert row['labelled'] == 8
    assert row['majority_share'] == 0.75
    assert row['impure']
    assert row['purity_level_1'] == 0.75
    assert row['purity_level_2'] == 1.0
    assert row['reaches'] is True
    summary = dn.summarise(rows)
    assert summary['share_reaching'] == 1.0
    assert summary['reach_at_level_1'] == 0
    assert summary['verdict'] == 'PASS'


def test_a_pure_or_tiny_cluster_is_not_an_impure_case() -> None:
    labels = dict(enumerate('aaaaaaaa'))
    rows = dn.judge_key(_z(), 0.35, labels)
    assert rows[0]['impure'] is False
    assert dn.summarise(rows)['verdict'] == 'INCONCLUSIVE'  # no impure cluster to judge
    # cut 0.15: four clusters of 2 labelled members each, under the 5-member minimum
    assert dn.judge_key(_z(), 0.15, dict(enumerate('abababab'))) == []


def test_verdict_fails_when_most_impure_clusters_stay_impure() -> None:
    # labels alternate a,b within every pair, so no descent to level 2 separates anything
    labels = dict(enumerate('abababab'))
    rows = dn.judge_key(_z(), 0.35, labels)
    assert rows[0]['purity_level_2'] == 0.5
    assert dn.summarise(rows)['verdict'] == 'FAIL'
