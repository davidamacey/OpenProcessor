"""eval_split: seeded, group-level calibration/heldout halves and the tuning-overlap refusal."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from scripts.bench import eval_split as sp


if TYPE_CHECKING:
    from pathlib import Path


def _groups(n: int = 40) -> list[str]:
    return [f'img{k:03d}.jpg' for k in range(n)]


def test_split_is_a_disjoint_cover_and_deterministic() -> None:
    a = sp.split_groups(_groups(), seed=7)
    b = sp.split_groups(list(reversed(_groups())), seed=7)
    assert a == b  # input order does not matter
    cal, held = a
    assert set(cal).isdisjoint(held)
    assert set(cal) | set(held) == set(_groups())
    assert len(cal) == 20
    assert sp.split_groups(_groups(), seed=8) != a


def test_split_balances_every_stratum() -> None:
    groups = _groups()
    strata = {g: 'rare' if k % 4 == 0 else 'common' for k, g in enumerate(groups)}
    cal, held = sp.split_groups(groups, seed=1, strata=strata)
    for name in ('rare', 'common'):
        c = sum(strata[g] == name for g in cal)
        h = sum(strata[g] == name for g in held)
        assert abs(c - h) <= 1


def test_manifest_round_trips_and_detects_tampering(tmp_path: Path) -> None:
    manifest = sp.build_manifest('coco-test', 3, 'source_image', _groups())
    path = tmp_path / 'm.json'
    sp.write_manifest(path, manifest)
    assert sp.load_manifest(path)['manifest_sha256'] == manifest['manifest_sha256']
    doc = json.loads(path.read_text())
    doc['heldout'].append('sneaked.jpg')
    path.write_text(json.dumps(doc))
    with pytest.raises(sp.SplitError, match='hash'):
        sp.load_manifest(path)


def test_items_follow_their_group_and_unknown_groups_are_refused() -> None:
    manifest = sp.build_manifest('d', 1, 'source_image', ['a', 'b', 'c', 'd'])
    held = set(manifest['heldout'])
    item_group = {'i1': 'a', 'i2': 'a', 'i3': 'b', 'i4': 'c', 'i5': 'd'}
    got = sp.items_in_split(manifest, 'heldout', item_group)
    assert got == {i for i, g in item_group.items() if g in held}
    assert ('i1' in got) == ('i2' in got)  # an image never straddles the halves
    with pytest.raises(sp.SplitError, match='not in the split manifest'):
        sp.items_in_split(manifest, 'heldout', {**item_group, 'i6': 'zzz'})


def test_overlap_between_tuning_and_scoring_groups_is_refused() -> None:
    manifest = sp.build_manifest('d', 1, 'source_image', _groups())
    sp.assert_no_overlap(manifest, 'calibration', manifest, 'heldout')
    with pytest.raises(sp.SplitError, match='overlap'):
        sp.assert_no_overlap(manifest, 'heldout', manifest, 'heldout')
    with pytest.raises(sp.SplitError, match='overlap'):
        sp.assert_no_overlap(manifest, 'full', manifest, 'heldout')
    other = sp.build_manifest('d', 99, 'source_image', _groups())
    with pytest.raises(sp.SplitError, match='overlap'):  # another seed puts some groups on both
        sp.assert_no_overlap(other, 'calibration', manifest, 'calibration')


def test_pin_manifest_groups_near_duplicate_variants_with_their_parent() -> None:
    pin = {
        'images': [
            {'file_name': 'p.jpg', 'group': 'rare'},
            {'file_name': 'q.jpg', 'group': 'similar'},
        ],
        'near_duplicates': [{'parent': 'p.jpg', 'variants': [{'file_name': 'nd_p_1.jpg'}]}],
    }
    groups, strata, variant_parent = sp.groups_from_pin(pin)
    assert groups == ['p.jpg', 'q.jpg']
    assert strata == {'p.jpg': 'rare', 'q.jpg': 'similar'}
    assert variant_parent == {'nd_p_1.jpg': 'p.jpg'}
