"""Tests for scripts/datasets/fetch_coco_subset.py (G-05).

Exercises the pure selection/license-filtering/manifest logic against a
small synthetic COCO-shaped fixture -- never touches the network (the
real annotations zip and image downloads are covered by the fetcher's
own retry/checksum logic in test_common.py, and were run for real as a
one-off smoke test per the task's acceptance criteria, not on every
CI run).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

from scripts.datasets import fetch_coco_subset as f


if TYPE_CHECKING:
    from pathlib import Path


# license ids matching the real COCO licenses[] block, for readability
LIC_BY_NC_SA = 1  # not allowed (NonCommercial)
LIC_BY = 4  # allowed
LIC_BY_SA = 5  # allowed
LIC_NO_KNOWN = 7  # allowed
LIC_USGOV = 8  # allowed


def _image(id_, file_name, license_, w=640, h=480):
    return {
        'id': id_,
        'file_name': file_name,
        'license': license_,
        'width': w,
        'height': h,
        'flickr_url': f'http://example.invalid/{file_name}',
        'coco_url': f'http://images.cocodataset.org/val2017/{file_name}',
    }


def _ann(image_id, category_id, bbox, *, iscrowd=0):
    return {'image_id': image_id, 'category_id': category_id, 'bbox': bbox, 'iscrowd': iscrowd}


# --- license resolution -------------------------------------------------------


def test_resolve_license_names_maps_all_aliases() -> None:
    names = f.resolve_license_names(['by', 'by-sa', 'no-known', 'usgov'])
    assert names == {
        'Attribution License',
        'Attribution-ShareAlike License',
        'No known copyright restrictions',
        'United States Government Work',
    }


def test_resolve_license_names_rejects_unknown_token() -> None:
    with pytest.raises(f.FetchError):
        f.resolve_license_names(['not-a-real-license'])


# --- primary-class + area-threshold logic -------------------------------------


def test_primary_class_picks_largest_qualifying_box() -> None:
    images = [_image(1, 'a.jpg', LIC_BY, w=100, h=100)]
    # car box covers 4% of frame (qualifies), bicycle box covers 0.5% (too small)
    annotations = [
        _ann(1, 3, [0, 0, 20, 20]),  # car: area 400 / 10000 = 4%
        _ann(1, 2, [50, 50, 7, 7]),  # bicycle: area 49 / 10000 = 0.49% -- excluded
    ]
    cat_names: dict[int, str] = {2: 'bicycle', 3: 'car', 17: 'cat'}
    result = f.primary_class_per_image(images, annotations, cat_names, {2, 3, 17})
    assert result == {1: 'car'}


def test_primary_class_ignores_iscrowd_boxes() -> None:
    images = [_image(1, 'a.jpg', LIC_BY, w=100, h=100)]
    annotations = [_ann(1, 3, [0, 0, 50, 50], iscrowd=1)]
    cat_names: dict[int, str] = {2: 'bicycle', 3: 'car', 17: 'cat'}
    assert f.primary_class_per_image(images, annotations, cat_names, {2, 3, 17}) == {}


def test_primary_class_absent_when_no_qualifying_box() -> None:
    images = [_image(1, 'a.jpg', LIC_BY, w=1000, h=1000)]
    annotations = [_ann(1, 3, [0, 0, 5, 5])]  # 25/1e6 = 0.0025% -- too small
    cat_names: dict[int, str] = {2: 'bicycle', 3: 'car', 17: 'cat'}
    assert f.primary_class_per_image(images, annotations, cat_names, {2, 3, 17}) == {}


# --- license filtering on top of primary-class resolution --------------------


def test_eligible_images_filters_by_license_name() -> None:
    images = [
        _image(1, 'allowed.jpg', LIC_BY),
        _image(2, 'blocked.jpg', LIC_BY_NC_SA),
    ]
    primary = {1: 'car', 2: 'car'}
    lic_names = {
        LIC_BY: 'Attribution License',
        LIC_BY_NC_SA: 'Attribution-NonCommercial-ShareAlike License',
    }
    allowed = {'Attribution License'}
    out = f.eligible_images(images, primary, lic_names, allowed, 'val2017')
    assert [r['image_id'] for r in out] == [1]


def test_eligible_images_sorted_by_image_id() -> None:
    images = [_image(3, 'c.jpg', LIC_BY), _image(1, 'a.jpg', LIC_BY)]
    primary = {1: 'car', 3: 'car'}
    lic_names = {LIC_BY: 'Attribution License'}
    out = f.eligible_images(images, primary, lic_names, {'Attribution License'}, 'val2017')
    assert [r['image_id'] for r in out] == [1, 3]


# --- per-class selection: val2017 first, topped up from train2017 ------------


def _pool(image_ids, cls, split):
    return [
        {
            'image_id': i,
            'file_name': f'{i}.jpg',
            'split': split,
            'primary_class': cls,
            'license_name': 'Attribution License',
            'flickr_url': '',
            'coco_url': '',
            'width': 640,
            'height': 480,
        }
        for i in image_ids
    ]


def test_select_per_class_prefers_val_then_tops_up_from_train() -> None:
    val_pool = _pool([1, 2], 'car', 'val2017')
    train_pool = _pool([10, 11, 12], 'car', 'train2017')
    out = f.select_per_class(val_pool, train_pool, ['car'], per_class=4, seed=1)
    assert len(out) == 4
    assert sum(1 for r in out if r['split'] == 'val2017') == 2
    assert sum(1 for r in out if r['split'] == 'train2017') == 2


def test_select_per_class_is_deterministic() -> None:
    val_pool = _pool(range(100), 'car', 'val2017')
    train_pool = _pool(range(100, 200), 'car', 'train2017')
    a = f.select_per_class(val_pool, train_pool, ['car'], per_class=10, seed=7)
    b = f.select_per_class(val_pool, train_pool, ['car'], per_class=10, seed=7)
    assert a == b


def test_select_per_class_warns_but_does_not_fail_when_pool_too_small(caplog) -> None:
    val_pool = _pool([1], 'car', 'val2017')
    train_pool = _pool([10], 'car', 'train2017')
    out = f.select_per_class(val_pool, train_pool, ['car'], per_class=10, seed=1)
    assert len(out) == 2  # takes everything available, no crash


# --- side sets are disjoint from the main selection ---------------------------


def test_select_side_sets_disjoint_from_main() -> None:
    val_pool = _pool(range(50), 'car', 'val2017')
    train_pool = _pool(range(50, 100), 'car', 'train2017')
    main = f.select_per_class(val_pool, train_pool, ['car'], per_class=20, seed=1)
    main_ids = {r['image_id'] for r in main}
    sides = f.select_side_sets(
        val_pool, train_pool, main_ids, {'upload': 5, 'post_promote': 5}, seed=1
    )
    upload_ids = {r['image_id'] for r in sides['upload']}
    promote_ids = {r['image_id'] for r in sides['post_promote']}
    assert not (upload_ids & main_ids)
    assert not (promote_ids & main_ids)
    assert not (upload_ids & promote_ids)
    assert len(upload_ids) == 5
    assert len(promote_ids) == 5


# --- manifest pin: write-once, then verify --------------------------------


def test_verify_or_write_manifest_writes_pin_on_first_run(tmp_path: Path) -> None:
    manifest = tmp_path / 'pin.json'
    selected = _pool([1, 2], 'car', 'val2017')
    f.verify_or_write_manifest(manifest, selected)
    assert manifest.is_file()
    pinned = json.loads(manifest.read_text())
    assert [p['image_id'] for p in pinned] == [1, 2]


def test_verify_or_write_manifest_passes_when_selection_matches_pin(tmp_path: Path) -> None:
    manifest = tmp_path / 'pin.json'
    selected = _pool([1, 2], 'car', 'val2017')
    f.verify_or_write_manifest(manifest, selected)
    # Re-running with the identical selection must not raise.
    f.verify_or_write_manifest(manifest, selected)


def test_verify_or_write_manifest_raises_on_drift(tmp_path: Path) -> None:
    manifest = tmp_path / 'pin.json'
    f.verify_or_write_manifest(manifest, _pool([1, 2], 'car', 'val2017'))
    with pytest.raises(f.FetchError, match='selection drift'):
        f.verify_or_write_manifest(manifest, _pool([1, 3], 'car', 'val2017'))


# --- CLI parsing ---------------------------------------------------------


def test_parse_side_sets() -> None:
    assert f.parse_side_sets('upload=12,post_promote=24') == {'upload': 12, 'post_promote': 24}
    assert f.parse_side_sets('') == {}


def test_parse_side_sets_rejects_malformed_entry() -> None:
    with pytest.raises(f.FetchError):
        f.parse_side_sets('upload=notanumber')


# --- --negatives / --val-only --------------------------------------------------

_LICENSES: list[dict[str, Any]] = [
    {'id': LIC_BY_NC_SA, 'name': 'Attribution-NonCommercial-ShareAlike License'},
    {'id': LIC_BY, 'name': 'Attribution License'},
]
_CATEGORIES: list[dict[str, Any]] = [{'id': 3, 'name': 'car'}, {'id': 1, 'name': 'person'}]


def _annotation_split(first_id: int, *, with_cars: int, empty: int, person_only: int):
    images, anns = [], []
    nxt = first_id
    for _ in range(with_cars):
        images.append(_image(nxt, f'{nxt}.jpg', LIC_BY))
        anns.append(_ann(nxt, 3, [10, 10, 300, 300]))
        nxt += 1
    for _ in range(person_only):
        images.append(_image(nxt, f'{nxt}.jpg', LIC_BY))
        anns.append(_ann(nxt, 1, [10, 10, 300, 300]))
        nxt += 1
    for i in range(empty):
        # Every other empty frame carries a disallowed license.
        images.append(_image(nxt, f'{nxt}.jpg', LIC_BY if i % 2 == 0 else LIC_BY_NC_SA))
        nxt += 1
    return images, anns


def test_select_negatives_excludes_target_class_frames_and_bad_licenses() -> None:
    images, anns = _annotation_split(1, with_cars=2, empty=6, person_only=2)
    lic = {lic['id']: lic['name'] for lic in _LICENSES}
    out = f.select_negatives(
        images, anns, {3}, lic, {'Attribution License'}, 'val2017', 10, 7, set()
    )
    ids = {r['image_id'] for r in out}
    # cars 1-2 carry the target class; person-only frames 3-4 do not, so they
    # are negatives for ``car`` (full annotations kept); of empties 5..10 only
    # the BY ones (5, 7, 9) survive the license filter.
    assert ids == {3, 4, 5, 7, 9}
    assert all(r['primary_class'] == '' for r in out)


def test_select_negatives_is_seeded_and_capped() -> None:
    images, anns = _annotation_split(1, with_cars=0, empty=40, person_only=0)
    lic = {lic['id']: lic['name'] for lic in _LICENSES}
    kw: dict[str, Any] = {
        'target_class_ids': {3},
        'license_id_to_name': lic,
        'allowed_license_names': {'Attribution License'},
    }
    a = f.select_negatives(images, anns, split='val2017', n=5, seed=3, exclude_ids=set(), **kw)
    b = f.select_negatives(images, anns, split='val2017', n=5, seed=3, exclude_ids=set(), **kw)
    c = f.select_negatives(images, anns, split='val2017', n=5, seed=4, exclude_ids=set(), **kw)
    assert a == b
    assert len(a) == 5
    assert a != c


def _stub_annotations(monkeypatch: pytest.MonkeyPatch) -> None:
    val_images, val_anns = _annotation_split(1, with_cars=3, empty=6, person_only=1)
    train_images, train_anns = _annotation_split(1000, with_cars=5, empty=0, person_only=0)
    val = {
        'images': val_images,
        'annotations': val_anns,
        'categories': _CATEGORIES,
        'licenses': _LICENSES,
    }
    train = {
        'images': train_images,
        'annotations': train_anns,
        'categories': _CATEGORIES,
        'licenses': _LICENSES,
    }
    monkeypatch.setattr(f, 'ensure_annotations', lambda _cache: (val, train))


def _args(tmp_path: Path, *extra: str):
    return f.build_parser().parse_args(
        [
            '--out',
            str(tmp_path / 'out'),
            '--classes',
            'car',
            '--per-class',
            '6',
            '--licenses',
            'by',
            '--skip-download',
            *extra,
        ]
    )


def test_run_val_only_never_tops_up_from_train(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_annotations(monkeypatch)
    topped = f.run(_args(tmp_path))
    assert topped['n_selected'] == 6  # 3 val + 3 train
    val_only = f.run(_args(tmp_path, '--val-only'))
    assert val_only['n_selected'] == 3
    assert val_only['val_only'] is True


def test_run_negatives_written_with_attribution_and_pinned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_annotations(monkeypatch)
    manifest = tmp_path / 'm' / 'pin.json'
    summary = f.run(_args(tmp_path, '--val-only', '--negatives', '2', '--manifest', str(manifest)))
    assert summary['n_negatives'] == 2
    neg_csv = (tmp_path / 'out' / 'ATTRIBUTION.csv').read_text()
    assert neg_csv.count('Attribution License') == 3 + 2  # 3 cars + 2 negatives
    pinned = json.loads(manifest.read_text())
    assert [r['primary_class'] for r in pinned].count('') == 2
    gt = json.loads((tmp_path / 'out' / 'coco_gt.json').read_text())
    neg_ids = set(summary['negative_image_ids'])
    assert neg_ids <= {i['id'] for i in gt['images']}
    assert not [a for a in gt['annotations'] if a['image_id'] in neg_ids and a['category_id'] == 3]
    # a second run with the same seed must verify cleanly against the pin
    f.run(_args(tmp_path, '--val-only', '--negatives', '2', '--manifest', str(manifest)))
