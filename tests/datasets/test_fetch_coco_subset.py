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
from typing import TYPE_CHECKING

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
