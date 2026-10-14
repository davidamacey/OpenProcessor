"""Generic export: split assignment, holdout, class selection and box normalization."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from curation.export_fakes import generic_service
from src.services.curation.export import _ExportRow, hash_split
from src.services.curation.export_split import stratified_split


def test_hash_split_is_deterministic():
    a = hash_split('crop-1', seed=42, train_ratio=0.8, val_ratio=0.1)
    b = hash_split('crop-1', seed=42, train_ratio=0.8, val_ratio=0.1)
    assert a == b
    assert a in {'train', 'val', 'test'}


def test_hash_split_changes_with_seed_sometimes():
    # Not a strict guarantee for every key, but across many keys at least
    # one seed change should move at least one key to a different bucket.
    keys = [f'crop-{i}' for i in range(200)]
    a = {k: hash_split(k, seed=1, train_ratio=0.8, val_ratio=0.1) for k in keys}
    b = {k: hash_split(k, seed=2, train_ratio=0.8, val_ratio=0.1) for k in keys}
    assert a != b


@pytest.mark.asyncio
async def test_classes_with_objects_excludes_registry_classes_with_no_labeled_crops(tmp_path):
    """E2: class_count is the registry size (data.yaml's nc); a class
    with zero labeled objects in this export must not count as one that
    "has data" -- classes_with_objects is the honest denominator."""
    docs = [
        {
            'crop_id': 'crop-1',
            'image_id': 'img-1',
            'image_path': 'a.jpg',
            'bbox_norm': [0.1, 0.1, 0.5, 0.5],
            'class_id': 0,
            'class_name': 'car',
        },
    ]
    # 'truck' and 'bus' are registered classes with no crops at all.
    service = generic_service(tmp_path, docs, ['car', 'truck', 'bus'])
    result = await service.export_dataset(version_tag='t1', seed=7, copy_images=False)

    assert result.class_count == 3
    assert result.classes_with_objects == 1
    manifest = json.loads(Path(result.manifest_path).read_text())
    assert manifest['class_count'] == 3
    assert manifest['classes_with_objects'] == 1


# ---------------------------------------------------------------------------
# Restored stratified-sampling cases
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_data_yaml_nc_matches_class_count(tmp_path):
    docs = [
        {
            'crop_id': 'crop-1',
            'image_id': 'img-1',
            'image_path': 'a.jpg',
            'bbox_norm': [0.1, 0.1, 0.5, 0.5],
            'class_id': 0,
            'class_name': 'car',
        },
    ]
    service = generic_service(tmp_path, docs, ['car', 'truck', 'bus'])

    result = await service.export_dataset(copy_images=False)

    data_yaml = (Path(result.export_dir) / 'data.yaml').read_text()
    assert 'nc: 3' in data_yaml
    assert result.class_count == 3


@pytest.mark.asyncio
async def test_frozen_holdout_rows_land_in_test_split(tmp_path):
    """``export.py``'s own module docstring advertises this; pin it."""
    docs = [
        {
            'crop_id': f'crop-{i}',
            'image_id': f'img-{i}',
            'image_path': f'{i}.jpg',
            'bbox_norm': [0.0, 0.0, 1.0, 1.0],
            'class_id': 0,
            'class_name': 'car',
            # Every row hashes wherever it hashes EXCEPT this one, which is
            # frozen -- it must land in test regardless of the hash bucket.
            'test_holdout': i == 0,
        }
        for i in range(20)
    ]
    service = generic_service(tmp_path, docs, ['car'])

    result = await service.export_dataset(seed=42, copy_images=False)

    # Files are per source image: crop-0 sits on img-0.
    label_path = Path(result.export_dir) / 'labels' / 'test' / 'img-0.txt'
    assert label_path.is_file()
    assert not list((Path(result.export_dir) / 'labels').glob('*/crop-*.txt'))


@pytest.mark.asyncio
async def test_stratification_distributes_classes_across_splits(tmp_path):
    """Each class's actual per-split ratio should track the target ratio
    closely -- not just "some" items in each split by luck."""
    docs = [
        {
            'crop_id': f'{name}-{i}',
            'image_id': f'{name}-img-{i}',
            'image_path': f'{name}-{i}.jpg',
            'bbox_norm': [0.0, 0.0, 1.0, 1.0],
            'class_id': cls_idx,
            'class_name': name,
        }
        for cls_idx, name in enumerate(['car', 'truck'])
        for i in range(50)
    ]
    service = generic_service(tmp_path, docs, ['car', 'truck'])

    result = await service.export_dataset(seed=1, copy_images=False)

    for name in ('car', 'truck'):
        train_files = list((Path(result.export_dir) / 'labels' / 'train').glob(f'{name}-*.txt'))
        val_files = list((Path(result.export_dir) / 'labels' / 'val').glob(f'{name}-*.txt'))
        test_files = list((Path(result.export_dir) / 'labels' / 'test').glob(f'{name}-*.txt'))
        total = len(train_files) + len(val_files) + len(test_files)
        assert total == 50
        # Target is 0.8/0.1/0.1 -- allow a couple of rows of slack from rounding.
        assert abs(len(train_files) - 40) <= 2
        assert abs(len(val_files) - 5) <= 2
        assert abs(len(test_files) - 5) <= 2


def test_stratified_split_keeps_group_together():
    """Rows sharing a group_key value never straddle a split."""
    rows = [
        _ExportRow(
            item_id=f'crop-{i}',
            image_id='img-shared',  # all ten items were cut from one image
            image_path='shared.jpg',
            bbox_norm=[0.0, 0.0, 1.0, 1.0],
            class_id=0,
            class_name='car',
        )
        for i in range(10)
    ]
    splits = stratified_split(rows, seed=3, train_ratio=0.8, val_ratio=0.1)
    assert len(set(splits.values())) == 1  # every row in the shared group got the same split


def test_region_bbox_round_trip():
    """normalized bbox -> YOLO cx/cy/w/h text -> back to a normalized bbox."""
    from src.services.curation.export_images import normalized_box, yolo_line

    bbox = [0.1, 0.2, 0.5, 0.6]

    def _yolo_to_norm(cx, cy, w, h):
        return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]

    box = normalized_box(bbox)
    assert box is not None
    parts = yolo_line(3, box).split()
    cid = int(parts[0])
    cx, cy, w, h = (float(x) for x in parts[1:])
    recovered = _yolo_to_norm(cx, cy, w, h)
    assert cid == 3
    for a, b in zip(bbox, recovered, strict=True):
        assert a == pytest.approx(b, abs=1e-5)


def test_normalized_box_clamps_to_the_frame_and_rejects_empty_boxes():
    from src.services.curation.export_images import normalized_box

    assert normalized_box([-0.1, 0.2, 0.5, 1.2]) == (0.0, 0.2, 0.5, 1.0)
    assert normalized_box([0.5, 0.2, 0.5, 0.6]) is None  # zero width
    assert normalized_box([0.6, 0.2, 0.5, 0.6]) is None  # inverted
    assert normalized_box([1.1, 0.2, 1.5, 0.6]) is None  # wholly outside the frame
    assert normalized_box([0.1, 0.2, 0.5]) is None
    assert normalized_box(None) is None


@pytest.mark.asyncio
async def test_letterbox_resize_is_refused(tmp_path):
    """Labels are normalized to the source frame; a letterbox pad would
    shift every box against them."""
    service = generic_service(tmp_path, [], ['alpha'])
    with pytest.raises(ValueError, match='letterbox'):
        await service.export_dataset(resize_mode='letterbox')  # type: ignore[arg-type]
