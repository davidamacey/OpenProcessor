"""A dataset is untrusted input: malformed or hostile ``data.yaml`` /
COCO JSON / label files / path references become issues, never exceptions
and never a read outside the allowed roots."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
from PIL import Image

from src.services.curation.dataset_import.coco import CocoAnnotationFile, scan_coco
from src.services.curation.dataset_import.paths import dataset_path_guard, resolve_ref
from src.services.curation.dataset_import.yolo import scan_yolo


if TYPE_CHECKING:
    from pathlib import Path


def _image(path: Path, size: tuple[int, int] = (100, 100)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new('RGB', size, color='red').save(path, format='JPEG')


def _codes(scan) -> set[str]:
    return {i.code for i in scan.issues.issues()}


def _yolo(root: Path, yaml_text: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / 'data.yaml').write_text(yaml_text)
    _image(root / 'images/train/a.jpg')
    (root / 'labels/train').mkdir(parents=True, exist_ok=True)
    (root / 'labels/train/a.txt').write_text('0 0.5 0.5 0.2 0.2\n')


@pytest.mark.parametrize(
    'yaml_text',
    [
        'train: images/train\nnames: [unterminated\n',
        '- just\n- a\n- list\n',
        'train: images/train\nnames: car\n',
        'train: images/train\nnames: {a: car}\n',
        'train: images/train\nnames: {-1: car}\n',
        'train: images/train\nnc: banana\nnames: [car]\n',
        'train: images/train\nnames: [car]\nnc: 7\n',
        '!!python/object/apply:os.system ["true"]\n',
    ],
)
def test_malformed_data_yaml_is_a_blocking_issue_not_an_exception(
    tmp_path: Path, yaml_text: str
) -> None:
    _yolo(tmp_path, yaml_text)
    scan = scan_yolo(tmp_path)
    assert scan.entries == []
    assert _codes(scan) & {'data_yaml_invalid', 'data_yaml_names_missing'}


def test_oversize_data_yaml_is_refused_unparsed(tmp_path: Path) -> None:
    _yolo(tmp_path, 'train: images/train\nnames: [car]\n' + '# pad\n' * 300_000)
    scan = scan_yolo(tmp_path)
    assert scan.entries == []
    assert 'data_yaml_invalid' in _codes(scan)


def test_pruned_data_yaml_nc_equal_to_max_index_plus_one_is_accepted(tmp_path: Path) -> None:
    """Review R2-m2: ``nc: 3`` with ``names: {0: car, 2: truck}`` is what a
    pruned export writes; refusing it rejected a legitimate dataset."""
    _yolo(tmp_path, 'train: images/train\nnc: 3\nnames:\n  0: car\n  2: truck\n')
    scan = scan_yolo(tmp_path)
    assert 'data_yaml_invalid' not in _codes(scan)
    assert [b.dataset_class for b in scan.entries[0].boxes] == ['car']


def test_nc_matching_neither_count_is_still_refused(tmp_path: Path) -> None:
    _yolo(tmp_path, 'train: images/train\nnc: 5\nnames:\n  0: car\n  2: truck\n')
    assert 'data_yaml_invalid' in _codes(scan_yolo(tmp_path))


def test_yaml_split_entry_outside_the_allowed_roots_is_not_read(tmp_path: Path) -> None:
    allowed = tmp_path / 'allowed'
    outside = tmp_path / 'outside'
    _image(outside / 'secret.jpg')
    _yolo(allowed, f'train: {outside}\nnames: [car]\n')
    guard = dataset_path_guard(allowed)
    scan = scan_yolo(allowed, path_guard=guard)
    assert scan.entries == []
    assert 'dataset_path_not_allowed' in _codes(scan)


def test_list_file_naming_a_file_outside_the_roots_is_skipped(tmp_path: Path) -> None:
    allowed = tmp_path / 'allowed'
    outside = tmp_path / 'outside'
    _image(outside / 'secret.jpg')
    _image(allowed / 'images/ok.jpg')
    (allowed / 'train.txt').write_text(f'images/ok.jpg\n{outside / "secret.jpg"}\n')
    (allowed / 'data.yaml').write_text('train: train.txt\nnames: [car]\n')
    scan = scan_yolo(allowed, path_guard=dataset_path_guard(allowed))
    assert [e.source_stem for e in scan.entries] == ['ok']
    assert 'image_path_not_servable' in _codes(scan)


def test_symlinked_label_pointing_outside_the_roots_is_never_opened(tmp_path: Path) -> None:
    allowed = tmp_path / 'allowed'
    outside = tmp_path / 'outside.txt'
    outside.write_text('0 0.5 0.5 0.2 0.2\n')
    _yolo(allowed, 'train: images/train\nnames: [car]\n')
    label = allowed / 'labels/train/a.txt'
    label.unlink()
    label.symlink_to(outside)
    scan = scan_yolo(allowed, path_guard=dataset_path_guard(allowed))
    assert scan.entries[0].boxes == []
    assert 'dataset_path_not_allowed' in _codes(scan)


def test_symlinked_image_pointing_outside_the_roots_is_skipped(tmp_path: Path) -> None:
    allowed = tmp_path / 'allowed'
    outside = tmp_path / 'outside'
    _image(outside / 'secret.jpg')
    _yolo(allowed, 'train: images/train\nnames: [car]\n')
    link = allowed / 'images/train/link.jpg'
    link.symlink_to(outside / 'secret.jpg')
    scan = scan_yolo(allowed, path_guard=dataset_path_guard(allowed))
    assert [e.source_stem for e in scan.entries] == ['a']
    assert 'image_path_not_servable' in _codes(scan)


def test_oversize_label_file_is_refused_unparsed(tmp_path: Path) -> None:
    _yolo(tmp_path, 'train: images/train\nnames: [car]\n')
    (tmp_path / 'labels/train/a.txt').write_text('0 0.5 0.5 0.2 0.2\n' * 600_000)
    scan = scan_yolo(tmp_path)
    assert scan.entries[0].boxes == []
    assert 'dataset_file_too_large' in _codes(scan)


def test_dataset_over_the_file_cap_is_refused(tmp_path: Path, monkeypatch) -> None:
    _yolo(tmp_path, 'train: images/train\nnames: [car]\n')
    _image(tmp_path / 'images/train/b.jpg')
    monkeypatch.setenv('OP_DATASET_PREVIEW_MAX_FILES', '1')
    scan = scan_yolo(tmp_path)
    assert scan.entries == []
    assert 'dataset_too_large' in _codes(scan)


# --------------------------------------------------------------------- COCO


def _coco(tmp_path: Path, data) -> list[CocoAnnotationFile]:
    ann = tmp_path / 'instances_train.json'
    ann.write_text(json.dumps(data) if not isinstance(data, str) else data)
    return [CocoAnnotationFile(path=ann, images_dir=tmp_path)]


@pytest.mark.parametrize(
    'data',
    [
        '{not json',
        '[1, 2, 3]',
        {'images': {}, 'annotations': [], 'categories': []},
        {'images': [], 'annotations': [], 'categories': 'car'},
        {'images': [], 'annotations': []},
        '[' * 100_000 + ']' * 100_000,
    ],
    ids=['truncated', 'array', 'images-not-list', 'categories-str', 'no-categories', 'nested-100k'],
)
def test_malformed_coco_json_is_a_blocking_issue(tmp_path: Path, data) -> None:
    scan = scan_coco(_coco(tmp_path, data))
    assert scan.entries == []
    assert 'coco_json_invalid' in _codes(scan)


def test_coco_entries_with_the_wrong_shape_never_raise(tmp_path: Path) -> None:
    _image(tmp_path / 'a.jpg')
    data = {
        'images': [
            {'id': 1, 'file_name': 'a.jpg', 'width': 100, 'height': 100},
            'not an object',
            {'id': 2},
            {'id': 3, 'file_name': 5},
        ],
        'annotations': [
            {'image_id': 1, 'category_id': 10, 'bbox': [10, 10, 20, 20]},
            {'image_id': 1, 'category_id': 10, 'bbox': 'wide'},
            {'image_id': 1, 'category_id': 10, 'bbox': [1, 2, 3]},
            {'image_id': 1, 'category_id': 99, 'bbox': [1, 2, 3, 4]},
            {'image_id': 1, 'category_id': 10, 'bbox': ['a', 'b', 'c', 'd']},
            'not an object',
            {'category_id': 10},
        ],
        'categories': [{'id': 10, 'name': 'car'}, {'id': 11}, 'bad', {'name': 'x'}],
    }
    scan = scan_coco(_coco(tmp_path, data))
    assert [e.rel_path for e in scan.entries] == ['a.jpg']
    assert [b.dataset_class for b in scan.entries[0].boxes] == ['car']
    assert {'coco_json_invalid', 'coco_bbox_out_of_image'} <= _codes(scan)


@pytest.mark.parametrize(
    'file_name', ['../outside.jpg', '/etc/passwd', 'sub/../../outside.jpg', 'a\x00.jpg', '~/x.jpg']
)
def test_coco_file_name_cannot_leave_the_images_dir(tmp_path: Path, file_name: str) -> None:
    root = tmp_path / 'ds'
    root.mkdir()
    _image(tmp_path / 'outside.jpg')
    data = {
        'images': [{'id': 1, 'file_name': file_name, 'width': 100, 'height': 100}],
        'annotations': [],
        'categories': [{'id': 10, 'name': 'car'}],
    }
    ann = root / 'instances_train.json'
    ann.write_text(json.dumps(data))
    scan = scan_coco([CocoAnnotationFile(path=ann, images_dir=root)])
    assert scan.entries == []
    assert 'image_path_not_servable' in _codes(scan)


def test_coco_symlinked_image_out_of_the_images_dir_is_skipped(tmp_path: Path) -> None:
    root = tmp_path / 'ds'
    root.mkdir()
    _image(tmp_path / 'outside.jpg')
    (root / 'link.jpg').symlink_to(tmp_path / 'outside.jpg')
    data = {
        'images': [{'id': 1, 'file_name': 'link.jpg', 'width': 100, 'height': 100}],
        'annotations': [],
        'categories': [{'id': 10, 'name': 'car'}],
    }
    ann = root / 'instances_train.json'
    ann.write_text(json.dumps(data))
    scan = scan_coco([CocoAnnotationFile(path=ann, images_dir=root)])
    assert scan.entries == []


def test_coco_declared_size_that_disagrees_with_the_image_skips_it(tmp_path: Path) -> None:
    _image(tmp_path / 'a.jpg', size=(100, 50))
    data = {
        'images': [{'id': 1, 'file_name': 'a.jpg', 'width': 50, 'height': 100}],
        'annotations': [{'image_id': 1, 'category_id': 10, 'bbox': [1, 1, 5, 5]}],
        'categories': [{'id': 10, 'name': 'car'}],
    }
    scan = scan_coco(_coco(tmp_path, data))
    assert scan.entries == []
    assert 'image_size_mismatch' in _codes(scan)


def test_oversize_coco_json_is_refused_unread(tmp_path: Path, monkeypatch) -> None:
    import src.services.curation.dataset_import.coco as coco_mod

    monkeypatch.setattr(coco_mod, 'MAX_COCO_JSON_BYTES', 10)
    scan = scan_coco(_coco(tmp_path, {'images': [], 'annotations': [], 'categories': []}))
    assert 'dataset_file_too_large' in _codes(scan)


# --------------------------------------------------------------- path policy


@pytest.mark.parametrize(
    'ref', ['', '/abs', '~/home', '../up', 'a/../../up', 'a\x00b', '..\\up', '\\\\host\\share']
)
def test_resolve_ref_rejects_every_escape(tmp_path: Path, ref: str) -> None:
    assert resolve_ref(tmp_path, ref) is None


def test_resolve_ref_accepts_a_contained_reference_and_rejects_a_symlink_out(
    tmp_path: Path,
) -> None:
    (tmp_path / 'in').mkdir()
    (tmp_path / 'in/ok.jpg').write_bytes(b'x')
    (tmp_path / 'in/escape').symlink_to(tmp_path.parent)
    assert resolve_ref(tmp_path, 'in/ok.jpg') == (tmp_path / 'in/ok.jpg').resolve()
    assert resolve_ref(tmp_path, 'in/escape/anything') is None


@pytest.mark.parametrize(
    'row',
    ['0 nan 0.5 0.2 0.2', '0 0.5 0.5 inf 0.2', '0 0.1 0.1 nan 0.9 0.9 0.2 0.9'],
    ids=['nan-center', 'inf-width', 'nan-polygon-vertex'],
)
def test_a_non_finite_yolo_coordinate_is_a_row_issue_not_a_box(tmp_path: Path, row: str) -> None:
    _yolo(tmp_path, 'train: images/train\nnames: [car]\n')
    (tmp_path / 'labels/train/a.txt').write_text(f'{row}\n0 0.5 0.5 0.2 0.2\n')
    scan = scan_yolo(tmp_path)
    assert [b.dataset_class for b in scan.entries[0].boxes] == ['car']
    assert 'label_row_malformed' in _codes(scan)


@pytest.mark.parametrize(
    'bbox', ['[NaN, 0, 5, 5]', '[0, 0, Infinity, 5]'], ids=['nan-x', 'inf-width']
)
def test_a_non_finite_coco_bbox_is_a_row_issue_not_a_box(tmp_path: Path, bbox: str) -> None:
    _image(tmp_path / 'a.jpg')
    text = (
        '{"images": [{"id": 1, "file_name": "a.jpg", "width": 100, "height": 100}],'
        '"categories": [{"id": 10, "name": "car"}],'
        f'"annotations": [{{"image_id": 1, "category_id": 10, "bbox": {bbox}}},'
        '{"image_id": 1, "category_id": 10, "bbox": [10, 10, 20, 20]}]}'
    )
    scan = scan_coco(_coco(tmp_path, text))
    assert [b.dataset_class for b in scan.entries[0].boxes] == ['car']
    assert 'coco_bbox_out_of_image' in _codes(scan)
