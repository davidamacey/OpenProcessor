"""W10.2.1/W10.17: YOLO dataset scanning."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from src.services.curation.dataset_import.scan import FormatUndetectedError
from src.services.curation.dataset_import.yolo import DatasetPathNotAllowedError, scan_yolo


if TYPE_CHECKING:
    from pathlib import Path


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # A byte-valid-enough placeholder; scan_yolo never decodes images.
    path.write_bytes(b'\xff\xd8\xff\xdb' + b'0' * 16)


def _make_dataset(root: Path, *, names: dict[int, str] | list[str] | None = None) -> None:
    names = names if names is not None else ['car', 'truck']
    names_yaml = (
        '\n'.join(f'  {i}: {n}' for i, n in enumerate(names))
        if isinstance(names, list)
        else '\n'.join(f'  {i}: {n}' for i, n in names.items())
    )
    (root / 'data.yaml').write_text(f'train: images/train\nval: images/val\nnames:\n{names_yaml}\n')
    _write_image(root / 'images/train/a.jpg')
    _write_image(root / 'images/train/b.jpg')
    _write_image(root / 'images/val/c.jpg')
    (root / 'labels/train').mkdir(parents=True, exist_ok=True)
    (root / 'labels/val').mkdir(parents=True, exist_ok=True)
    (root / 'labels/train/a.txt').write_text('0 0.5 0.5 0.2 0.2\n')
    (root / 'labels/train/b.txt').write_text('')  # negative
    # c.txt (val) intentionally missing -> label_file_missing


class TestBasicScan:
    def test_splits_and_label_states(self, tmp_path: Path) -> None:
        _make_dataset(tmp_path)
        scan = scan_yolo(tmp_path)
        by_stem = {e.source_stem: e for e in scan.entries}
        assert by_stem['a'].label_state == 'labeled'
        assert by_stem['a'].split == 'train'
        assert len(by_stem['a'].boxes) == 1
        assert by_stem['a'].boxes[0].dataset_class == 'car'
        assert by_stem['b'].label_state == 'negative'
        assert by_stem['c'].label_state == 'unlabeled'
        codes = {i.code for i in scan.issues.issues()}
        assert 'label_file_missing' in codes

    def test_valid_key_stored_as_val(self, tmp_path: Path) -> None:
        (tmp_path / 'data.yaml').write_text('valid: images/val\nnames:\n  0: car\n')
        _write_image(tmp_path / 'images/val/a.jpg')
        scan = scan_yolo(tmp_path)
        assert scan.entries[0].split == 'val'

    def test_names_as_list(self, tmp_path: Path) -> None:
        _make_dataset(tmp_path, names=['car', 'truck'])
        scan = scan_yolo(tmp_path)
        assert scan.class_box_counts == {'car': 1}

    def test_missing_label_option_negative(self, tmp_path: Path) -> None:
        _make_dataset(tmp_path)
        scan = scan_yolo(tmp_path, missing_label='negative')
        by_stem = {e.source_stem: e for e in scan.entries}
        assert by_stem['c'].label_state == 'negative'

    def test_orphan_label_file(self, tmp_path: Path) -> None:
        _make_dataset(tmp_path)
        (tmp_path / 'labels/train/orphan.txt').write_text('0 0.1 0.1 0.1 0.1\n')
        scan = scan_yolo(tmp_path)
        codes = {i.code for i in scan.issues.issues()}
        assert 'label_file_orphan' in codes


class TestRowIssues:
    def _scan_one_label(self, tmp_path: Path, label_line: str, names: list[str] | None = None):
        names = names or ['car', 'truck', 'bus', 'x', 'x', 'x', 'x', 'x']
        names_yaml = '\n'.join(f'  {i}: {n}' for i, n in enumerate(names))
        (tmp_path / 'data.yaml').write_text(f'train: images/train\nnames:\n{names_yaml}\n')
        _write_image(tmp_path / 'images/train/a.jpg')
        (tmp_path / 'labels/train').mkdir(parents=True, exist_ok=True)
        (tmp_path / 'labels/train/a.txt').write_text(label_line + '\n')
        return scan_yolo(tmp_path)

    def test_class_out_of_range(self, tmp_path: Path) -> None:
        scan = self._scan_one_label(tmp_path, '7 0.5 0.5 0.2 0.2', names=['car'])
        codes = {i.code for i in scan.issues.issues()}
        assert 'label_class_out_of_range' in codes
        assert scan.entries[0].boxes == []

    def test_malformed_row(self, tmp_path: Path) -> None:
        scan = self._scan_one_label(tmp_path, '0 0.5 0.5 0.2')
        codes = {i.code for i in scan.issues.issues()}
        assert 'label_row_malformed' in codes

    def test_coords_out_of_range(self, tmp_path: Path) -> None:
        scan = self._scan_one_label(tmp_path, '0 0.5 0.5 1.2 0.2')
        codes = {i.code for i in scan.issues.issues()}
        assert 'label_coords_out_of_range' in codes

    def test_coords_clamped_within_tolerance(self, tmp_path: Path) -> None:
        scan = self._scan_one_label(tmp_path, '0 0.001 0.5 0.003 0.2')
        codes = {i.code for i in scan.issues.issues()}
        assert 'label_coords_clamped' in codes
        assert len(scan.entries[0].boxes) == 1

    def test_polygon_row_becomes_box(self, tmp_path: Path) -> None:
        # A YOLO-seg polygon: cls x1 y1 x2 y2 x3 y3 (7 fields).
        scan = self._scan_one_label(tmp_path, '0 0.1 0.1 0.5 0.1 0.3 0.5')
        codes = {i.code for i in scan.issues.issues()}
        assert 'yolo_polygon_to_box' in codes
        box = scan.entries[0].boxes[0]
        assert box.bbox_norm == pytest.approx((0.1, 0.1, 0.5, 0.5))

    def test_degenerate_box_skipped(self, tmp_path: Path) -> None:
        scan = self._scan_one_label(tmp_path, '0 0.5 0.5 0.0 0.0')
        codes = {i.code for i in scan.issues.issues()}
        assert 'label_box_degenerate' in codes
        assert scan.entries[0].boxes == []


def test_data_yaml_path_escaping_root_rejected(tmp_path: Path) -> None:
    """A ``path:`` (or resolved image) outside the allowed roots is
    ``dataset_path_not_allowed`` via the ``path_guard`` callback."""
    _make_dataset(tmp_path)
    with pytest.raises(DatasetPathNotAllowedError):
        scan_yolo(tmp_path, path_guard=lambda _p: False)


def test_format_undetected_no_yaml(tmp_path: Path) -> None:
    with pytest.raises(FormatUndetectedError):
        scan_yolo(tmp_path)
