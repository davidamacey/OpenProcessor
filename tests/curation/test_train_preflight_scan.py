"""Tests for src.services.training.preflight_scan (P2-8).

The empty_labels + plate_pairing preflight checks were hardcoded to always
report 'ok' with no scan ever run. These tests build a small temp export
fixture (labels dir + class_registry.json) and exercise the real scan.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.services.training.preflight_scan import scan_export_labels


def _write_registry(export_dir: Path) -> None:
    export_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        'classes': [
            {'class_id': 1, 'class_name': 'sedan'},
            {'class_id': 2, 'class_name': 'suv'},
            {'class_id': 80, 'class_name': 'license_plate'},
        ],
        # dense export ids: sedan=0, suv=1, license_plate=2
        'export_id_map': {'1': 0, '2': 1, '80': 2},
    }
    (export_dir / 'class_registry.json').write_text(json.dumps(payload))
    (export_dir / 'manifest.json').write_text(json.dumps({'dataset_kind': 'vehicle'}))


def _write_label(export_dir: Path, split: str, name: str, lines: list[str]) -> None:
    d = export_dir / 'labels' / split
    d.mkdir(parents=True, exist_ok=True)
    (d / f'{name}.txt').write_text('\n'.join(lines) + ('\n' if lines else ''))


def test_scan_reports_unknown_when_export_dir_missing(tmp_path: Path) -> None:
    result = scan_export_labels(tmp_path / 'does_not_exist')
    assert result.status == 'unknown'
    assert result.reason


def test_scan_counts_empty_labeled_image(tmp_path: Path) -> None:
    export_dir = tmp_path / 'export'
    _write_registry(export_dir)
    _write_label(export_dir, 'train', 'good', ['0 0.5 0.5 0.2 0.2'])
    _write_label(export_dir, 'train', 'empty', [])  # 0 rows -- background image

    result = scan_export_labels(export_dir)
    assert result.status == 'ok'
    assert result.total_images == 2
    assert result.empty_label_images == 1


def test_scan_paired_and_orphan_plate_boxes(tmp_path: Path) -> None:
    export_dir = tmp_path / 'export'
    _write_registry(export_dir)
    # sedan box covering the middle of the image; plate box sits inside it -> paired.
    _write_label(
        export_dir,
        'train',
        'paired',
        ['0 0.5 0.5 0.6 0.6', '2 0.5 0.5 0.1 0.05'],
    )
    # plate box far outside any vehicle box -> orphan.
    _write_label(
        export_dir,
        'train',
        'orphan',
        ['0 0.2 0.2 0.1 0.1', '2 0.9 0.9 0.05 0.03'],
    )

    result = scan_export_labels(export_dir)
    assert result.plate_boxes == 2
    assert result.unpaired_plate_boxes == 1


def test_scan_whole_dataset_empty_reports_empty_for_every_image(tmp_path: Path) -> None:
    export_dir = tmp_path / 'export'
    _write_registry(export_dir)
    _write_label(export_dir, 'train', 'a', [])
    _write_label(export_dir, 'train', 'b', [])

    result = scan_export_labels(export_dir)
    assert result.total_images == 2
    assert result.empty_label_images == 2


def test_scan_include_classes_filter_translates_registry_to_dense_ids(tmp_path: Path) -> None:
    """A subset filter that excludes suv (registry id 2 -> dense id 1) makes
    an suv-only image report as empty, exactly mirroring subset_dataset.py's
    own filtering at train time."""
    export_dir = tmp_path / 'export'
    _write_registry(export_dir)
    _write_label(export_dir, 'train', 'suv_only', ['1 0.5 0.5 0.3 0.3'])  # dense id 1 = suv

    result = scan_export_labels(export_dir, include_classes=[1])  # keep only sedan (registry id 1)
    assert result.total_images == 1
    assert result.empty_label_images == 1


def test_scan_reports_unknown_past_cap(tmp_path: Path) -> None:
    export_dir = tmp_path / 'export'
    _write_registry(export_dir)
    _write_label(export_dir, 'train', 'a', ['0 0.5 0.5 0.2 0.2'])
    _write_label(export_dir, 'train', 'b', ['0 0.5 0.5 0.2 0.2'])

    result = scan_export_labels(export_dir, scan_cap=1)
    assert result.status == 'unknown'
    assert 'cap' in (result.reason or '')


def test_scan_is_cached_per_export_and_manifest(tmp_path: Path, monkeypatch) -> None:
    """A repeat call for the same export dir must not re-scan the label
    files (exports are immutable once written)."""
    from src.services.training import preflight_scan as mod

    export_dir = tmp_path / 'export'
    _write_registry(export_dir)
    _write_label(export_dir, 'train', 'a', ['0 0.5 0.5 0.2 0.2'])

    mod._scan_cache.clear()
    first = scan_export_labels(export_dir)
    calls = []
    real_glob = Path.glob

    def _tracking_glob(self, pattern):
        calls.append((self, pattern))
        return real_glob(self, pattern)

    monkeypatch.setattr(Path, 'glob', _tracking_glob)
    second = scan_export_labels(export_dir)
    assert calls == [], 'cached scan must not touch the filesystem again'
    assert second == first
