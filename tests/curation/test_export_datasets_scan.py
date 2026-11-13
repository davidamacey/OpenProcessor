"""The export datasets on disk are what the dashboard's export step and the Export page count."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import TYPE_CHECKING

from src.services.curation import export_datasets


if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def _export(root: Path, name: str) -> None:
    (root / name).mkdir(parents=True)
    (root / name / 'manifest.json').write_text(json.dumps({'version_tag': name}))


def test_every_manifest_under_the_export_root_is_one_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _export(tmp_path, '20261113T113856Z')
    _export(tmp_path / 'plates', '20261113T120000Z')
    (tmp_path / 'empty_dir').mkdir()
    monkeypatch.setattr(
        export_datasets, 'get_curation_config', lambda: SimpleNamespace(export_root=tmp_path)
    )
    rows = export_datasets.scan_export_datasets()
    assert sorted((r['kind'], r['version_tag']) for r in rows) == [
        ('single_class', '20261113T120000Z'),
        ('yolo', '20261113T113856Z'),
    ]


def test_no_export_root_is_no_dataset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        export_datasets,
        'get_curation_config',
        lambda: SimpleNamespace(export_root=tmp_path / 'missing'),
    )
    assert export_datasets.scan_export_datasets() == []
