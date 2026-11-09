"""``_disk_usage`` reports the bytes under a project's export, upload and run directories."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

from src.services.projects.stats import _disk_usage


if TYPE_CHECKING:
    from pathlib import Path


def test_disk_usage_counts_bytes_per_directory(tmp_path: Path) -> None:
    exports, uploads = tmp_path / 'exports', tmp_path / 'uploads'
    (exports / 'sub').mkdir(parents=True)
    (exports / 'sub' / 'a.bin').write_bytes(b'0123456789')
    (exports / 'b.bin').write_bytes(b'xyz')
    uploads.mkdir()
    resources = SimpleNamespace(export_root=exports, upload_root=uploads)

    usage = _disk_usage('disk-usage-test', resources)

    assert (usage['exports_bytes'], usage['uploads_bytes']) == (13, 0)
    assert 'runs_bytes' not in usage


def test_one_ten_byte_file_is_ten_bytes(tmp_path: Path) -> None:
    (tmp_path / 'f.bin').write_bytes(b'0123456789')
    resources = SimpleNamespace(export_root=tmp_path, upload_root=tmp_path / 'none')

    assert _disk_usage('disk-usage-ten', resources)['exports_bytes'] == 10
