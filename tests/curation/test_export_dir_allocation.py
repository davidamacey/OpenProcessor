"""Found live: two exports started in the same second shared one
auto-named directory, so the second overwrote the first's artifacts."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from src.services.curation.export_retention import EXPORT_DIR_RE
from src.services.curation.export_support import allocate_export_dir


if TYPE_CHECKING:
    from pathlib import Path


def test_same_second_exports_get_distinct_directories(tmp_path: Path) -> None:
    now = datetime(2026, 10, 2, 1, 54, 34, tzinfo=UTC)

    first = allocate_export_dir(tmp_path, now)
    second = allocate_export_dir(tmp_path, now)

    assert first != second
    assert first.is_dir()
    assert second.is_dir()
    assert first.name == '20261002T015434Z'
    assert all(EXPORT_DIR_RE.match(d.name) for d in (first, second))


def test_a_leftover_directory_is_never_reused(tmp_path: Path) -> None:
    now = datetime(2026, 10, 2, 1, 54, 34, tzinfo=UTC)
    (tmp_path / '20261002T015434Z').mkdir()
    (tmp_path / '20261002T015435Z').mkdir()

    assert allocate_export_dir(tmp_path, now).name == '20261002T015436Z'
