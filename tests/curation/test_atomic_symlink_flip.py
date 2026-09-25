"""Tests for ``atomic_symlink_flip`` (F-38, fresh-start E2E findings 2026-09-25).

The ``current`` export symlink dangled on a stock install:
``atomic_symlink_flip`` linked to the export dir's path *verbatim*. With
the default relative ``OP_EXPORT_ROOT=./data/exports``, the target passed
in was the relative path ``data/exports/<timestamp>`` -- but a symlink
target resolves relative to the link's own directory, not the process
cwd, so ``data/exports/current`` pointed at
``data/exports/data/exports/<timestamp>``, which doesn't exist. Every
``GET /export/registry/{artifact}`` 404'd, and the UI's export status /
download buttons broke (F-61).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from src.services.curation.export_support import atomic_symlink_flip


if TYPE_CHECKING:
    import pytest


class TestAtomicSymlinkFlipSiblingTarget:
    def test_relative_sibling_target_resolves_to_the_real_directory(self, tmp_path: Path) -> None:
        """Reproduces F-38: export_root is relative, target is a relative
        sibling path (export_root / timestamp) -- the exact shape
        GenericYoloExportService and SingleClassExportService pass."""
        export_root = tmp_path / 'data' / 'exports'
        export_root.mkdir(parents=True)
        real_export_dir = export_root / '20260925T143527Z'
        real_export_dir.mkdir()
        (real_export_dir / 'manifest.json').write_text('{}')

        current_symlink = export_root / 'current'
        # The relative path exactly as export.py builds it: export_root / name.
        relative_target = export_root / real_export_dir.name
        atomic_symlink_flip(current_symlink, relative_target)

        assert current_symlink.is_symlink()
        resolved = current_symlink.resolve()
        assert resolved == real_export_dir.resolve()
        assert (current_symlink / 'manifest.json').exists()

    def test_sibling_target_link_is_the_bare_name_not_a_nested_path(self, tmp_path: Path) -> None:
        """The fix must not just happen to resolve -- the actual link text
        should be the sibling's bare name, not export_root/name again."""
        export_root = tmp_path / 'data' / 'exports'
        export_root.mkdir(parents=True)
        real_export_dir = export_root / '20260925T143527Z'
        real_export_dir.mkdir()

        current_symlink = export_root / 'current'
        atomic_symlink_flip(current_symlink, export_root / real_export_dir.name)

        link_text = current_symlink.readlink()
        assert str(link_text) == '20260925T143527Z'

    def test_reflip_to_a_new_sibling_replaces_the_old_target(self, tmp_path: Path) -> None:
        export_root = tmp_path / 'data' / 'exports'
        export_root.mkdir(parents=True)
        first = export_root / '20260925T100000Z'
        second = export_root / '20260925T120000Z'
        first.mkdir()
        second.mkdir()

        current_symlink = export_root / 'current'
        atomic_symlink_flip(current_symlink, export_root / first.name)
        atomic_symlink_flip(current_symlink, export_root / second.name)

        assert current_symlink.resolve() == second.resolve()
        # No leftover .tmp staging entries.
        leftovers = [p for p in export_root.iterdir() if p.name.startswith('.current.tmp')]
        assert leftovers == []


class TestAtomicSymlinkFlipRelativeExportRoot:
    def test_relative_op_export_root_does_not_produce_a_dangling_link(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Faithful repro of the reported bug: OP_EXPORT_ROOT is a relative
        path (default './data/exports'), so ``resolved_export_dir`` --
        built as ``self.config.export_root / timestamp`` in export.py --
        is itself a *relative* ``Path``, not absolute. Before the fix,
        ``symlink_to`` wrote that relative string verbatim, and a symlink
        target resolves relative to the *link's own directory* -- so
        ``data/exports/current`` pointed at
        ``data/exports/data/exports/<timestamp>``, which doesn't exist.
        """
        monkeypatch.chdir(tmp_path)
        export_root = Path('data/exports')  # exactly OP_EXPORT_ROOT's default shape
        (tmp_path / export_root).mkdir(parents=True)
        real_export_dir = export_root / '20260925T143527Z'
        (tmp_path / real_export_dir).mkdir()
        (tmp_path / real_export_dir / 'manifest.json').write_text('{}')

        current_symlink = export_root / 'current'
        atomic_symlink_flip(current_symlink, real_export_dir)

        resolved = current_symlink.resolve()
        assert resolved.exists(), f'current symlink dangles, resolved to {resolved}'
        assert resolved == (tmp_path / real_export_dir).resolve()
        assert (current_symlink / 'manifest.json').exists()


class TestAtomicSymlinkFlipNonSiblingTarget:
    def test_target_outside_the_symlinks_directory_is_resolved_absolute(
        self, tmp_path: Path
    ) -> None:
        """An export_dir override (not a sibling of the export root) must
        still produce a working link regardless of the reader's cwd."""
        export_root = tmp_path / 'data' / 'exports'
        export_root.mkdir(parents=True)
        elsewhere = tmp_path / 'elsewhere' / 'custom_export'
        elsewhere.mkdir(parents=True)

        current_symlink = export_root / 'current'
        atomic_symlink_flip(current_symlink, elsewhere)

        assert current_symlink.resolve() == elsewhere.resolve()

        # Absolute, so it stays valid no matter what cwd resolves the link.
        assert current_symlink.readlink().is_absolute()
