"""Unit tests for the source-image prefetch / release helpers.

Kernel page-cache state is hard to observe portably, so these tests
monkeypatch :func:`os.posix_fadvise` and assert the helpers issue the
expected ``WILLNEED`` / ``DONTNEED`` calls without crashing on missing
paths.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import pytest

from src.services.curation import source_image_cache


if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def fadvise_spy(monkeypatch: pytest.MonkeyPatch) -> list[tuple[int, int, int, int]]:
    """Capture (fd, offset, length, advice) tuples passed to os.posix_fadvise."""
    calls: list[tuple[int, int, int, int]] = []

    def _spy(fd: int, offset: int, length: int, advice: int) -> None:
        calls.append((fd, offset, length, advice))

    monkeypatch.setattr(os, 'posix_fadvise', _spy)
    return calls


class TestPrefetchPaths:
    def test_calls_willneed_for_each_existing_path(
        self, tmp_path: Path, fadvise_spy: list[tuple[int, int, int, int]]
    ) -> None:
        files = []
        for i in range(3):
            p = tmp_path / f'img_{i}.bin'
            p.write_bytes(b'abc' * 128)
            files.append(p)

        source_image_cache.prefetch_paths(files)

        assert len(fadvise_spy) == 3
        for fd, offset, length, advice in fadvise_spy:
            assert isinstance(fd, int)
            assert offset == 0
            assert length == 0
            assert advice == os.POSIX_FADV_WILLNEED

    def test_missing_paths_are_logged_and_skipped(
        self,
        tmp_path: Path,
        fadvise_spy: list[tuple[int, int, int, int]],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        existing = tmp_path / 'exists.bin'
        existing.write_bytes(b'x')
        missing = tmp_path / 'nope.bin'

        with caplog.at_level(logging.DEBUG, logger=source_image_cache.logger.name):
            source_image_cache.prefetch_paths([missing, existing])

        # Only the existing file got an fadvise call.
        assert len(fadvise_spy) == 1
        assert fadvise_spy[0][3] == os.POSIX_FADV_WILLNEED
        # The missing one was logged.
        assert any('missing file' in rec.message for rec in caplog.records)

    def test_empty_list_is_a_noop(self, fadvise_spy: list[tuple[int, int, int, int]]) -> None:
        source_image_cache.prefetch_paths([])
        assert fadvise_spy == []

    def test_depth_caps_iteration(
        self, tmp_path: Path, fadvise_spy: list[tuple[int, int, int, int]]
    ) -> None:
        files = []
        for i in range(5):
            p = tmp_path / f'img_{i}.bin'
            p.write_bytes(b'a')
            files.append(p)

        source_image_cache.prefetch_paths(files, depth=2)
        assert len(fadvise_spy) == 2

    def test_zero_depth_raises(self) -> None:
        with pytest.raises(ValueError, match='depth must be positive'):
            source_image_cache.prefetch_paths([], depth=0)


class TestReleaseAfterDecode:
    def test_calls_dontneed_for_existing_path(
        self, tmp_path: Path, fadvise_spy: list[tuple[int, int, int, int]]
    ) -> None:
        p = tmp_path / 'img.bin'
        p.write_bytes(b'abc')

        source_image_cache.release_after_decode(p)

        assert len(fadvise_spy) == 1
        _fd, offset, length, advice = fadvise_spy[0]
        assert offset == 0
        assert length == 0
        assert advice == os.POSIX_FADV_DONTNEED

    def test_accepts_open_fd(
        self, tmp_path: Path, fadvise_spy: list[tuple[int, int, int, int]]
    ) -> None:
        p = tmp_path / 'img.bin'
        p.write_bytes(b'abc')
        fd = os.open(str(p), os.O_RDONLY)
        try:
            source_image_cache.release_after_decode(fd)
        finally:
            os.close(fd)

        assert len(fadvise_spy) == 1
        called_fd, _, _, advice = fadvise_spy[0]
        assert called_fd == fd
        assert advice == os.POSIX_FADV_DONTNEED

    def test_missing_path_is_logged_and_skipped(
        self,
        tmp_path: Path,
        fadvise_spy: list[tuple[int, int, int, int]],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        missing = tmp_path / 'nope.bin'
        with caplog.at_level(logging.DEBUG, logger=source_image_cache.logger.name):
            source_image_cache.release_after_decode(missing)

        assert fadvise_spy == []
        assert any('missing file' in rec.message for rec in caplog.records)
