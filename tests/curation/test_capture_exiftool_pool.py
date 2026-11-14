"""Reading capture tags from bytes without a whole-file temporary file, in a pool of ExifTool
processes (the batch ingest reads many photos at once).

``exiftool -stay_open True -@ -`` takes its commands on standard input, so the photo cannot also
arrive there (https://exiftool.org/exiftool_pod.html, ``-stay_open``). Each process is handed the
bytes through a named pipe instead (a private ``0700`` directory, one pipe per process), written
by a short-lived thread that ends quietly when ExifTool stops reading (``-fast`` stops after the
metadata segments).

Every file is generated (Pillow writes it, the real ``exiftool`` writes the tags), plus the
decoder-coverage fixtures (``scripts/research/gpu_pipeline/exif_fixtures.py`` makes them) when
``OP_EXIF_FIXTURES`` names their directory. The reference is the answer of the path reader,
the reader the bytes route used before it (an extensionless temporary file), minus ``SourceFile`` (the one key that names the
file).
"""

from __future__ import annotations

import asyncio
import io
import os
import signal
import stat
import subprocess
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from PIL import Image

from src.clients.exiftool_client import ExifToolClient, ExifToolError
from src.domain.capture_metadata import parse_capture


if TYPE_CHECKING:
    from collections.abc import Iterator

FIXTURES = Path(os.environ.get('OP_EXIF_FIXTURES', ''))
TAGS = (
    '-Make=Acme',
    '-Model=Cam1',
    '-DateTimeOriginal=2024:03:05 14:22:11',
    '-OffsetTimeOriginal=+01:00',
    '-GPSLatitude=47.5',
    '-GPSLatitudeRef=N',
    '-GPSLongitude=8.25',
    '-GPSLongitudeRef=E',
    '-Orientation#=6',
)
# One file per container family the ingest meets; TIFF and the ISO-BMFF family seek inside the file.
FORMATS = ('jpeg', 'jpeg_big_metadata', 'png', 'tiff', 'webp', 'avif', 'gif')


@pytest.fixture
def client() -> Iterator[ExifToolClient]:
    c = ExifToolClient(timeout_s=20)
    yield c
    c.close()


def _make(directory: Path, kind: str) -> Path:
    ext = {'jpeg_big_metadata': 'jpg'}.get(kind, kind)
    path = directory / f'{kind}.{ext}'
    image = Image.new('RGB', (64, 48), (200, 10, 10))
    image.save(path)
    if kind == 'gif':
        return path
    args = list(TAGS)
    if kind == 'jpeg_big_metadata':
        thumb = directory / 'thumb.jpg'
        Image.new('RGB', (160, 120), (1, 2, 3)).save(thumb, quality=100)
        args += [f'-ThumbnailImage<={thumb}', '-XMP-dc:Description=' + 'x' * 40_000]
    subprocess.run(
        ['exiftool', '-overwrite_original', '-q', *args, str(path)], check=True, timeout=60
    )
    return path


def _same(client: ExifToolClient, path: Path) -> None:
    # The bytes route never had a file name: the reference is the same bytes in an extensionless
    # file (ExifTool takes the extension as a hint, so a corrupt file answers differently).
    bare = path.parent / 'bare'
    bare.write_bytes(path.read_bytes())
    by_path = client.read(bare)
    by_bytes = client.read_bytes(path.read_bytes())
    by_path.pop('SourceFile', None)
    by_bytes.pop('SourceFile', None)
    assert by_bytes == by_path
    assert parse_capture(by_bytes) == parse_capture(by_path)


@pytest.mark.parametrize('kind', FORMATS)
def test_bytes_give_the_tags_of_the_file(client: ExifToolClient, tmp_path: Path, kind: str) -> None:
    path = _make(tmp_path, kind)
    _same(client, path)
    expected = 'absent' if kind == 'gif' else 'ok'
    assert parse_capture(client.read_bytes(path.read_bytes())).exif_status == expected


@pytest.mark.skipif(
    not FIXTURES.is_dir(), reason='OP_EXIF_FIXTURES names no decoder-coverage fixture directory'
)
def test_bytes_give_the_tags_of_every_fixture_file(client: ExifToolClient) -> None:
    files = sorted(FIXTURES.iterdir())
    assert files
    for path in files:
        _same(client, path)


def test_a_corrupt_and_a_truncated_file_answer_as_the_path_reader_does(
    client: ExifToolClient, tmp_path: Path
) -> None:
    good = _make(tmp_path, 'jpeg').read_bytes()
    for name, data in (
        ('bad', b'\xff\xd8garbage'),
        ('cut', good[: len(good) // 3]),
        ('empty', b''),
    ):
        path = tmp_path / f'{name}.jpg'
        path.write_bytes(data)
        _same(client, path)


def test_no_whole_file_temporary_file_is_made(
    client: ExifToolClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tempfile

    def refuse(*_a: Any, **_k: Any) -> Any:
        raise AssertionError('a whole-file temporary file was made')

    monkeypatch.setattr(tempfile, 'NamedTemporaryFile', refuse)
    monkeypatch.setattr(tempfile, 'TemporaryFile', refuse)
    monkeypatch.setattr(tempfile, 'SpooledTemporaryFile', refuse)
    monkeypatch.setattr(tempfile, 'mkstemp', refuse)
    path = _make(tmp_path, 'jpeg')
    assert parse_capture(client.read_bytes(path.read_bytes())).camera_make == 'Acme'


def test_a_large_file_that_exiftool_stops_reading_early_ends_without_a_stuck_writer(
    client: ExifToolClient, tmp_path: Path
) -> None:
    path = _make(tmp_path, 'jpeg')
    data = path.read_bytes() + b'\0' * (24 * 1024 * 1024)  # trailing bytes -fast never reads
    client.read_bytes(data)  # starts the process
    before = threading.active_count()
    started = time.monotonic()
    for _ in range(5):
        assert parse_capture(client.read_bytes(data)).camera_make == 'Acme'
    assert time.monotonic() - started < 10
    assert threading.active_count() <= before
    assert not [t for t in threading.enumerate() if t.name.startswith('exiftool-feed')]


def test_the_pipe_lives_in_a_private_directory_removed_at_close(tmp_path: Path) -> None:
    c = ExifToolClient(timeout_s=20)
    c.read_bytes(_make(tmp_path, 'jpeg').read_bytes())
    pipe = c._pipe_path
    assert pipe is not None
    assert stat.S_ISFIFO(pipe.stat().st_mode)
    assert stat.S_IMODE(pipe.parent.stat().st_mode) == 0o700
    c.close()
    assert not pipe.parent.exists()


def test_a_killed_process_is_replaced_on_the_next_bytes_read(
    client: ExifToolClient, tmp_path: Path
) -> None:
    data = _make(tmp_path, 'jpeg').read_bytes()
    client.read_bytes(data)
    old = client._process()
    assert old is not None
    os.kill(old.pid, signal.SIGKILL)
    old.wait(timeout=10)
    assert parse_capture(client.read_bytes(data)).camera_make == 'Acme'
    new = client._process()
    assert new is not None
    assert new.pid != old.pid


def test_a_process_that_never_opens_the_pipe_is_killed_at_the_timeout_and_no_writer_stays(
    tmp_path: Path,
) -> None:
    hung = tmp_path / 'exiftool'
    hung.write_text('#!/bin/sh\nexec sleep 300\n')
    hung.chmod(hung.stat().st_mode | stat.S_IXUSR)
    c = ExifToolClient(executable=str(hung), timeout_s=1.0)
    started = time.monotonic()
    with pytest.raises(ExifToolError):
        c.read_bytes(b'\xff\xd8' + b'\0' * 4_000_000)
    assert time.monotonic() - started < 15
    c.close()
    assert not [t for t in threading.enumerate() if t.name.startswith('exiftool-feed')]


# --- the pool -------------------------------------------------------------------------------


def test_pool_size_follows_the_cpu_count_within_bounds() -> None:
    from src.services.curation.capture_read import pool_size

    assert [pool_size(n) for n in (1, 8, 16, 24, 32, 48, 256)] == [2, 2, 2, 3, 4, 4, 4]


@pytest.mark.asyncio
async def test_concurrent_reads_use_several_processes_and_answer_each_photo(
    tmp_path: Path,
) -> None:
    from src.services.curation import capture_read

    jpeg = _make(tmp_path, 'jpeg').read_bytes()
    plain = io.BytesIO()
    Image.new('RGB', (16, 16)).save(plain, 'JPEG')
    try:
        results = await asyncio.gather(
            *[capture_read.read_capture(jpeg if i % 2 else plain.getvalue()) for i in range(24)]
        )
        assert [r.camera_make for r in results] == ['Acme' if i % 2 else None for i in range(24)]
        assert len(capture_read.live_processes()) > 1
    finally:
        capture_read.shutdown_capture_pool()
    assert capture_read.live_processes() == []


@pytest.mark.asyncio
async def test_an_outage_answers_unknown_for_the_photo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.services.curation import capture_read

    monkeypatch.setattr(
        capture_read,
        '_new_client',
        lambda: ExifToolClient(executable='/nonexistent/exiftool', timeout_s=5),
    )
    try:
        cap = await capture_read.read_capture(_make(tmp_path, 'jpeg').read_bytes())
    finally:
        capture_read.shutdown_capture_pool()
    assert cap.exif_status == 'unknown'


@pytest.mark.asyncio
async def test_a_read_that_outlasts_its_bound_is_unknown_and_does_not_hold_the_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.services.curation import capture_read

    release = threading.Event()

    class Stuck(ExifToolClient):
        def read_bytes(self, data: bytes) -> dict[str, Any]:
            release.wait(30)
            return {}

    monkeypatch.setattr(capture_read, '_new_client', lambda: Stuck(timeout_s=1))
    monkeypatch.setattr(capture_read, 'READ_BOUND_S', 0.5)
    try:
        started = time.monotonic()
        cap = await capture_read.read_capture(b'x')
        assert cap.exif_status == 'unknown'
        assert time.monotonic() - started < 5
    finally:
        release.set()
        capture_read.shutdown_capture_pool()
