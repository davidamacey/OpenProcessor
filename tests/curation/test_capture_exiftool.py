"""The ExifTool client against the real ``exiftool`` binary (research spatial plan 5.1, CM-1).

Every file is generated: Pillow writes a small JPEG and ``exiftool`` itself writes the tags into
it, so no photo is stored in the repository. The binary comes from Debian's
``libimage-exiftool-perl`` (the api image installs it; CI installs it before this module).
"""

from __future__ import annotations

import io
import os
import signal
import stat
import subprocess
import time
from typing import TYPE_CHECKING, Any

import pytest
from PIL import Image

from src.clients.exiftool_client import ExifToolClient, ExifToolError
from src.domain.capture_metadata import parse_capture


if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


def _jpeg(directory: Path, name: str, *tags: str) -> Path:
    path = directory / name
    Image.new('RGB', (64, 48), (200, 10, 10)).save(path)
    if tags:
        subprocess.run(
            ['exiftool', '-overwrite_original', '-q', *tags, str(path)], check=True, timeout=60
        )
    return path


@pytest.fixture
def client() -> Iterator[ExifToolClient]:
    c = ExifToolClient(timeout_s=20)
    yield c
    c.close()


def test_the_real_binary_returns_numeric_signed_gps_and_the_parser_reads_it(
    client: ExifToolClient, tmp_path: Path
) -> None:
    path = _jpeg(
        tmp_path,
        'gps.jpg',
        '-DateTimeOriginal=2024:03:05 14:22:11',
        '-OffsetTimeOriginal=+01:00',
        '-SubSecTimeOriginal=50',
        '-GPSLatitude=47.5',
        '-GPSLatitudeRef=S',
        '-GPSLongitude=8.25',
        '-GPSLongitudeRef=W',
        '-GPSAltitude=410',
        '-Make=Acme',
        '-Model=Cam1',
        '-ModifyDate=2025:01:01 00:00:00',
        '-Orientation#=6',
    )
    cap = parse_capture(client.read(path))
    assert cap.exif_status == 'ok'
    assert (cap.taken_local, cap.taken_utc) == ('2024-03-05T14:22:11.50', '2024-03-05T13:22:11.50')
    assert cap.location is not None
    assert (cap.location.latitude, cap.location.longitude) == (-47.5, -8.25)
    assert (cap.camera_make, cap.camera_model, cap.orientation) == ('Acme', 'Cam1', 6)


def test_a_file_without_exif_is_absent(client: ExifToolClient, tmp_path: Path) -> None:
    assert parse_capture(client.read(_jpeg(tmp_path, 'plain.jpg'))).exif_status == 'absent'


def test_a_corrupt_file_is_malformed_not_an_outage(client: ExifToolClient, tmp_path: Path) -> None:
    bad = tmp_path / 'bad.jpg'
    bad.write_bytes(b'\xff\xd8garbage')
    assert parse_capture(client.read(bad)).exif_status == 'malformed'
    # the process survives a bad file: the next read works
    good = _jpeg(tmp_path, 'ok.jpg', '-Make=Acme')
    assert parse_capture(client.read(good)).camera_make == 'Acme'


def test_one_process_serves_every_read(client: ExifToolClient, tmp_path: Path) -> None:
    path = _jpeg(tmp_path, 'a.jpg', '-Make=Acme')
    client.read(path)
    first = client._process()
    assert first is not None
    for _ in range(5):
        client.read(path)
    assert client._process() is first


def test_a_killed_process_is_replaced_on_the_next_read(
    client: ExifToolClient, tmp_path: Path
) -> None:
    path = _jpeg(tmp_path, 'a.jpg', '-Make=Acme')
    client.read(path)
    old = client._process()
    assert old is not None
    os.kill(old.pid, signal.SIGKILL)
    old.wait(timeout=10)
    assert parse_capture(client.read(path)).camera_make == 'Acme'
    new = client._process()
    assert new is not None
    assert new.pid != old.pid


def test_read_bytes_reads_an_in_memory_file(client: ExifToolClient, tmp_path: Path) -> None:
    path = _jpeg(tmp_path, 'a.jpg', '-Make=Acme')
    cap = parse_capture(client.read_bytes(path.read_bytes(), '.jpg'))
    assert cap.camera_make == 'Acme'
    buf = io.BytesIO()
    Image.new('RGB', (8, 8)).save(buf, 'PNG')
    assert parse_capture(client.read_bytes(buf.getvalue(), '.png')).exif_status == 'absent'


def test_a_missing_binary_is_a_typed_outage() -> None:
    c = ExifToolClient(executable='/nonexistent/exiftool', timeout_s=5)
    with pytest.raises(ExifToolError):
        c.read('/etc/hostname')


def test_a_hung_process_is_killed_at_the_timeout_and_reported(tmp_path: Path) -> None:
    hung = tmp_path / 'exiftool'
    hung.write_text('#!/bin/sh\nexec sleep 300\n')
    hung.chmod(hung.stat().st_mode | stat.S_IXUSR)
    c = ExifToolClient(executable=str(hung), timeout_s=1.0)
    started = time.monotonic()
    with pytest.raises(ExifToolError):
        c.read('/etc/hostname')
    assert time.monotonic() - started < 15
    c.close()


def _child_reads(path: str, out: Any) -> None:
    from src.clients.exiftool_client import get_exiftool_client

    client = get_exiftool_client()
    make = parse_capture(client.read(path)).camera_make
    process = client._process()
    out.put((make, None if process is None else process.pid))


def test_a_forked_child_starts_its_own_process_and_leaves_the_parents_alone(
    tmp_path: Path,
) -> None:
    import multiprocessing

    from src.clients.exiftool_client import get_exiftool_client

    path = _jpeg(tmp_path, 'a.jpg', '-Make=Acme')
    parent = get_exiftool_client()
    parent.read(path)
    process = parent._process()
    assert process is not None

    ctx = multiprocessing.get_context('fork')
    out = ctx.Queue()
    child = ctx.Process(target=_child_reads, args=(str(path), out))
    child.start()
    make, child_pid = out.get(timeout=60)
    assert make == 'Acme'
    assert child_pid not in (None, process.pid)  # its own exiftool, not the parent's pipes
    child.join(timeout=60)

    assert child.exitcode == 0
    assert process.poll() is None  # the child's exit did not tell the parent's exiftool to quit
    assert parse_capture(parent.read(path)).camera_make == 'Acme'
    assert parent._process() is process
    parent.close()
