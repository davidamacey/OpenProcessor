"""Archive structure caps and the upload directory's housekeeping: directory
headers count, a crashed upload's leftovers are swept, a re-upload restarts
the TTL, two uploads of one archive do not remove each other's files."""

from __future__ import annotations

import hashlib
import io
import os
import tarfile
import time
import zipfile
from typing import TYPE_CHECKING, Any

import pytest

from src.services.curation.dataset_import import upload
from src.services.curation.dataset_import.upload import (
    ArchiveInvalidError,
    datasets_root,
    receive_archive,
    safe_extract,
    sweep_uploads,
)


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator
    from pathlib import Path


@pytest.fixture(autouse=True)
def _bound_project() -> Iterator[None]:
    from src.config.curation import base_curation_config
    from src.config.project_context import bind_project
    from src.config.projects import new_project_record

    with bind_project(new_project_record('default', base_curation_config())):
        yield


def _tar_dirs(names: list[str]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode='w:gz') as tf:
        for name in names:
            info = tarfile.TarInfo(name)
            info.type = tarfile.DIRTYPE
            tf.addfile(info)
    return buf.getvalue()


def _zip(members: list[tuple[str, bytes]]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w') as zf:
        for name, data in members:
            zf.writestr(name, data)
    return buf.getvalue()


async def _stream(data: bytes) -> AsyncIterator[bytes]:
    yield data


def _extract(tmp_path: Path, data: bytes) -> Path:
    archive = tmp_path / 'in.bin'
    archive.write_bytes(data)
    dest = tmp_path / 'out'
    safe_extract(archive, dest)
    return dest


def test_tar_directory_members_count_against_the_member_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv('OP_DATASET_UPLOAD_MAX_FILES', '10')
    with pytest.raises(ArchiveInvalidError, match='more than 10 members'):
        _extract(tmp_path, _tar_dirs([f'd{i}/' for i in range(2000)]))
    assert list((tmp_path / 'out').iterdir()) == []


def test_zip_directory_depth_is_capped(tmp_path: Path) -> None:
    deep = '/'.join(['d'] * (upload._MAX_DEPTH + 1)) + '/'
    with pytest.raises(ArchiveInvalidError, match='nested deeper'):
        _extract(tmp_path, _zip([(deep, b'')]))


def test_tar_directory_depth_is_capped(tmp_path: Path) -> None:
    deep = '/'.join(['d'] * (upload._MAX_DEPTH + 1)) + '/'
    with pytest.raises(ArchiveInvalidError, match='nested deeper'):
        _extract(tmp_path, _tar_dirs([deep]))


@pytest.mark.asyncio
async def test_a_macos_resource_fork_folder_does_not_hide_the_single_root(
    tmp_path: Path,
) -> None:
    data = _zip([('ds/data.yaml', b'names: [a]'), ('__MACOSX/ds/._data.yaml', b'x')])
    result = await receive_archive(_stream(data), upload_root=tmp_path)
    assert result.dataset_path.name == 'ds'


def test_the_sweep_removes_a_crashed_uploads_leftovers_and_keeps_a_live_one(
    tmp_path: Path,
) -> None:
    incoming = datasets_root(tmp_path) / '.incoming'
    incoming.mkdir(parents=True)
    stale_part = incoming / 'a.part'
    stale_dir = incoming / 'b.dir'
    live_part = incoming / 'c.part'
    stale_part.write_bytes(b'x')
    (stale_dir / 'f').parent.mkdir()
    (stale_dir / 'f').write_bytes(b'x')
    live_part.write_bytes(b'x')
    old = time.time() - 2 * 3600
    os.utime(stale_part, (old, old))
    os.utime(stale_dir, (old, old))
    sweep_uploads(tmp_path, referenced=set())
    assert not stale_part.exists()
    assert not stale_dir.exists()
    assert live_part.exists()


@pytest.mark.asyncio
async def test_a_re_upload_restarts_the_ttl(tmp_path: Path) -> None:
    data = _zip([('ds/data.yaml', b'names: [a]')])
    first = await receive_archive(_stream(data), upload_root=tmp_path)
    folder = datasets_root(tmp_path) / first.upload_id
    old = time.time() - 71 * 3600
    os.utime(folder, (old, old))
    await receive_archive(_stream(data), upload_root=tmp_path)
    assert sweep_uploads(tmp_path, referenced=set(), now=time.time() + 2 * 3600) == []
    assert folder.exists()


@pytest.mark.asyncio
async def test_a_concurrent_upload_of_the_same_archive_is_not_removed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = _zip([('ds/data.yaml', b'names: [a]')])
    final = datasets_root(tmp_path) / hashlib.sha256(data).hexdigest()[:16]
    real = upload.safe_extract

    def other_upload_wins(archive: Path, dest: Path) -> Any:
        out = real(archive, dest)
        (final / 'ds').mkdir(parents=True)
        (final / 'ds' / 'data.yaml').write_bytes(b'names: [a]')
        (final / 'ds' / 'winner.txt').write_bytes(b'kept')
        return out

    monkeypatch.setattr(upload, 'safe_extract', other_upload_wins)
    result = await receive_archive(_stream(data), upload_root=tmp_path)
    assert (final / 'ds' / 'winner.txt').read_bytes() == b'kept'
    assert result.dataset_path == final / 'ds'
