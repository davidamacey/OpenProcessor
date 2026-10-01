"""Archive upload is the widest attack surface of the import: every member
name, type and size is attacker-controlled (W10.3)."""

from __future__ import annotations

import io
import stat
import tarfile
import zipfile
from typing import TYPE_CHECKING

import pytest

from src.services.curation.dataset_import.upload import (
    ArchiveInvalidError,
    UploadTooLargeError,
    datasets_root,
    receive_archive,
    safe_extract,
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


def _zip(members: list[tuple[str, bytes, int | None]], *, deflate: bool = False) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED if deflate else zipfile.ZIP_STORED) as zf:
        for name, data, mode in members:
            info = zipfile.ZipInfo(name)
            if deflate:
                info.compress_type = zipfile.ZIP_DEFLATED
            if mode is not None:
                info.external_attr = mode << 16
            zf.writestr(info, data)
    return buf.getvalue()


def _tar(entries: list[tarfile.TarInfo], payloads: dict[str, bytes], *, gz: bool = False) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode='w:gz' if gz else 'w') as tf:
        for info in entries:
            data = payloads.get(info.name)
            info.size = len(data) if data is not None else 0
            tf.addfile(info, io.BytesIO(data) if data is not None else None)
    return buf.getvalue()


def _tarinfo(name: str, kind: bytes = tarfile.REGTYPE, linkname: str = '') -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.type = kind
    info.linkname = linkname
    return info


async def _stream(data: bytes, chunk: int = 1000) -> AsyncIterator[bytes]:
    for i in range(0, len(data), chunk):
        yield data[i : i + chunk]


def _extract(tmp_path: Path, data: bytes) -> Path:
    archive = tmp_path / 'in.bin'
    archive.write_bytes(data)
    dest = tmp_path / 'out'
    safe_extract(archive, dest)
    return dest


def _tree(root: Path) -> list[str]:
    return sorted(p.relative_to(root).as_posix() for p in root.rglob('*'))


HOSTILE_ZIP_NAMES = [
    '../escape.txt',
    'a/../../escape.txt',
    '/abs/escape.txt',
    '..\\escape.txt',
    'C:/Windows/x',
    '~/x',
    '',
]


@pytest.mark.parametrize('name', HOSTILE_ZIP_NAMES)
def test_zip_member_names_that_leave_the_dir_are_refused_and_nothing_stays(
    tmp_path: Path, name: str
) -> None:
    data = _zip([('ok/data.yaml', b'names: [a]', None), (name, b'x', None)])
    if name == 'C:/Windows/x':
        # A drive-looking name is just a relative directory on Linux.
        assert 'C:' in _tree(_extract(tmp_path, data))
        return
    with pytest.raises(ArchiveInvalidError):
        _extract(tmp_path, data)
    assert _tree(tmp_path / 'out') == []
    assert not (tmp_path / 'escape.txt').exists()


@pytest.mark.parametrize('name', ['../escape.txt', '/abs/escape.txt', 'a/../../escape.txt'])
def test_tar_member_names_that_leave_the_dir_are_refused(tmp_path: Path, name: str) -> None:
    data = _tar([_tarinfo(name)], {name: b'x'})
    with pytest.raises(ArchiveInvalidError):
        _extract(tmp_path, data)
    assert _tree(tmp_path / 'out') == []


def test_zip_symlink_member_is_refused(tmp_path: Path) -> None:
    data = _zip([('link', b'/etc/passwd', stat.S_IFLNK | 0o777)])
    with pytest.raises(ArchiveInvalidError, match='non-regular'):
        _extract(tmp_path, data)


@pytest.mark.parametrize(
    'kind', [tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.CHRTYPE, tarfile.BLKTYPE, tarfile.FIFOTYPE]
)
def test_tar_non_regular_members_are_refused(tmp_path: Path, kind: bytes) -> None:
    data = _tar([_tarinfo('ds/x', kind, linkname='/etc/passwd')], {})
    with pytest.raises(ArchiveInvalidError, match='non-regular'):
        _extract(tmp_path, data)
    assert _tree(tmp_path / 'out') == []


def test_zip_member_over_the_ratio_is_a_bomb(tmp_path: Path) -> None:
    data = _zip([('ds/zeros.bin', b'\0' * (5 * 1024 * 1024), None)], deflate=True)
    with pytest.raises(ArchiveInvalidError, match='compression ratio'):
        _extract(tmp_path, data)


def test_total_uncompressed_size_is_capped_by_bytes_written_not_headers(
    tmp_path: Path, monkeypatch
) -> None:
    """A tar header can say 1 byte and carry more; the cap counts what is
    written (4x ``OP_DATASET_UPLOAD_MAX_BYTES``)."""
    monkeypatch.setenv('OP_DATASET_UPLOAD_MAX_BYTES', '1000')
    files = [_tarinfo(f'ds/f{i}.bin') for i in range(5)]
    data = _tar(files, {f'ds/f{i}.bin': b'x' * 900 for i in range(5)})
    with pytest.raises(ArchiveInvalidError, match='size over the limit'):
        _extract(tmp_path, data)
    assert _tree(tmp_path / 'out') == []


def test_zip_declared_total_over_the_cap_is_refused_before_extracting(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv('OP_DATASET_UPLOAD_MAX_BYTES', '100')
    data = _zip([(f'ds/f{i}', b'x' * 200, None) for i in range(3)])
    with pytest.raises(ArchiveInvalidError, match='size over the limit'):
        _extract(tmp_path, data)


@pytest.mark.parametrize('maker', ['zip', 'tar'])
def test_member_count_cap(tmp_path: Path, monkeypatch, maker: str) -> None:
    monkeypatch.setenv('OP_DATASET_UPLOAD_MAX_FILES', '3')
    names = [f'ds/f{i}.txt' for i in range(5)]
    data = (
        _zip([(n, b'x', None) for n in names])
        if maker == 'zip'
        else _tar([_tarinfo(n) for n in names], dict.fromkeys(names, b'x'))
    )
    with pytest.raises(ArchiveInvalidError, match='more than 3 members'):
        _extract(tmp_path, data)
    assert _tree(tmp_path / 'out') == []


def test_encrypted_zip_member_is_refused(tmp_path: Path) -> None:
    raw = bytearray(_zip([('ds/a.txt', b'x', None)]))
    # Set the "encrypted" general-purpose flag bit in the local and central headers.
    for marker in (b'PK\x03\x04', b'PK\x01\x02'):
        i = raw.find(marker)
        offset = i + (6 if marker == b'PK\x03\x04' else 8)
        raw[offset] |= 0x1
    with pytest.raises(ArchiveInvalidError, match='encrypted'):
        _extract(tmp_path, bytes(raw))


@pytest.mark.parametrize('data', [b'', b'not an archive at all', b'PK\x03\x04' + b'\0' * 30])
def test_garbage_is_an_archive_invalid_not_an_exception(tmp_path: Path, data: bytes) -> None:
    with pytest.raises(ArchiveInvalidError):
        _extract(tmp_path, data)


def test_extracted_files_get_plain_permissions(tmp_path: Path) -> None:
    data = _zip([('ds/run.sh', b'#!/bin/sh', stat.S_IFREG | 0o4755)])
    dest = _extract(tmp_path, data)
    mode = (dest / 'ds/run.sh').stat().st_mode
    assert not mode & (stat.S_ISUID | stat.S_IXUSR)


# ------------------------------------------------------------ receive_archive


@pytest.mark.asyncio
async def test_happy_path_is_content_addressed_and_idempotent(tmp_path: Path) -> None:
    data = _zip([('ds/data.yaml', b'names: [a]', None), ('ds/images/a.jpg', b'x', None)])
    first = await receive_archive(_stream(data), upload_root=tmp_path)
    assert first.files == 2
    assert first.dataset_path == datasets_root(tmp_path) / first.upload_id / 'ds'
    assert (first.dataset_path / 'data.yaml').read_bytes() == b'names: [a]'
    again = await receive_archive(_stream(data), upload_root=tmp_path)
    assert again.upload_id == first.upload_id
    assert again.dataset_path == first.dataset_path
    leftovers = list((datasets_root(tmp_path) / '.incoming').iterdir())
    assert leftovers == []


@pytest.mark.asyncio
async def test_tar_gz_is_accepted(tmp_path: Path) -> None:
    data = _tar([_tarinfo('ds/data.yaml')], {'ds/data.yaml': b'names: [a]'}, gz=True)
    result = await receive_archive(_stream(data), upload_root=tmp_path)
    assert (result.dataset_path / 'data.yaml').is_file()


@pytest.mark.asyncio
async def test_upload_over_the_cap_is_refused_mid_stream_and_leaves_nothing(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv('OP_DATASET_UPLOAD_MAX_BYTES', '2500')
    read = 0

    async def endless() -> AsyncIterator[bytes]:
        nonlocal read
        while True:
            read += 1000
            yield b'x' * 1000

    with pytest.raises(UploadTooLargeError) as exc:
        await receive_archive(endless(), upload_root=tmp_path)
    assert exc.value.limit == 2500
    assert read <= 3000  # stopped as soon as it passed the cap
    root = datasets_root(tmp_path)
    assert [p.name for p in root.rglob('*') if p.is_file()] == ['.project']


@pytest.mark.asyncio
async def test_a_refused_archive_leaves_no_extracted_dir(tmp_path: Path) -> None:
    data = _zip([('../escape.txt', b'x', None)])
    with pytest.raises(ArchiveInvalidError):
        await receive_archive(_stream(data), upload_root=tmp_path)
    root = datasets_root(tmp_path)
    assert [p.name for p in root.iterdir() if not p.name.startswith('.')] == []
    assert [p.name for p in root.rglob('*') if p.is_file()] == ['.project']


def test_sweep_removes_only_expired_unreferenced_uploads(tmp_path: Path, monkeypatch) -> None:
    import os
    import time

    from src.services.curation.dataset_import.upload import sweep_uploads

    root = datasets_root(tmp_path)
    for name in ('old_unref', 'old_ref', 'fresh'):
        (root / name).mkdir(parents=True)
        (root / name / 'f').write_text('x')
    old = time.time() - 100 * 3600
    for name in ('old_unref', 'old_ref'):
        os.utime(root / name, (old, old))
    removed = sweep_uploads(tmp_path, referenced={'old_ref'})
    assert removed == ['old_unref']
    assert sorted(p.name for p in root.iterdir() if p.name != '.project') == ['fresh', 'old_ref']
    assert (root / '.project').read_text() == 'default'  # the sweep stamps a dir it cleans
