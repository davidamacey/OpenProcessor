"""Archive upload for a dataset (W10.3): one ``.zip`` / ``.tar`` /
``.tar.gz``, streamed to disk, extracted by :func:`safe_extract` under
``<upload_root>/datasets/<sha256[:16]>/`` (content-addressed, inside a
configured root, so every extracted image passes the servability guard).

An archive is attacker-controlled: nothing is trusted from its headers.
Member names resolve through :func:`~.paths.resolve_ref`; only regular
files and directories are extracted (a symlink, hardlink, device or fifo
fails the whole archive); the member count and the BYTES ACTUALLY WRITTEN
are capped (a header that lies about a size cannot get past the cap); and
any failure removes everything extracted so far.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import stat
import tarfile
import time
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import IO, TYPE_CHECKING

from src.services.curation.dataset_import import limits
from src.services.curation.dataset_import.paths import resolve_ref


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

_CHUNK = 1024 * 1024
_ZIP_MAGIC = b'PK\x03\x04'
_GZIP_MAGIC = b'\x1f\x8b'
_EXPANSION_FACTOR = 4
_MAX_RATIO = 250


class ArchiveInvalidError(Exception):
    pass


class UploadTooLargeError(Exception):
    def __init__(self, limit: int) -> None:
        super().__init__(f'upload over {limit} bytes')
        self.limit = limit


@dataclass(frozen=True)
class UploadResult:
    upload_id: str
    dataset_path: Path
    bytes: int
    files: int


def datasets_root(upload_root: Path) -> Path:
    return upload_root / 'datasets'


def _kind(path: Path) -> str:
    with path.open('rb') as fh:
        head = fh.read(512)
    if head.startswith(_ZIP_MAGIC):
        return 'zip'
    if head.startswith(_GZIP_MAGIC) or head[257:262] == b'ustar':
        return 'tar'
    raise ArchiveInvalidError('not a zip, tar or tar.gz archive')


class _Budget:
    """Running count of files and bytes written to disk."""

    def __init__(self) -> None:
        self.files = 0
        self.bytes = 0
        self.max_files = limits.upload_max_files()
        self.max_bytes = _EXPANSION_FACTOR * limits.upload_max_bytes()

    def add_file(self) -> None:
        self.files += 1
        if self.files > self.max_files:
            raise ArchiveInvalidError(f'more than {self.max_files} members')

    def add_bytes(self, n: int) -> None:
        self.bytes += n
        if self.bytes > self.max_bytes:
            raise ArchiveInvalidError('uncompressed size over the limit')


def _target(dest: Path, name: str) -> Path:
    target = resolve_ref(dest, name.rstrip('/') or '.')
    if target is None or target == dest.resolve():
        raise ArchiveInvalidError(f'unsafe member name: {name[:80]!r}')
    return target


def _copy_limited(src: IO[bytes], out: Path, budget: _Budget) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('wb') as fh:
        while chunk := src.read(_CHUNK):
            budget.add_bytes(len(chunk))
            fh.write(chunk)


def _extract_zip(archive: Path, dest: Path, budget: _Budget) -> None:
    try:
        zf = zipfile.ZipFile(archive)
    except zipfile.BadZipFile as exc:
        raise ArchiveInvalidError(f'corrupt zip: {exc}') from exc
    with zf:
        infos = zf.infolist()
        if len(infos) > budget.max_files:
            raise ArchiveInvalidError(f'more than {budget.max_files} members')
        if sum(i.file_size for i in infos) > budget.max_bytes:
            raise ArchiveInvalidError('uncompressed size over the limit')
        for info in infos:
            mode = info.external_attr >> 16
            if info.flag_bits & 0x1:
                raise ArchiveInvalidError('encrypted member')
            # Many writers store permission bits only (no file-type bits): a
            # member is refused only when it declares a type that is not a
            # regular file or directory (symlink, device, fifo, socket).
            if stat.S_IFMT(mode) not in (0, stat.S_IFREG, stat.S_IFDIR):
                raise ArchiveInvalidError(f'non-regular member: {info.filename[:80]!r}')
            target = _target(dest, info.filename)
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if info.compress_size and info.file_size / info.compress_size > _MAX_RATIO:
                raise ArchiveInvalidError(f'suspicious compression ratio: {info.filename[:80]!r}')
            budget.add_file()
            with zf.open(info) as src:
                _copy_limited(src, target, budget)


def _extract_tar(archive: Path, dest: Path, budget: _Budget) -> None:
    try:
        tf = tarfile.open(archive, mode='r:*')  # noqa: SIM115 - closed below
    except (tarfile.TarError, OSError) as exc:
        raise ArchiveInvalidError(f'corrupt tar: {exc}') from exc
    with tf:
        for member in tf:
            if not (member.isreg() or member.isdir()):
                raise ArchiveInvalidError(f'non-regular member: {member.name[:80]!r}')
            target = _target(dest, member.name)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            budget.add_file()
            src = tf.extractfile(member)
            if src is None:
                raise ArchiveInvalidError(f'unreadable member: {member.name[:80]!r}')
            with src:
                _copy_limited(src, target, budget)


def safe_extract(archive: Path, dest: Path) -> tuple[int, int]:
    """Extract ``archive`` into the EMPTY directory ``dest``; returns
    ``(files, bytes)``. Raises :class:`ArchiveInvalidError` and leaves
    ``dest`` empty on any failure."""
    dest.mkdir(parents=True, exist_ok=True)
    budget = _Budget()
    try:
        if _kind(archive) == 'zip':
            _extract_zip(archive, dest, budget)
        else:
            _extract_tar(archive, dest, budget)
    except (
        ArchiveInvalidError,
        OSError,
        tarfile.TarError,
        zipfile.BadZipFile,
        RuntimeError,
    ) as exc:
        shutil.rmtree(dest, ignore_errors=True)
        dest.mkdir(parents=True, exist_ok=True)
        if isinstance(exc, ArchiveInvalidError):
            raise
        raise ArchiveInvalidError(f'extraction failed: {exc}') from exc
    return budget.files, budget.bytes


def _single_root(path: Path) -> Path:
    """An archive wrapping everything in one folder imports from that folder."""
    children = [p for p in path.iterdir() if not p.name.startswith('.')]
    if len(children) == 1 and children[0].is_dir():
        return children[0]
    return path


async def receive_archive(stream: AsyncIterator[bytes], *, upload_root: Path) -> UploadResult:
    """Stream ``stream`` to disk, then extract it content-addressed.

    Raises :class:`UploadTooLargeError` the moment the stream passes
    ``OP_DATASET_UPLOAD_MAX_BYTES`` and :class:`ArchiveInvalidError` for an
    unsafe archive (nothing is left extracted).
    """
    root = datasets_root(upload_root)
    incoming = root / '.incoming'
    incoming.mkdir(parents=True, exist_ok=True)
    cap = limits.upload_max_bytes()
    tmp = incoming / f'{uuid.uuid4().hex}.part'
    digest = hashlib.sha256()
    size = 0
    try:
        with tmp.open('wb') as fh:
            async for chunk in stream:
                size += len(chunk)
                if size > cap:
                    raise UploadTooLargeError(cap)
                digest.update(chunk)
                fh.write(chunk)
        upload_id = digest.hexdigest()[:16]
        final = root / upload_id
        if final.is_dir() and any(final.iterdir()):
            files = sum(1 for p in final.rglob('*') if p.is_file())
            return UploadResult(upload_id, _single_root(final), size, files)
        staging = incoming / f'{uuid.uuid4().hex}.dir'
        try:
            files, _written = safe_extract(tmp, staging)
            final.parent.mkdir(parents=True, exist_ok=True)
            if final.exists():
                shutil.rmtree(final, ignore_errors=True)
            staging.replace(final)
        finally:
            shutil.rmtree(staging, ignore_errors=True)
        return UploadResult(upload_id, _single_root(final), size, files)
    finally:
        tmp.unlink(missing_ok=True)


def sweep_uploads(
    upload_root: Path, *, referenced: set[str], now: float | None = None
) -> list[str]:
    """Remove uploads older than ``OP_DATASET_UPLOAD_TTL_H`` that no import
    references. Returns the removed upload ids."""
    root = datasets_root(upload_root)
    if not root.is_dir():
        return []
    cutoff = (now if now is not None else time.time()) - limits.upload_ttl_hours() * 3600
    removed = []
    for path in root.iterdir():
        if path.name.startswith('.') or not path.is_dir() or path.name in referenced:
            continue
        if path.stat().st_mtime < cutoff:
            shutil.rmtree(path, ignore_errors=True)
            removed.append(path.name)
    return removed


def upload_id_of(path: str | Path, upload_root: Path) -> str | None:
    """The upload id a dataset path lives under, if any."""
    try:
        rel = Path(os.path.realpath(path)).relative_to(os.path.realpath(datasets_root(upload_root)))
    except ValueError:
        return None
    return rel.parts[0] if rel.parts else None


__all__ = [
    'ArchiveInvalidError',
    'UploadResult',
    'UploadTooLargeError',
    'datasets_root',
    'receive_archive',
    'safe_extract',
    'sweep_uploads',
    'upload_id_of',
]
