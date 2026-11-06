"""The control bundle: one gzip-compressed JSON-lines file and its manifest.

``control_bundle.jsonl.gz`` holds one record per source document or file, the index mappings
and settings of every control index, and the bytes of every kept job file, so the rollback can
rebuild the 0.6.0 state from it alone. ``MANIFEST.json`` lists the counts and the bundle's
sha256. gzip comes from the standard library (review fix R-31: Python 3.13 has no standard-library
zstd and no zstd package is pinned).

Record shapes (``t`` is the type):

* ``index``: ``name``, ``mappings``, ``settings``
* ``doc``: ``index``, ``id``, ``source``
* ``file``: ``area``, ``rel``, ``enc`` (``utf8`` or ``b64``), ``data``, ``bulk`` (true for ledgers
  and queue databases, which the load streams instead of holding), ``sha256``, ``mtime`` (UTC ISO)
* ``blob``: ``key``, ``source_area``, ``source_rel``: a file copied to a storage key (run logs, UMAP
  reducers); the bytes are in the store, the record says where they came from
"""

from __future__ import annotations

import gzip
import hashlib
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from src.utils.atomic_write import atomic_write_json


if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

BUNDLE_NAME = 'control_bundle.jsonl.gz'
MANIFEST_NAME = 'MANIFEST.json'
FORMAT_VERSION = 1


class BundleError(RuntimeError):
    """The bundle is missing, torn or does not match its manifest."""


@dataclass
class Manifest:
    """Counts per record family and the bundle's sha256."""

    created_at: str
    from_version: str
    counts: dict[str, int] = field(default_factory=dict)
    sha256: str = ''
    blobs: list[str] = field(default_factory=list)
    format_version: int = FORMAT_VERSION

    def count(self, family: str) -> None:
        self.counts[family] = self.counts.get(family, 0) + 1

    def to_json(self) -> dict[str, Any]:
        return {
            'format_version': self.format_version,
            'created_at': self.created_at,
            'from_version': self.from_version,
            'counts': dict(sorted(self.counts.items())),
            'blobs': sorted(self.blobs),
            'sha256': self.sha256,
        }

    @classmethod
    def from_json(cls, raw: dict[str, Any]) -> Manifest:
        return cls(
            created_at=raw['created_at'],
            from_version=raw['from_version'],
            counts={str(k): int(v) for k, v in raw['counts'].items()},
            sha256=raw['sha256'],
            blobs=list(raw.get('blobs', [])),
            format_version=int(raw.get('format_version', FORMAT_VERSION)),
        )


def family_of(record: dict[str, Any]) -> str:
    """The manifest family a record counts under: ``index:<name>``, ``doc:<index>``, ``file:<area>``."""
    kind = record['t']
    if kind == 'index':
        return f'index:{record["name"]}'
    if kind == 'doc':
        return f'doc:{record["index"]}'
    if kind == 'file':
        return f'file:{record["area"]}'
    return f'blob:{record["source_area"]}'


class BundleWriter:
    """Writes records and, on :meth:`close`, the manifest. A context manager."""

    def __init__(self, directory: Path, *, created_at: str, from_version: str) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.directory = directory
        self.path = directory / BUNDLE_NAME
        self.manifest = Manifest(created_at=created_at, from_version=from_version)
        self._tmp = directory / (BUNDLE_NAME + '.partial')
        self._fh = gzip.open(self._tmp, 'wt', encoding='utf-8', compresslevel=6)  # noqa: SIM115  # closed by close() or __exit__

    def write(self, record: dict[str, Any]) -> None:
        self._fh.write(json.dumps(record, separators=(',', ':'), sort_keys=True))
        self._fh.write('\n')
        self.manifest.count(family_of(record))
        if record['t'] == 'blob':
            self.manifest.blobs.append(record['key'])

    def close(self) -> Manifest:
        self._fh.close()
        self.manifest.sha256 = sha256_file(self._tmp)
        self._tmp.replace(self.path)
        atomic_write_json(self.directory / MANIFEST_NAME, self.manifest.to_json(), default=None)
        return self.manifest

    def __enter__(self) -> BundleWriter:
        return self

    def __exit__(self, exc_type: object, *_: object) -> None:
        if exc_type is not None:
            self._fh.close()
            self._tmp.unlink(missing_ok=True)
            return
        self.close()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest(directory: Path) -> Manifest:
    try:
        return Manifest.from_json(json.loads((directory / MANIFEST_NAME).read_text('utf-8')))
    except (OSError, ValueError, KeyError) as exc:
        raise BundleError(f'{directory}: no readable {MANIFEST_NAME}: {exc}') from exc


def iter_records(directory: Path) -> Iterator[dict[str, Any]]:
    """Every record, in file order. The caller verifies the bundle first (:func:`check_bundle`)."""
    with gzip.open(directory / BUNDLE_NAME, 'rt', encoding='utf-8') as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def check_bundle(directory: Path) -> Manifest:
    """Refuse a bundle whose bytes or counts differ from its manifest."""
    manifest = read_manifest(directory)
    path = directory / BUNDLE_NAME
    if not path.is_file():
        raise BundleError(f'{path}: the bundle is missing')
    actual = sha256_file(path)
    if actual != manifest.sha256:
        raise BundleError(f'{path}: sha256 {actual} differs from the manifest {manifest.sha256}')
    counted: dict[str, int] = {}
    for record in iter_records(directory):
        family = family_of(record)
        counted[family] = counted.get(family, 0) + 1
    if counted != manifest.counts:
        raise BundleError(f'{path}: record counts differ from the manifest')
    return manifest
