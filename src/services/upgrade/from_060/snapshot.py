"""The bundle read back into memory, except the bulk files (ledgers, queue databases).

Two passes over the gzip file: :func:`read_snapshot` keeps indexes, documents and small files;
:func:`iter_bulk` then streams the bulk files one at a time into the ``COPY`` loaders.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from src.services.upgrade.from_060.bundle import iter_records


if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


@dataclass(frozen=True)
class BundleFile:
    area: str
    rel: str
    data: bytes
    mtime: str | None
    sha256: str

    def text(self) -> str:
        return self.data.decode('utf-8')

    def json(self) -> Any:
        return json.loads(self.data)

    def jsonl(self) -> Iterator[dict[str, Any]]:
        for line in self.data.decode('utf-8').splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue  # a torn last line from a crash, as the 0.6.0 reader skipped it
            if isinstance(row, dict):
                yield row


def decode_file(record: dict[str, Any]) -> BundleFile:
    raw = record['data']
    data = base64.b64decode(raw) if record['enc'] == 'b64' else raw.encode('utf-8')
    return BundleFile(record['area'], record['rel'], data, record.get('mtime'), record['sha256'])


@dataclass
class Snapshot:
    """Indexes, their documents and the small files, keyed for lookup."""

    index_defs: dict[str, tuple[dict[str, Any], dict[str, Any]]] = field(default_factory=dict)
    docs: dict[str, dict[str, dict[str, Any]]] = field(default_factory=dict)
    files: dict[str, dict[str, BundleFile]] = field(default_factory=dict)
    blobs: list[dict[str, Any]] = field(default_factory=list)

    def index_docs(self, index: str) -> dict[str, dict[str, Any]]:
        return self.docs.get(index, {})

    def area_files(self, area: str) -> dict[str, BundleFile]:
        return self.files.get(area, {})

    def file(self, area: str, rel: str) -> BundleFile | None:
        return self.files.get(area, {}).get(rel)


def read_snapshot(directory: Path) -> Snapshot:
    snap = Snapshot()
    for record in iter_records(directory):
        kind = record['t']
        if kind == 'index':
            snap.index_defs[record['name']] = (record['mappings'], record['settings'])
            snap.docs.setdefault(record['name'], {})
        elif kind == 'doc':
            snap.docs.setdefault(record['index'], {})[record['id']] = record['source']
        elif kind == 'file':
            if record.get('bulk'):
                continue
            file = decode_file(record)
            snap.files.setdefault(file.area, {})[file.rel] = file
        elif kind == 'blob':
            snap.blobs.append(record)
    return snap


def iter_bulk(directory: Path) -> Iterator[BundleFile]:
    """Every bulk file (ledger, queue database), one at a time, in bundle order."""
    for record in iter_records(directory):
        if record['t'] == 'file' and record.get('bulk'):
            yield decode_file(record)
