"""Step 1, ``export_bundle``: read every 0.6.0 source and write the control bundle.

Reads only. OpenSearch is read through :class:`ControlReader` (``GET``, ``search`` with
``search_after``, ``_mapping``/``_settings``, ``HEAD``); files are opened for reading and never
touched. The bundle has everything the load needs and everything the rollback needs to rebuild
the 0.6.0 state, so nothing the upgrade deletes exists only in OpenSearch or on a volume.
"""

from __future__ import annotations

import base64
import datetime as dt
import hashlib
import sqlite3
from typing import TYPE_CHECKING, Any

from src.clients.storage.keys import make_key
from src.services.upgrade.from_060 import sources as legacy
from src.services.upgrade.from_060.bundle import BundleWriter


if TYPE_CHECKING:
    from pathlib import Path

    from src.clients.storage.base import BlobStore
    from src.services.upgrade.from_060.admin_client import ControlReader
    from src.services.upgrade.from_060.bundle import Manifest
    from src.services.upgrade.from_060.layout import Area, Layout

RUN_LOG_SUFFIX = '.run.log'
RUN_LOG_KEY_NAME = 'run.log'


def _file_bytes(path: Path) -> bytes:
    """The file's bytes; a queue database is read as one consistent image (``serialize`` includes
    the WAL), through a read-only connection."""
    if path.name.endswith('queue.sqlite'):
        uri = f'file:{path.as_posix()}?mode=ro'
        conn = sqlite3.connect(uri, uri=True)
        try:
            return bytes(conn.serialize())
        finally:
            conn.close()
    return path.read_bytes()


def _encode(data: bytes) -> tuple[str, str]:
    try:
        return 'utf8', data.decode('utf-8')
    except UnicodeDecodeError:
        return 'b64', base64.b64encode(data).decode('ascii')


def run_log_key(slug: str, rel: str) -> str:
    """The storage key a training run log is copied to (``projects/<slug>/training/<run>/run.log``)."""
    run_id = rel.removesuffix(RUN_LOG_SUFFIX)
    return make_key('projects', slug, 'training', run_id, RUN_LOG_KEY_NAME)


def _write_area(writer: BundleWriter, store: BlobStore, area: Area) -> None:
    for rel, path in area.files():
        data = _file_bytes(path)
        sha = hashlib.sha256(data).hexdigest()
        if area.family == 'training' and rel.endswith(RUN_LOG_SUFFIX) and area.slug:
            key = run_log_key(area.slug, rel)
            store.write_atomic(key, data)
            writer.write(
                {
                    't': 'blob',
                    'key': key,
                    'source_area': area.label,
                    'source_rel': rel,
                    'sha256': sha,
                }
            )
            continue
        enc, text = _encode(data)
        writer.write(
            {
                't': 'file',
                'area': area.label,
                'rel': rel,
                'enc': enc,
                'data': text,
                'bulk': area.is_bulk(rel),
                'sha256': sha,
                'mtime': dt.datetime.fromtimestamp(path.stat().st_mtime, dt.UTC).isoformat(),
            }
        )


async def _write_index(writer: BundleWriter, reader: ControlReader, name: str) -> bool:
    if not await reader.index_exists(name):
        return False
    mappings, settings = await reader.definition(name)
    writer.write({'t': 'index', 'name': name, 'mappings': mappings, 'settings': settings})
    async for doc_id, source in reader.iter_docs(name):
        if doc_id.startswith(legacy.TRANSIENT_DOC_PREFIXES):
            continue
        writer.write({'t': 'doc', 'index': name, 'id': doc_id, 'source': source})
    return True


async def project_records(reader: ControlReader, projects_index: str) -> list[dict[str, Any]]:
    """The ``op_projects`` project documents (not the revision counter)."""
    if not await reader.index_exists(projects_index):
        return []
    return [
        source
        async for doc_id, source in reader.iter_docs(projects_index)
        if doc_id.startswith(legacy.PROJECT_DOC_PREFIX)
    ]


def control_index_names(records: list[dict[str, Any]], layout: Layout) -> list[str]:
    """Every control index to export, in a stable order: the two global ones, then each project's."""
    names = [layout.projects_index, layout.global_configs_index]
    for record in sorted(records, key=lambda r: r['slug']):
        for role in legacy.CONTROL_ROLES:
            name = record['resources']['indexes'].get(role)
            if name and name not in names:
                names.append(name)
    return names


async def export_bundle(
    reader: ControlReader,
    layout: Layout,
    store: BlobStore,
    directory: Path,
    *,
    now: dt.datetime,
) -> Manifest:
    """Write ``control_bundle.jsonl.gz`` and ``MANIFEST.json`` into ``directory``."""
    records = await project_records(reader, layout.projects_index)
    with BundleWriter(directory, created_at=now.isoformat(), from_version='0.6.0') as writer:
        for name in control_index_names(records, layout):
            await _write_index(writer, reader, name)
        for area in layout.areas(records):
            _write_area(writer, store, area)
    return writer.manifest
