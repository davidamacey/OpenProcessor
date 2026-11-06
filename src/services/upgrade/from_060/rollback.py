"""``rollback_to_060``: rebuild the 0.6.0 control state from the bundle (02c section 2.3).

Runs with OpenSearch up and every 0.6.0 and 0.7.0 API and worker stopped: a 0.6.0 API that
started first would create the missing control indexes empty. In order: recreate each control
index from the mappings in the bundle and re-index its documents; restore the state and job files
(the run logs from their storage keys) and give ``class_registry.json`` its name back; then check
what was restored against the bundle. Marking the PostgreSQL data abandoned and starting the 0.6.0
services are the installer's steps and come only after this one reports no problem.

Anything created after the upgrade (classes, configs, jobs) exists only in PostgreSQL and is not
carried back.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from src.services.upgrade.from_060 import sources as legacy
from src.services.upgrade.from_060.bundle import check_bundle, iter_records
from src.services.upgrade.from_060.snapshot import decode_file
from src.utils.atomic_write import atomic_write_bytes


if TYPE_CHECKING:
    from pathlib import Path

    from src.clients.storage.base import BlobStore
    from src.services.upgrade.from_060.admin_client import ControlReader, ControlWriter
    from src.services.upgrade.from_060.layout import Layout


class RollbackError(RuntimeError):
    """The bundle cannot be restored over what is there."""


@dataclass
class RollbackReport:
    indexes: dict[str, int] = field(default_factory=dict)  # name -> documents restored
    files_restored: int = 0
    files_kept: int = 0
    registries_renamed_back: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems


async def _restore_indexes(
    bundle: Path, reader: ControlReader, writer: ControlWriter, report: RollbackReport
) -> None:
    definitions: dict[str, tuple[dict[str, object], dict[str, object]]] = {}
    docs: dict[str, list[tuple[str, dict[str, object]]]] = {}
    for record in iter_records(bundle):
        if record['t'] == 'index':
            definitions[record['name']] = (record['mappings'], record['settings'])
            docs.setdefault(record['name'], [])
        elif record['t'] == 'doc':
            docs[record['index']].append((record['id'], record['source']))
    for name, (mappings, settings) in definitions.items():
        if await reader.index_exists(name):
            present = {doc_id async for doc_id, _ in reader.iter_docs(name)}
            wanted = {doc_id for doc_id, _ in docs[name]}
            if present and present != wanted:
                msg = f'{name} exists with {len(present)} documents that are not the bundle'
                raise RollbackError(msg)
            if present:
                report.indexes[name] = len(present)
                continue
            await writer.delete_index(name)  # an empty index a 0.6.0 process created
        await writer.create_index(name, mappings, settings)
        report.indexes[name] = await writer.bulk_index(name, docs[name])


def _restore_file(target: Path, data: bytes, mtime: str | None, report: RollbackReport) -> None:
    if (
        target.is_file()
        and hashlib.sha256(target.read_bytes()).digest() == hashlib.sha256(data).digest()
    ):
        report.files_kept += 1
        return
    atomic_write_bytes(target, data)
    if mtime:
        import datetime as dt

        stamp = dt.datetime.fromisoformat(mtime).timestamp()
        os.utime(target, (stamp, stamp))
    report.files_restored += 1


def _restore_files(bundle: Path, layout: Layout, store: BlobStore, report: RollbackReport) -> None:
    from src.services.upgrade.from_060.delete_sources import KEPT_AREA_PREFIXES

    records = [r for r in iter_records(bundle) if r['t'] in {'file', 'blob'}]
    projects = [
        r['source']
        for r in iter_records(bundle)
        if r['t'] == 'doc'
        and r['index'] == layout.projects_index
        and str(r['id']).startswith(legacy.PROJECT_DOC_PREFIX)
    ]
    areas = {area.label: area for area in layout.areas(projects)}
    for record in records:
        if record['t'] == 'blob':
            area, rel, data, mtime = areas[record['source_area']], record['source_rel'], None, None
            with store.open_read(record['key']) as fh:
                data = fh.read()
        else:
            file = decode_file(record)
            area, rel, data, mtime = areas[file.area], file.rel, file.data, file.mtime
        if area.label.startswith(KEPT_AREA_PREFIXES):
            continue
        target = area.target(rel)
        if area.label.startswith('class_registry:') and rel == legacy.CLASS_REGISTRY_FILE:
            renamed = target.with_name(legacy.CLASS_REGISTRY_RENAMED)
            if renamed.is_file() and not target.exists():
                renamed.replace(target)
                report.registries_renamed_back.append(area.label.partition(':')[2])
        _restore_file(target, data, mtime, report)


async def _verify(
    bundle: Path,
    reader: ControlReader,
    layout: Layout,
    report: RollbackReport,
    expected: dict[str, int],
) -> None:
    for family, count in expected.items():
        if not family.startswith('doc:'):
            continue
        index = family.removeprefix('doc:')
        found = sum([1 async for _ in reader.iter_docs(index)])
        if found != count:
            report.problems.append(
                f'{index}: restored {found} documents, the manifest says {count}'
            )
    projects = [
        r['source']
        for r in iter_records(bundle)
        if r['t'] == 'doc'
        and r['index'] == layout.projects_index
        and str(r['id']).startswith(legacy.PROJECT_DOC_PREFIX)
    ]
    areas = {area.label: area for area in layout.areas(projects)}
    for record in iter_records(bundle):
        if record['t'] not in {'file', 'blob'}:
            continue
        label = record['area'] if record['t'] == 'file' else record['source_area']
        rel = record['rel'] if record['t'] == 'file' else record['source_rel']
        if label.startswith(('exports:', 'models')):
            continue
        target = areas[label].target(rel)
        if not target.is_file():
            report.problems.append(f'{label}/{rel}: not restored')
        elif hashlib.sha256(target.read_bytes()).hexdigest() != record['sha256']:
            report.problems.append(f'{label}/{rel}: differs from the bundle')


async def rollback(
    bundle: Path,
    reader: ControlReader,
    writer: ControlWriter,
    layout: Layout,
    store: BlobStore,
) -> RollbackReport:
    """Restore the 0.6.0 control state. A non-empty ``problems`` means do not start 0.6.0."""
    manifest = check_bundle(bundle)
    report = RollbackReport()
    await _restore_indexes(bundle, reader, writer, report)
    _restore_files(bundle, layout, store, report)
    await _verify(bundle, reader, layout, report, manifest.counts)
    return report
