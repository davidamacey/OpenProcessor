#!/usr/bin/env python3
"""Move each project's class registry between ``class_registry.json`` and its OpenSearch document.

Since 0.7.0 the class registry is one OpenSearch document per project; the file is an export
snapshot only. A project that still has the file and no document answers 409
``registry_not_migrated`` on every class route until this has run for it. It runs through
``./openprocessor curation migrate-registry`` (and by itself on ``openprocessor upgrade``).

    python -m scripts.curation.migrate_class_registry --all --dry-run   # plan, writes nothing
    python -m scripts.curation.migrate_class_registry --all             # migrate every project
    python -m scripts.curation.migrate_class_registry --project demo   # default: --project default
    python -m scripts.curation.migrate_class_registry --all --reverse   # rollback to the file

Forward, per project: no file means nothing to do. No document: the document is created from the
file with every class id preserved and ``next_class_id = max(id) + 1``. A document equal to the
file only gets the file renamed. A document written back to the file by an earlier ``--reverse``
(the file is then the authority of the downgrade period) is replaced from the file, and so is the
still-untouched empty document of a new project when a file is copied in. Any other
difference is an error for that project and nothing is written for it. The file is renamed to
``class_registry.json.migrated``, never deleted.

``--reverse`` writes the document's current classes to ``class_registry.json`` (so classes added
after the migration are not lost to an older version), keeps the document and marks it.

Each run writes a JSON report to stdout and to
``<project_state_dir>/registry_migration/<timestamp>-{dry-run,run,reverse}.json``.
Exit codes: 0 done, 1 a project failed or conflicts, 2 bad arguments or OpenSearch unreachable.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ruff: noqa: E402
from opensearchpy.exceptions import OpenSearchException

from src.clients.curation_opensearch import ClassRegistryFile
from src.clients.curation_opensearch.registry import get_class_registry
from src.clients.curation_opensearch.registry_transfer import (
    mark_reversed,
    replace_from_file,
    seed_from_file,
)
from src.config.project_context import bind_project
from src.services.projects.guard import make_script_opensearch
from src.services.projects.script_binding import (
    add_project_argument,
    bind_script_project,
    script_project_registry,
)
from src.utils.atomic_write import atomic_write_json


if TYPE_CHECKING:
    from src.clients.curation_opensearch.registry import RegistryDoc
    from src.config.projects import ProjectRecord


MIGRATED_SUFFIX = '.migrated'


def _timestamp() -> str:
    return datetime.now(UTC).strftime('%Y%m%dT%H%M%S%fZ')


def _entries(doc: ClassRegistryFile) -> list[dict[str, Any]]:
    """The classes as compared between file and document. ``added_at`` is left out: an older
    file without it gets the parse time, which differs on every read."""
    return sorted(
        (c.model_dump(exclude={'added_at'}) for c in doc.classes), key=lambda e: e['class_id']
    )


def _id_range(doc: ClassRegistryFile) -> list[int] | None:
    ids = [c.class_id for c in doc.classes]
    return [min(ids), max(ids)] if ids else None


def _rename_to_migrated(path: Path) -> Path:
    """Rename the file to ``.migrated`` (one already there is kept under a timestamp)."""
    migrated = path.with_name(path.name + MIGRATED_SUFFIX)
    if migrated.exists():
        migrated.replace(path.with_name(f'{migrated.name}.{_timestamp()}'))
    path.replace(migrated)
    return migrated


def _read_file(path: Path) -> ClassRegistryFile:
    return ClassRegistryFile.model_validate(json.loads(path.read_text(encoding='utf-8')))


async def _forward(client: Any, path: Path, *, dry_run: bool, report: dict[str, Any]) -> None:
    store = get_class_registry()
    if not path.exists():
        report['action'] = 'nothing_to_do'
        return
    file_doc = _read_file(path)
    report['file_classes'] = len(file_doc.classes)
    report['id_range'] = _id_range(file_doc)
    found = await store.read_versioned(client)
    if found is None:
        report['action'] = 'seed'
        if dry_run:
            return
        if await seed_from_file(client, store, file_doc):
            report['archived_to'] = str(_rename_to_migrated(path))
            report['applied'] = True
            return
        # Another process created the document between the read and the create: compare it.
        found = await store.read_versioned(client)
        if found is None:
            msg = 'the registry document vanished while migrating'
            raise ValueError(msg)
    doc: RegistryDoc = found[0]
    report['doc_classes'] = len(doc.classes)
    if _entries(doc) == _entries(file_doc):
        report['action'] = 'rename_only'
    elif doc.reversed_to_file_at:
        report['action'] = 'reseed_after_reverse'
    elif not doc.classes and doc.registry_revision == 0:
        # The empty document a new project gets on first use, never written since: a file
        # copied in afterwards (the worked example) is the registry to load.
        report['action'] = 'reseed_empty'
    else:
        report['action'] = 'conflict'
        report['error'] = (
            f'the registry document ({len(doc.classes)} classes, revision '
            f'{doc.registry_revision}) and {path.name} ({len(file_doc.classes)} classes) differ and '
            'no rollback explains it; nothing was written. Decide which is right, then remove '
            f'or rename {path.name} (keep the document) or delete the document (keep the file).'
        )
        return
    if dry_run:
        return
    if report['action'] in ('reseed_after_reverse', 'reseed_empty'):
        await replace_from_file(client, store, file_doc)
    report['archived_to'] = str(_rename_to_migrated(path))
    report['applied'] = True


async def _reverse(client: Any, path: Path, *, dry_run: bool, report: dict[str, Any]) -> None:
    store = get_class_registry()
    found = await store.read_versioned(client)
    if found is None:
        report['action'] = 'nothing_to_do'
        return
    doc = found[0]
    report['doc_classes'] = len(doc.classes)
    report['id_range'] = _id_range(doc)
    report['action'] = 'reverse'
    if dry_run:
        return
    snapshot = ClassRegistryFile(classes=doc.classes, updated_at=doc.updated_at)
    if path.exists():
        path.replace(path.with_name(f'{path.name}{MIGRATED_SUFFIX}.{_timestamp()}'))
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, snapshot.model_dump(), indent=2)
    migrated = path.with_name(path.name + MIGRATED_SUFFIX)
    if migrated.exists():
        migrated.replace(path.with_name(f'{path.name}{MIGRATED_SUFFIX}.{_timestamp()}'))
    await mark_reversed(client, store)
    report['applied'] = True


async def migrate_project(
    client: Any, record: ProjectRecord, *, dry_run: bool, reverse: bool
) -> dict[str, Any]:
    """The report for one project; never raises for a per-project failure."""
    path = record.resources.class_registry_path
    report: dict[str, Any] = {
        'project': record.slug,
        'file': str(path),
        'action': None,
        'applied': False,
    }
    try:
        with bind_project(record):
            if reverse:
                await _reverse(client, path, dry_run=dry_run, report=report)
            else:
                await _forward(client, path, dry_run=dry_run, report=report)
    except (OpenSearchException, OSError, ValueError) as exc:
        report['action'] = report['action'] or 'error'
        report['error'] = f'{type(exc).__name__}: {exc}'
    return report


def _write_report(record: ProjectRecord, kind: str, report: dict[str, Any], stamp: str) -> None:
    target = record.resources.project_state_dir / 'registry_migration' / f'{stamp}-{kind}.json'
    try:
        atomic_write_json(target, report, indent=2)
    except OSError as exc:
        report['report_write_error'] = str(exc)


async def run(args: argparse.Namespace) -> int:
    client = make_script_opensearch([args.opensearch_url], use_ssl=False, timeout=60)
    try:
        registry = script_project_registry()
        await registry.ensure_fresh()
        if not registry.refreshed:
            print(
                'error: the project registry could not be read (is OpenSearch up?)', file=sys.stderr
            )
            return 2
        records = [r for r in registry.existing_projects() if args.all or r.slug == args.project]
        if not records:
            print(f'error: no such project {args.project!r}', file=sys.stderr)
            return 2
        kind = 'reverse' if args.reverse else ('dry-run' if args.dry_run else 'run')
        stamp = datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')
        reports = []
        for record in sorted(records, key=lambda r: r.slug):
            report = await migrate_project(
                client, record, dry_run=args.dry_run, reverse=args.reverse
            )
            _write_report(record, kind, report, stamp)
            reports.append(report)
        summary = {
            'mode': kind,
            'at': stamp,
            'projects': reports,
            'failed': [r['project'] for r in reports if r.get('error')],
        }
        print(json.dumps(summary, indent=2))
        return 1 if summary['failed'] else 0
    finally:
        await client.close()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_project_argument(p)
    p.add_argument(
        '--all', action='store_true', help='Every project that exists (instead of --project)'
    )
    p.add_argument('--dry-run', action='store_true', help='Report the plan, write nothing')
    p.add_argument(
        '--reverse',
        action='store_true',
        help='Write each document back to class_registry.json (rollback to 0.6.x)',
    )
    p.add_argument(
        '--opensearch-url', default=os.environ.get('OPENSEARCH_URL', 'http://opensearch:9200')
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.all:
        bind_script_project(args.project)
    return asyncio.run(run(args))


if __name__ == '__main__':
    sys.exit(main())
