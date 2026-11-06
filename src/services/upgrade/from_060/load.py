"""Step 2, ``load``: the bundle into PostgreSQL, in the caller's one transaction.

The caller opens ``deployment_tx`` and commits; this module writes and never commits. A load that
fails rolls back whole, so a re-run starts clean. The order follows the foreign keys: projects,
then everything that points at a project, then jobs and what points at jobs, then the bulk files
(ledgers, queue databases, label-operation members), which stream through ``COPY`` or the
repository's batched member writer.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from sqlalchemy import delete

from src.db.models import Event
from src.services.upgrade.from_060 import control_rows, direct_rows, ledger_copy
from src.services.upgrade.from_060.model import LoadReport
from src.services.upgrade.from_060.parse_core import parse_all
from src.services.upgrade.from_060.parse_jobs import (
    import_ledger_rows,
    parse_combine,
    parse_exports,
    parse_imports,
    parse_ingest_sources,
    parse_interrupted,
    parse_promotions,
    parse_training,
    pending_queue_rows,
)
from src.services.upgrade.from_060.parse_records import member_ids, parse_holdouts, parse_label_ops
from src.services.upgrade.from_060.snapshot import iter_bulk, read_snapshot


if TYPE_CHECKING:
    import datetime as dt
    import uuid
    from pathlib import Path

    from sqlalchemy.ext.asyncio import AsyncSession

    from src.clients.storage.base import BlobStore
    from src.services.upgrade.from_060.admin_client import ControlReader
    from src.services.upgrade.from_060.model import (
        ExportSrc,
        GlobalData,
        HoldoutSrc,
        ImportSrc,
        IngestSourceSrc,
        JobRow,
        LabelOpSrc,
        ProjectData,
        PromotionSrc,
        RetiredSlug,
        TrainingRunSrc,
    )
    from src.services.upgrade.from_060.snapshot import Snapshot

REDUCER_KEY = 'projects/{slug}/umap/current.bin'


class Parsed:
    """Everything the bundle yields, parsed once and shared by the load and the verify."""

    def __init__(self, snap: Snapshot, *, now: dt.datetime) -> None:
        self.global_data: GlobalData
        self.projects: list[ProjectData]
        self.retired: list[RetiredSlug]
        self.global_data, self.projects, self.retired = parse_all(snap, now=now)
        by_slug = {p.project.slug: p.project for p in self.projects}
        self.holdouts: dict[str, tuple[HoldoutSrc, ...]] = {}
        self.label_ops: dict[str, tuple[LabelOpSrc, ...]] = {}
        self.imports: list[ImportSrc] = []
        self.sources: list[IngestSourceSrc] = []
        self.runs: list[TrainingRunSrc] = []
        self.exports: list[ExportSrc] = []
        self.alive: list[JobRow] = []
        for slug, project in by_slug.items():
            self.holdouts[slug] = parse_holdouts(snap, project)
            self.label_ops[slug] = parse_label_ops(snap, project)
            self.imports.extend(parse_imports(snap, project))
            self.sources.extend(parse_ingest_sources(snap, project))
            self.runs.extend(parse_training(snap, project))
            self.exports.extend(parse_exports(snap, project))
            self.alive.extend(parse_interrupted(snap, project))
        runs = {(r.project_slug, r.legacy_id): r for r in self.runs}
        self.promotions: list[PromotionSrc] = parse_promotions(snap, by_slug, runs)
        self.combines = parse_combine(snap, by_slug)


def _stage_blobs(parsed: Parsed, store: BlobStore) -> dict[str, str | None]:
    """Write the files the rows point at (holdout snapshots, UMAP reducers) to storage; idempotent.
    Returns the reducer key per project slug."""
    reducers: dict[str, str | None] = {}
    for holdouts in parsed.holdouts.values():
        for h in holdouts:
            if h.snapshot_key:
                store.write_atomic(h.snapshot_key, json.dumps(h.record, indent=2).encode('utf-8'))
    for pd in parsed.projects:
        view = pd.view_state
        key = None
        if view is not None and view.reducer_b64:
            key = REDUCER_KEY.format(slug=pd.project.slug)
            store.write_atomic(key, base64.b64decode(view.reducer_b64))
        reducers[pd.project.slug] = key
    return reducers


async def _load_configs(
    session: AsyncSession,
    parsed: Parsed,
    project_ids: dict[str, uuid.UUID],
    report: LoadReport,
) -> dict[tuple[str | None, str, str], uuid.UUID]:
    ids: dict[tuple[str | None, str, str], uuid.UUID] = {}
    for endpoint in parsed.global_data.endpoints:
        doc = endpoint.doc
        ids[(None, doc.kind, doc.name)] = await control_rows.load_vlm_endpoint(session, endpoint)
        report.add('config_documents')
        report.add('config_revisions', len(doc.revisions))
        known = {r.revision for r in doc.revisions}
        report.add('vlm_endpoint_probes', sum(1 for p in endpoint.probes if p.revision in known))
    for pd in parsed.projects:
        for doc in pd.configs:
            pid = project_ids[pd.project.slug]
            ids[(pd.project.slug, doc.kind, doc.name)] = await control_rows.load_config(
                session, pid, doc
            )
            report.add('config_documents')
            report.add('config_revisions', len(doc.revisions))
    return ids


async def _load_project_rows(
    session: AsyncSession,
    parsed: Parsed,
    project_ids: dict[str, uuid.UUID],
    config_ids: dict[tuple[str | None, str, str], uuid.UUID],
    reducers: dict[str, str | None],
    report: LoadReport,
) -> None:
    for pd in parsed.projects:
        slug = pd.project.slug
        pid = project_ids[slug]
        if pd.registry is not None:
            await control_rows.load_registry(session, pid, pd.registry)
            report.add('project_classes', len(pd.registry.classes))
        scoped = {
            (kind, name): cid
            for (owner, kind, name), cid in config_ids.items()
            if owner in (slug, None)
        }
        await control_rows.load_activations(session, pid, pd.activations, scoped)
        report.add(
            'config_activation_events',
            sum(1 for e in pd.activations if (e.kind, e.to_name) in scoped),
        )
        if pd.settings is not None:
            await control_rows.load_settings(session, pid, pd.settings)
        if pd.keymap is not None:
            await control_rows.load_keymap(session, pid, pd.keymap)
        for key, state in {
            **({'cluster_refresh': pd.cluster_refresh} if pd.cluster_refresh else {}),
            **pd.markers,
        }.items():
            await control_rows.load_state_marker(session, pid, key, state)
            report.add('project_state')
        holdouts = parsed.holdouts[slug]
        if holdouts:
            await control_rows.load_holdouts(session, pid, holdouts)
            report.add('holdout_freezes', len(holdouts))
        if pd.view_state is not None:
            await direct_rows.projection_state(session, pid, pd.view_state, reducers[slug])
            report.add('projection_states')


async def _load_jobs(
    session: AsyncSession, parsed: Parsed, project_ids: dict[str, uuid.UUID], report: LoadReport
) -> None:
    n = await direct_rows.exports(session, parsed.exports, project_ids)
    report.add('dataset_exports', n)
    export_ids = {
        (e.project_slug, e.artifact_prefix.rsplit('/', 1)[-1]): e.id for e in parsed.exports
    }
    n = await direct_rows.training_runs(session, parsed.runs, project_ids, export_ids)
    report.add('training_runs', n)
    run_ids = {(r.project_slug, r.legacy_id): r.id for r in parsed.runs}
    n = await direct_rows.promotions(session, parsed.promotions, project_ids, run_ids)
    report.add('promotions', n)
    n = await direct_rows.dataset_imports(session, parsed.imports, project_ids)
    report.add('dataset_imports', n)
    n = await direct_rows.ingest_sources(session, parsed.sources, project_ids)
    report.add('ingest_sources', n)
    n = await direct_rows.combines(session, parsed.combines, project_ids)
    report.add('combine_runs', n)
    n = await direct_rows.jobs(session, parsed.alive, project_ids)
    report.add('interrupted_jobs', n)


PRIOR_BATCH = 1000


async def member_priors(
    reader: ControlReader, items_index: str, legacy_op_id: str, ids: list[str]
) -> dict[str, Any]:
    """The state each item had before the operation: the ``class_id_history`` entry whose
    ``op_id`` is the operation (it carries the class fields, op id, writer and restorable). An
    item with no such entry (pruned, or deleted) has no prior."""
    priors: dict[str, Any] = {}
    for start in range(0, len(ids), PRIOR_BATCH):
        histories = await reader.get_field(
            items_index, ids[start : start + PRIOR_BATCH], 'class_id_history'
        )
        for item_id, history in histories.items():
            entries = [e for e in history or [] if e.get('op_id') == legacy_op_id]
            if entries:
                priors[item_id] = entries[-1]
    return priors


async def _load_bulk(
    session: AsyncSession,
    bundle: Path,
    parsed: Parsed,
    project_ids: dict[str, uuid.UUID],
    report: LoadReport,
    reader: ControlReader,
) -> None:
    imports = {(i.project_slug, i.legacy_id): i for i in parsed.imports}
    sources = {(s.project_slug, s.config['legacy_source_id']): s for s in parsed.sources}
    ops = {(slug, op.legacy_id): op for slug, group in parsed.label_ops.items() for op in group}
    done_ops: set[tuple[str, str]] = set()
    indexes = {pd.project.slug: pd.project.indexes['items'] for pd in parsed.projects}

    def items_index(slug: str) -> str:
        return indexes[slug]

    for file in iter_bulk(bundle):
        area, _, slug = file.area.partition(':')
        parts = file.rel.split('/')
        if area == 'dataset_import' and len(parts) == 3 and parts[1] == 'ledger':
            imported = imports.get((slug, parts[0]))
            if imported is not None:
                n = await ledger_copy.copy_import_images(
                    session, import_ledger_rows(file, imported.id, undone=imported.undone)
                )
                report.add('dataset_import_images', n)
        elif (
            area == 'bulk_ingest'
            and len(parts) == 3
            and parts[0] == 'sources'
            and parts[2] == 'queue.sqlite'
        ):
            source = sources.get((slug, parts[1]))
            if source is not None:
                n = await ledger_copy.copy_ingest_files(
                    session, pending_queue_rows(source.id, file.data)
                )
                report.add('ingest_source_files', n)
        elif area == 'state' and len(parts) == 3 and parts[0] == 'label_ops':
            op = ops.get((slug, parts[1]))
            if op is not None:
                ids = member_ids(file.data)
                priors = await member_priors(reader, items_index(slug), op.legacy_id, ids)
                await control_rows.load_label_op(session, project_ids[slug], op, ids, priors)
                report.add('label_operations')
                report.add('label_operation_members', len(ids))
                done_ops.add((slug, op.legacy_id))
    for key, op in ops.items():  # operations whose member list is missing or was never written
        if key not in done_ops:
            await control_rows.load_label_op(session, project_ids[key[0]], op, [], {})
            report.add('label_operations')


@dataclass
class LoadResult:
    report: LoadReport
    parsed: Parsed
    project_ids: dict[str, uuid.UUID]
    config_ids: dict[tuple[str | None, str, str], uuid.UUID]


async def run_load(
    session: AsyncSession,
    bundle: Path,
    store: BlobStore,
    reader: ControlReader,
    *,
    now: dt.datetime,
) -> LoadResult:
    """Write every row. The caller owns the transaction (``deployment_tx``) and the commit."""
    snap = read_snapshot(bundle)
    parsed = Parsed(snap, now=now)
    report = LoadReport()
    reducers = _stage_blobs(parsed, store)

    n = await direct_rows.retired_slugs(session, parsed.retired)
    report.add('retired_slugs', n)
    project_ids: dict[str, uuid.UUID] = {}
    for pd in parsed.projects:
        project_ids[pd.project.slug] = await control_rows.load_project(session, pd.project)
        report.add('projects')
    config_ids = await _load_configs(session, parsed, project_ids, report)
    if await direct_rows.local_vlm_desired(session, parsed.global_data):
        report.add('deployment_settings')
    await _load_project_rows(session, parsed, project_ids, config_ids, reducers, report)
    await _load_jobs(session, parsed, project_ids, report)
    await _load_bulk(session, bundle, parsed, project_ids, report, reader)
    await direct_rows.set_projects_revision(session, parsed.global_data.projects_revision)
    # The events table is only the 24-hour UI feed, and the repositories' creates wrote some;
    # nothing a reader could resume from survives the upgrade (decision D-73).
    await session.execute(delete(Event))
    return LoadResult(report, parsed, project_ids, config_ids)
