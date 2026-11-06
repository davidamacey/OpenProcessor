"""Step 3, ``verify``: the loaded rows against the bundle, before anything is deleted.

Any count or checksum mismatch makes the step fail and the upgrade stop with the report; nothing
is deleted (trap T-25). Items whose class disagrees with the registry are reported but are not a
failure: the data is as 0.6.0 left it. A project whose OpenSearch indexes are missing is moved to
status ``failed`` and reported, as a 0.6.0 ``failed`` project is.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from sqlalchemy import func, select, table as sa_table, update

from src.db.models import ConfigRevision, Project, ProjectClass
from src.services.upgrade.class_reconcile import reconcile_classes
from src.services.upgrade.from_060.parse_jobs import import_ledger_rows, pending_queue_rows
from src.services.upgrade.from_060.parse_records import member_ids
from src.services.upgrade.from_060.snapshot import iter_bulk
from src.services.upgrade.from_060.steps import StepContext, verify_all
from src.services.upgrade.from_060.steps.legacy_repair import adds_region_class


if TYPE_CHECKING:
    from pathlib import Path

    from sqlalchemy.ext.asyncio import AsyncSession

    from src.services.upgrade.from_060.admin_client import ControlReader
    from src.services.upgrade.from_060.load import Parsed


@dataclass
class VerifyReport:
    problems: list[str] = field(default_factory=list)
    tables: dict[str, dict[str, int]] = field(default_factory=dict)  # table -> expected, actual
    class_drift: dict[str, dict[str, Any]] = field(default_factory=dict)
    failed_projects: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems

    def to_json(self) -> dict[str, Any]:
        return {
            'ok': self.ok,
            'problems': self.problems,
            'tables': self.tables,
            'class_drift': self.class_drift,
            'failed_projects': self.failed_projects,
        }


def _sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(',', ':')).encode()
    ).hexdigest()


def bulk_counts(bundle: Path, parsed: Parsed) -> dict[str, int]:
    """Rows the streamed files must have produced, counted independently of the load."""
    imports = {(i.project_slug, i.legacy_id): i for i in parsed.imports}
    sources = {(s.project_slug, s.config['legacy_source_id']): s for s in parsed.sources}
    counts = {'dataset_import_images': 0, 'ingest_source_files': 0, 'label_operation_members': 0}
    for file in iter_bulk(bundle):
        area, _, slug = file.area.partition(':')
        parts = file.rel.split('/')
        if area == 'dataset_import' and len(parts) == 3 and (slug, parts[0]) in imports:
            counts['dataset_import_images'] += sum(
                1 for _ in import_ledger_rows(file, None, undone=False)
            )
        elif area == 'bulk_ingest' and parts[-1] == 'queue.sqlite' and len(parts) == 3:
            source = sources.get((slug, parts[1]))
            if source is not None:
                counts['ingest_source_files'] += sum(1 for _ in pending_queue_rows(None, file.data))
        elif area == 'state' and len(parts) == 3 and parts[0] == 'label_ops':
            counts['label_operation_members'] += len(member_ids(file.data))
    return counts


def expected_counts(parsed: Parsed, bulk: dict[str, int]) -> dict[str, int]:
    projects = parsed.projects
    configs = [d for pd in projects for d in pd.configs] + [
        e.doc for e in parsed.global_data.endpoints
    ]
    config_ids: set[tuple[str | None, str, str]] = {
        (pd.project.slug, d.kind, d.name) for pd in projects for d in pd.configs
    }
    config_ids |= {(None, e.doc.kind, e.doc.name) for e in parsed.global_data.endpoints}
    scoped_events = 0
    for pd in projects:
        known = {(k, n) for (owner, k, n) in config_ids if owner in (pd.project.slug, None)}
        scoped_events += sum(
            1 for e in pd.activations if e.to_name is None or (e.kind, e.to_name) in known
        )
    jobs = (
        len(parsed.alive) + sum(1 for i in parsed.imports) + len(parsed.runs) + len(parsed.combines)
    )
    probes = sum(
        1
        for e in parsed.global_data.endpoints
        for p in e.probes
        if p.revision in {r.revision for r in e.doc.revisions}
    )
    return {
        'projects': len(projects),
        'project_indexes': 2 * len(projects),
        'project_slugs': len(projects) + len(parsed.retired),
        'project_classes': sum(len(pd.registry.classes) for pd in projects if pd.registry)
        + sum(
            1 for pd in projects if adds_region_class(pd)
        ),  # the legacy_repair step's seeded class
        'config_documents': len(configs),
        'config_revisions': sum(len(d.revisions) for d in configs),
        'config_activation_events': scoped_events,
        'holdout_freezes': sum(len(h) for h in parsed.holdouts.values()),
        'label_operations': sum(len(o) for o in parsed.label_ops.values()),
        'dataset_imports': len(parsed.imports),
        'ingest_sources': len(parsed.sources),
        'training_runs': len(parsed.runs),
        'models': len(parsed.promotions),
        'promotions': len(parsed.promotions),
        'dataset_exports': len(parsed.exports),
        'combine_runs': len(parsed.combines),
        'vlm_endpoint_probes': probes,
        'projection_states': sum(1 for pd in projects if pd.view_state is not None),
        'jobs': jobs,
        **bulk,
    }


# Tables where the loader writes exactly the rows of the bundle and nothing derived.
_EXACT = (
    'projects', 'project_slugs', 'project_classes', 'config_documents', 'config_revisions',
    'config_activation_events', 'holdout_freezes', 'label_operations', 'dataset_imports',
    'ingest_sources', 'training_runs', 'models', 'promotions', 'dataset_exports', 'combine_runs',
    'vlm_endpoint_probes', 'projection_states', 'jobs', 'project_indexes',
    'dataset_import_images', 'ingest_source_files', 'label_operation_members',
)  # fmt: skip
# The serving-profile step adds one deployment-wide document and revision.
_STEP_EXTRA = {'config_documents': 1, 'config_revisions': 1}


async def _actual_counts(session: AsyncSession) -> dict[str, int]:
    out: dict[str, int] = {}
    for table in _EXACT:
        # table names come from the constant tuple above, never from input
        out[table] = int(
            (await session.execute(select(func.count()).select_from(sa_table(table)))).scalar_one()
        )
    return out


async def _check_counts(
    session: AsyncSession, parsed: Parsed, bundle: Path, report: VerifyReport
) -> None:
    expected = expected_counts(parsed, bulk_counts(bundle, parsed))
    actual = await _actual_counts(session)
    for table in _EXACT:
        want = expected[table] + _STEP_EXTRA.get(table, 0)
        report.tables[table] = {'expected': want, 'actual': actual[table]}
        if want != actual[table]:
            report.problems.append(f'{table}: expected {want} rows, found {actual[table]}')


async def _check_classes(
    session: AsyncSession, parsed: Parsed, project_ids: dict[str, Any], report: VerifyReport
) -> dict[str, dict[int, str]]:
    registries: dict[str, dict[int, str]] = {}
    for pd in parsed.projects:
        rows = (
            await session.execute(
                select(ProjectClass.class_id, ProjectClass.name).where(
                    ProjectClass.project_id == project_ids[pd.project.slug]
                )
            )
        ).all()
        found = dict(rows)
        registries[pd.project.slug] = found
        if pd.registry is None:
            continue
        want = {c.class_id: c.name for c in pd.registry.classes}
        missing = {k: v for k, v in want.items() if found.get(k) != v}
        if missing:
            report.problems.append(
                f'{pd.project.slug}: class ids/names differ from the source at {sorted(missing)[:10]}'
            )
    return registries


async def _check_config_bodies(
    session: AsyncSession,
    parsed: Parsed,
    config_ids: dict[tuple[str | None, str, str], Any],
    report: VerifyReport,
) -> None:
    for (owner, kind, name), config_id in config_ids.items():
        docs = (
            [d for pd in parsed.projects if pd.project.slug == owner for d in pd.configs]
            if owner
            else [e.doc for e in parsed.global_data.endpoints]
        )
        doc = next((d for d in docs if d.kind == kind and d.name == name), None)
        if doc is None:
            continue
        stored: dict[int, Any] = dict(
            (
                await session.execute(
                    select(ConfigRevision.revision, ConfigRevision.body).where(
                        ConfigRevision.config_id == config_id
                    )
                )
            ).all()
        )
        for rev in doc.revisions:
            if _sha(stored.get(rev.revision)) != _sha(rev.body):
                report.problems.append(
                    f'{owner or "deployment"}/{kind}/{name}@{rev.revision}: body checksum differs'
                )


async def _check_indexes(
    session: AsyncSession, reader: ControlReader, parsed: Parsed, report: VerifyReport
) -> None:
    for pd in parsed.projects:
        missing = [n for n in pd.project.indexes.values() if not await reader.index_exists(n)]
        if missing:
            report.failed_projects.append(pd.project.slug)
            await session.execute(
                update(Project)
                .where(Project.slug == pd.project.slug)
                .values(status='failed', pre_delete_status=None, updated_at=func.now())
            )


async def verify_load(
    session: AsyncSession,
    reader: ControlReader,
    parsed: Parsed,
    bundle: Path,
    ctx: StepContext,
) -> VerifyReport:
    """Verify the loaded rows. ``ctx`` is the load's step context (ids and the reader)."""
    report = VerifyReport()
    await _check_counts(session, parsed, bundle, report)
    registries = await _check_classes(session, parsed, ctx.project_ids, report)
    await _check_config_bodies(session, parsed, ctx.config_ids, report)
    await _check_indexes(session, reader, parsed, report)
    for pd in parsed.projects:
        if pd.project.slug in report.failed_projects:
            continue
        drift = await reconcile_classes(
            reader, pd.project.indexes['items'], registries[pd.project.slug]
        )
        report.class_drift[pd.project.slug] = drift.to_json()
    report.problems.extend(await verify_all(ctx))
    return report
