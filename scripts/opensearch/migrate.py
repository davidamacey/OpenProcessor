"""The one place that creates and migrates OpenSearch indexes (review fix R-7).

Called once by ``scripts/db/migrate.py`` after Alembic, in the one-shot ``db-migrate`` step, so
no API worker repeats it. Idempotent: an index that already carries the current schema stamp
costs one read and no write.

* The deployment-wide ``visual_search_*`` indexes are always created.
* While the 0.6.0 control data is not loaded (``upgrade_state`` ``in_progress`` or the
  ``op_projects`` index present) the projects are not in PostgreSQL yet, so only those global
  indexes are touched; the second ``db-migrate`` of ``openprocessor upgrade`` finds the loaded
  projects and creates, migrates and stamps their indexes.
* Otherwise the ``default`` project of a fresh install is created, then each project is bound in turn and its indexes are created, migrated and stamped
  (``src.services.curation.index_bootstrap.ensure_indexes``), then its kNN graph is warmed.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from src.clients.opensearch.client import OpenSearchClient
from src.config.curation_roles import IndexRole
from src.config.env import env_str
from src.config.project_context import bind_project
from src.core.logging import get_logger
from src.db.engine import dispose_engines, one_off_connection
from src.db.repos import projects as projects_repo
from src.db.schema_check import legacy_projects_index_exists, upgrade_state
from src.db.session import async_session_factory, deployment_tx
from src.services.curation.index_bootstrap import ensure_indexes, warm_knn_indexes
from src.services.projects.bootstrap import bootstrap_default_project
from src.services.projects.records import record_from_row
from src.services.upgrade.from_060 import analysis_settings
from src.services.upgrade.from_060.admin_client import build_client


if TYPE_CHECKING:
    from collections.abc import Sequence

    from src.config.projects import ProjectRecord

logger = get_logger(__name__)

# A project in these states owns indexes; a building one is being created by an API replica that
# creates its indexes itself, and a deleting/deleted/failed one is not served.
MIGRATED_STATUSES = frozenset({'active', 'archived'})


class IndexMigrationError(RuntimeError):
    """An index could not be created or migrated; ``db-migrate`` fails and nothing starts."""


@dataclass
class MigrateReport:
    global_indexes: dict[str, bool] = field(default_factory=dict)
    projects: list[str] = field(default_factory=list)
    skipped_for_upgrade: bool = False


async def migrate(
    client: Any,
    global_client: Any,
    records: Sequence[ProjectRecord],
    *,
    upgrade_pending: bool,
    warm: bool = True,
) -> MigrateReport:
    """Create the global indexes and, unless the 0.6.0 data is not loaded, migrate each project.

    ``client`` is the raw ``AsyncOpenSearch`` for the project indexes; ``global_client`` is the
    ``OpenSearchClient`` that owns ``create_all_indexes``.
    """
    report = MigrateReport(skipped_for_upgrade=upgrade_pending)
    report.global_indexes = await global_client.create_all_indexes(force_recreate=False)
    failed = [name for name, ok in report.global_indexes.items() if not ok]
    if failed:
        raise IndexMigrationError(f'could not create the global indexes: {", ".join(failed)}')
    if upgrade_pending:
        logger.info('opensearch_migrate_global_only', reason='the 0.6.0 data is not loaded')
        return report
    for record in sorted(records, key=lambda r: r.slug):
        if record.status not in MIGRATED_STATUSES:
            continue
        with bind_project(record):
            if not await ensure_indexes(client):
                raise IndexMigrationError(
                    f"project '{record.slug}': its indexes could not be created or migrated"
                )
            if warm:
                await warm_knn_indexes(client)
        report.projects.append(record.slug)
        logger.info('opensearch_migrate_project', project=record.slug)
    return report


def upgrade_pending() -> bool:
    """Whether the 0.6.0 control data is still to be loaded (``upgrade_state`` ``in_progress`` or
    the ``op_projects`` index present): the projects are not in PostgreSQL yet."""
    with one_off_connection('app', direct=False, application_name='op-os-migrate-gate') as conn:
        in_progress = upgrade_state(conn) == 'in_progress'
    return in_progress or legacy_projects_index_exists()


async def read_records() -> list[ProjectRecord]:
    """Every project, from the ``projects`` table."""
    async with (
        async_session_factory()() as session,
        deployment_tx(session, 'opensearch migrate: read the projects'),
    ):
        rows = await session.run_sync(projects_repo.list_projects_sync)
    return [record_from_row(row) for row in rows]


async def run() -> MigrateReport:
    """The whole step: what ``db-migrate`` runs after Alembic."""
    url = env_str('OPENSEARCH_URL', 'http://opensearch:9200')
    pending = upgrade_pending()
    global_client = OpenSearchClient(hosts=[url])
    client = build_client(url)
    try:
        records: list[ProjectRecord] = []
        if not pending:
            await bootstrap_default_project()
            records = await read_records()
        return await migrate(client, global_client, records, upgrade_pending=pending)
    finally:
        await client.close()
        await global_client.close()
        await dispose_engines()


async def run_analysis_step(*, dry_run: bool) -> analysis_settings.AnalysisReport:
    """Step ``analysis-settings`` on its own: list (``dry_run``) or apply the compact text
    normalizer to the items index of every active or archived project."""
    client = build_client(env_str('OPENSEARCH_URL', 'http://opensearch:9200'))
    try:
        records = await read_records()
        names = [
            record.resources.indexes[IndexRole.ITEMS]
            for record in records
            if record.status in MIGRATED_STATUSES
        ]

        def say(line: str) -> None:
            sys.stdout.write(f'{line}\n')

        if dry_run:
            return await analysis_settings.dry_run(client, names, say=say)
        return await analysis_settings.apply(client, names, say=say)
    finally:
        await client.close()
        await dispose_engines()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--step', choices=('analysis-settings',), help='run only this step (default: the migrate)'
    )
    parser.add_argument('--dry-run', action='store_true', help='with --step: change nothing')
    args = parser.parse_args(argv)
    if args.dry_run and args.step is None:
        parser.error('--dry-run needs --step')
    if args.step is None:
        asyncio.run(run())
        return 0
    try:
        asyncio.run(run_analysis_step(dry_run=args.dry_run))
    except analysis_settings.AnalysisSettingsError as exc:
        sys.stderr.write(f'{exc}\n')
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())
