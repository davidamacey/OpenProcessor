"""RR-7: the delete removes a project's rows per table in batches, never in one cascade."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest

from projects._delete_rows import (
    big_partition,
    bound_engine,
    connect,
    count,
    deleting_project,
    job_with_items,
)
from src.services.projects.delete import plan_purge, purge_project_rows


if TYPE_CHECKING:
    from fixtures.postgres import PgDatabase

pytestmark = pytest.mark.db

ROWS = 200_000


class Killed(Exception):  # noqa: N818 - a test double for a worker that dies mid-table
    pass


def test_two_hundred_thousand_rows_delete_under_a_short_statement_timeout(pg: PgDatabase) -> None:
    with connect(pg) as conn:
        project = deleting_project(conn)
        partition = big_partition(conn, project, ROWS)
        job = job_with_items(conn, project, ROWS)
        batches: list[tuple[str, int]] = []

        async def seen(table: str, deleted: int) -> None:
            batches.append((table, deleted))

        with bound_engine(pg, statement_timeout='30s'):
            totals = asyncio.run(purge_project_rows(project, on_rows_deleted=seen))

        assert totals['cluster_memberships'] == ROWS
        assert totals['job_items'] == ROWS
        assert count(conn, 'cluster_memberships', 'partition_id', partition) == 0
        assert count(conn, 'job_items', 'job_id', job) == 0
        assert count(conn, 'jobs', 'project_id', project) == 0
        # one callback per 10,000-row batch, children before their parents
        assert [d for t, d in batches if t == 'job_items'] == list(range(10_000, ROWS + 1, 10_000))
        tables = list(dict.fromkeys(t for t, _ in batches))
        assert tables.index('cluster_memberships') < tables.index('cluster_partitions')
        assert tables.index('job_items') < tables.index('jobs')


def test_a_kill_between_batches_resumes_where_the_table_stopped(pg: PgDatabase) -> None:
    with connect(pg) as conn:
        project = deleting_project(conn)
        partition = big_partition(conn, project, 45_000)

        async def die_after_two_batches(table: str, deleted: int) -> None:
            if table == 'cluster_memberships' and deleted == 20_000:
                raise Killed

        with bound_engine(pg):
            with pytest.raises(Killed):
                asyncio.run(purge_project_rows(project, on_rows_deleted=die_after_two_batches))
            assert count(conn, 'cluster_memberships', 'partition_id', partition) == 25_000

            totals = asyncio.run(purge_project_rows(project))

        assert totals['cluster_memberships'] == 25_000  # only what the first run left
        assert count(conn, 'cluster_memberships', 'partition_id', partition) == 0
        assert count(conn, 'cluster_partitions', 'project_id', project) == 0


def test_the_order_is_read_from_the_catalog_so_a_new_table_is_covered(pg: PgDatabase) -> None:
    """Every table that carries a ``project_id`` is in the plan, children before parents."""
    from src.services.projects.delete import _FOREIGN_KEYS, _PROJECT_ID_TABLES, ForeignKey

    with connect(pg) as conn:
        fks = [
            ForeignKey(r[0], r[1], r[2], tuple(r[3]), tuple(r[4]))
            for r in conn.execute(_FOREIGN_KEYS.text).fetchall()
        ]
        id_tables = [r[0] for r in conn.execute(_PROJECT_ID_TABLES.text).fetchall()]
    plan = [t for t, _ in plan_purge(fks, id_tables)]
    kept = {fk.child for fk in fks if fk.parent == 'projects' and fk.on_delete in 'nd'}
    assert not (set(id_tables) - kept - set(plan)), 'a project-owned table is missing from the plan'
    position = {t: i for i, t in enumerate(plan)}
    for fk in fks:
        if fk.child in position and fk.parent in position and fk.child != fk.parent:
            assert position[fk.child] < position[fk.parent], (
                f'{fk.child} must go before {fk.parent}'
            )
