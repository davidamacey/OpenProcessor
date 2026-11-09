"""PostgreSQL rows (a job, a near-duplicate run and group) seeded per project, so the read
routes that answer from the database get ids that exist."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

import psycopg

from curation.cross_project_leak._pg_world import seed_world


if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path


def _one(cursor: Any) -> Any:
    row = cursor.fetchone()
    assert row is not None
    return row[0]


def _profile(conn: Any, project: uuid.UUID) -> tuple[uuid.UUID, int]:
    config_id = _one(
        conn.execute(
            'INSERT INTO config_documents (project_id, kind, name, current_revision, updated_by) '
            "VALUES (%s, 'near_dup_profile', %s, 1, 'test') RETURNING id",
            (project, f'sweep-{uuid.uuid4().hex[:8]}'),
        )
    )
    conn.execute(
        'INSERT INTO config_revisions (config_id, revision, body, body_sha256, created_by) '
        "VALUES (%s, 1, '{}', 'x', 'test')",
        (config_id,),
    )
    return config_id, 1


def _job(conn: Any, project: uuid.UUID) -> uuid.UUID:
    return _one(
        conn.execute(
            'INSERT INTO jobs (project_id, kind, queue, status, max_queue_wait, requested_by) '
            "VALUES (%s, 'noop', 'jobs', 'queued', interval '1 hour', 'test') RETURNING id",
            (project,),
        )
    )


_RUN = (
    'INSERT INTO near_dup_runs (project_id, level, mode, status, params, space, algorithm_version, '
    'profile_config_id, profile_revision, finished_at) '
    "VALUES (%s, 'image', 'full', 'current', '{}', 'pe@abc:pe_embedding', 'nd-1', %s, %s, now()) "
    'RETURNING id'
)


def seed_pg_rows(pg: Any, records: Mapping[str, Any], root: Path) -> dict[str, dict[str, str]]:
    """Per slug: ``job_id``, ``run_id`` and ``group_id`` (the group's lineage, which is what the
    wire calls its id) of rows that belong to that project, and the rows of ``seed_world``."""
    out: dict[str, dict[str, str]] = {}
    with psycopg.connect(pg.conninfo('super'), autocommit=True) as conn:
        for slug, record in records.items():
            project = record.id
            profile = _profile(conn, project)
            run = _one(conn.execute(_RUN, (project, *profile)))
            lineage = uuid.uuid5(uuid.NAMESPACE_DNS, f'{slug}-near-dup-group')
            group = _one(
                conn.execute(
                    'INSERT INTO near_dup_groups (run_id, purpose, lineage_id, representative_id, '
                    "member_count, min_edge_cosine, min_cosine_to_rep) VALUES (%s, 'dedup', %s, %s, "
                    '2, 1, 1) RETURNING id',
                    (run, lineage, f'{slug}-item-0001'),
                )
            )
            for n, rep in ((1, True), (2, False)):
                conn.execute(
                    'INSERT INTO near_dup_members (run_id, purpose, member_id, group_id, '
                    "is_representative, cosine_to_rep) VALUES (%s, 'dedup', %s, %s, %s, 1)",
                    (run, f'{slug}-item-000{n}', group, rep),
                )
            out[slug] = {
                'job_id': str(_job(conn, project)),
                'run_id': str(run),
                'group_id': str(lineage),
                **seed_world(conn, slug, project, str(root / 'images' / slug)),
            }
    return out
