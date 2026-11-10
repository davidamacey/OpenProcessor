"""The VLM endpoint registry is deployment-wide, so a project-scoped job reads its probes through
a deployment-scope session of its own.

Row-level security lets a project-scope session see only its own project's rows, and a probe
belongs to a deployment-wide config document (``project_id`` NULL). Read in a worker's project
transaction, "an endpoint answered its latest probe" was always false, so every assist loop
paused as ``vlm_unavailable`` as soon as it started.
"""

from __future__ import annotations

import pytest
from fixtures.job_rows import make_project, project_session, sql

from src.services.jobs import blockers


pytestmark = [pytest.mark.db, pytest.mark.usefixtures('bound_async_engine')]


def _probed_endpoint(pg) -> None:
    sql(pg, 'DELETE FROM deployment_settings')
    [(config_id,)] = sql(
        pg,
        'INSERT INTO config_documents (project_id, kind, name, current_revision, updated_by) '
        "VALUES (NULL, 'vlm_endpoint', 'scoped', 1, 'test') RETURNING id",
    )
    sql(
        pg,
        'INSERT INTO config_revisions (config_id, revision, body, body_sha256, created_by) '
        "VALUES (%s, 1, '{}', 'x', 'test')",
        config_id,
    )
    sql(
        pg,
        "INSERT INTO vlm_endpoint_probes (config_id, revision, ok, result) VALUES (%s, 1, true, '{}')",
        config_id,
    )


@pytest.mark.asyncio
async def test_a_project_scoped_job_sees_the_deployment_wide_probe(pg) -> None:
    _probed_endpoint(pg)
    project = make_project(pg)

    with project_session(pg.engine(), project) as session:
        assert blockers.vlm_available(session) is False  # the row is hidden from this scope

    assert await blockers.vlm_ready() is True


@pytest.mark.asyncio
async def test_no_ok_probe_is_not_ready(pg) -> None:
    sql(pg, 'DELETE FROM deployment_settings')
    sql(pg, 'DELETE FROM vlm_endpoint_probes')
    assert await blockers.vlm_ready() is False
