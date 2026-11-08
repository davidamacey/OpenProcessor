"""The hourly VLM usage table accepts the open-vocabulary gate as a source."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

import psycopg
import pytest
from _research_seed import make_project
from psycopg import errors


if TYPE_CHECKING:
    from fixtures.postgres import PgDatabase

pytestmark = pytest.mark.db

_INSERT = (
    'INSERT INTO vlm_usage_hourly (project_id, endpoint_config_id, source, hour) '
    "VALUES (%s, %s, %s, '2026-11-05 10:00:00+00')"
)


@pytest.fixture
def conn(pg: PgDatabase):
    with psycopg.connect(pg.conninfo('super'), autocommit=True) as c:
        yield c


def _endpoint(conn) -> object:
    return conn.execute(
        'INSERT INTO config_documents (project_id, kind, name, current_revision, updated_by) '
        "VALUES (NULL, 'vlm_endpoint', %s, 1, 'test') RETURNING id",
        (f'local-{uuid.uuid4().hex[:8]}',),
    ).fetchone()[0]


def test_the_open_vocab_gate_is_a_usage_source(conn) -> None:
    project, endpoint = make_project(conn, 'usage-gate'), _endpoint(conn)
    conn.execute(_INSERT, (project, endpoint, 'open_vocab_gate'))
    assert conn.execute(
        "SELECT count(*) FROM vlm_usage_hourly WHERE source = 'open_vocab_gate'"
    ).fetchone() == (1,)


def test_an_unknown_source_is_still_refused(conn) -> None:
    project, endpoint = make_project(conn, 'usage-other'), _endpoint(conn)
    with pytest.raises(errors.CheckViolation, match='ck_vlm_usage_hourly_source'):
        conn.execute(_INSERT, (project, endpoint, 'made_up'))
