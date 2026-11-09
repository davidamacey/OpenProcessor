"""Rule h: a system pause is lifted once its cause is gone; a person's pause never is (3A-14).

``project_paused`` resumes when ``projects.paused_at`` is cleared, ``vlm_unavailable`` when the
VLM is free and an endpoint answers. ``user``, ``audit`` and ``budget`` wait for a person.
"""

from __future__ import annotations

import pytest
from fixtures.job_rows import insert_row, make_project, sql
from fixtures.reaper_support import Broker, environment, reaper, row, status
from psycopg.types.json import Jsonb

from src.services.jobs import blockers


pytestmark = pytest.mark.db


@pytest.fixture
def broker() -> Broker:
    return Broker()


@pytest.fixture
def env(pg, pg_env, monkeypatch):
    yield from environment(pg, monkeypatch)


def pause_project(env, project) -> None:
    sql(env, "UPDATE projects SET paused_at = now(), paused_by = 'test' WHERE id = %s", project)


def test_a_project_paused_job_stays_while_the_project_is_paused_then_resumes(env, broker) -> None:
    project = make_project(env)
    job = insert_row(env, project, 'paused', pause_reason='project_paused')
    pause_project(env, project)
    reaper(broker).tick()
    assert status(env, job) == 'paused'
    assert broker.sent == []

    sql(env, 'UPDATE projects SET paused_at = NULL, paused_by = NULL WHERE id = %s', project)
    report = reaper(broker).tick()
    assert report.system_resumed == 1
    assert row(env, job)[0] == 'queued'
    assert row(env, job)[6] is None  # pause_reason cleared
    assert [j for _k, j in broker.sent] == [job]


@pytest.mark.parametrize('reason', ['user', 'audit', 'budget'])
def test_a_pause_that_waits_for_a_person_is_never_resumed(env, broker, reason) -> None:
    project = make_project(env)
    job = insert_row(env, project, 'paused', pause_reason=reason)
    reaper(broker).tick()
    assert (status(env, job), row(env, job)[6]) == ('paused', reason)
    assert broker.sent == []


def test_a_vlm_unavailable_job_resumes_when_the_vlm_answers(env, broker) -> None:
    project = make_project(env)
    job = insert_row(env, project, 'paused', pause_reason='vlm_unavailable')
    answers = {'free': False}
    reaper(broker, vlm_available=lambda _s: answers['free']).tick()
    assert status(env, job) == 'paused'
    answers['free'] = True
    reaper(broker, vlm_available=lambda _s: answers['free']).tick()
    assert status(env, job) == 'queued'


def test_the_vlm_is_asked_once_per_tick(env, broker) -> None:
    project = make_project(env)
    for _ in range(3):
        insert_row(env, project, 'paused', pause_reason='vlm_unavailable')
    asked: list[int] = []

    def probe(_session) -> bool:
        asked.append(1)
        return True

    report = reaper(broker, vlm_available=probe).tick()
    assert report.system_resumed == 3
    assert asked == [1]


def test_the_default_vlm_check_needs_the_setting_cleared_and_an_ok_probe(env) -> None:
    from sqlalchemy.orm import Session

    from src.db.engine import get_sync_engine
    from src.db.session import sync_deployment_tx

    def available() -> bool:
        with Session(get_sync_engine()) as s, sync_deployment_tx(s, 'test'):
            return blockers.vlm_available(s)

    sql(env, 'DELETE FROM deployment_settings')
    assert available() is False  # no endpoint has been probed ok
    project = make_project(env)
    [(config_id,)] = sql(
        env,
        'INSERT INTO config_documents (project_id, kind, name, current_revision, updated_by) '
        "VALUES (NULL, 'vlm_endpoint', 'e1', 1, 'test') RETURNING id",
    )
    sql(
        env,
        'INSERT INTO config_revisions (config_id, revision, body, body_sha256, created_by) '
        "VALUES (%s, 1, '{}', 'x', 'test')",
        config_id,
    )
    sql(
        env,
        "INSERT INTO vlm_endpoint_probes (config_id, revision, ok, result) VALUES (%s, 1, true, '{}')",
        config_id,
    )
    assert available() is True
    sql(
        env,
        'INSERT INTO deployment_settings (key, value, updated_by) VALUES (%s, %s, %s)',
        'vlm_paused_for_training',
        Jsonb({'arbiter': {'claims': ['a-running-job']}}),
        'test',
    )
    assert available() is False  # taken for training
    assert project  # the helper project keeps the module's fixtures honest
