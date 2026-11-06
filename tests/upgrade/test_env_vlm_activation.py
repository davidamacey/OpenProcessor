"""A project that activated the 0.6.0 ``env`` built-in loads pointing at a stored endpoint document."""

from __future__ import annotations

import datetime as dt

import pytest
from sqlalchemy import text

from src.services.labeling.vlm_endpoints import ENV_ENDPOINT_NAME, env_endpoint, probe_fingerprint
from src.services.upgrade.from_060 import env_endpoint as upgrade_env
from src.services.upgrade.from_060.model import (
    ENV_NAME,
    ENV_REVISION,
    ProbeSrc,
    UnresolvedBuiltinError,
)


NOW = dt.datetime(2026, 2, 1, tzinfo=dt.UTC)


def rows(pg, sql: str):
    with pg.engine('super').connect() as conn:
        return conn.execute(text(sql)).all()


def test_the_upgrade_and_the_api_name_the_env_endpoint_alike() -> None:
    assert ENV_NAME == ENV_ENDPOINT_NAME


@pytest.mark.db
@pytest.mark.asyncio
async def test_the_demo_project_points_at_the_stored_env_endpoint(make_runner, clean_db) -> None:
    await make_runner().run()
    assert rows(
        clean_db,
        'SELECT d.name, d.project_id IS NULL, a.revision, d.description, r.body ->> $$base_url$$, '
        'r.body ->> $$model$$ FROM config_activations a JOIN projects p ON p.id = a.project_id '
        'JOIN config_documents d ON d.id = a.config_id '
        'JOIN config_revisions r ON r.config_id = d.id AND r.revision = a.revision '
        "WHERE p.slug = 'demo' AND a.axis = 'vlm'",
    ) == [
        (
            'env',
            True,
            ENV_REVISION,
            'From OP_VLM_URL / OP_VLM_MODEL',
            'http://vlm-env:8000/v1',
            'env-model',
        )
    ]
    # the move away from the stored endpoint is kept in the history
    assert rows(
        clean_db,
        'SELECT f.name, e.from_revision, t.name, e.to_revision FROM config_activation_events e '
        'JOIN projects p ON p.id = e.project_id JOIN config_documents t ON t.id = e.to_config_id '
        'JOIN config_documents f ON f.id = e.from_config_id '
        "WHERE p.slug = 'demo' AND e.axis = 'vlm' ORDER BY e.id DESC LIMIT 1",
    ) == [('local-vlm', 1, 'env', ENV_REVISION)]
    # the project that never moved still points at the stored endpoint
    assert rows(
        clean_db,
        'SELECT d.name FROM config_activations a JOIN projects p ON p.id = a.project_id '
        "JOIN config_documents d ON d.id = a.config_id WHERE p.slug = 'default' AND a.axis = 'vlm'",
    ) == [('local-vlm',)]


@pytest.mark.db
@pytest.mark.asyncio
async def test_the_external_acknowledgements_load_as_project_state(make_runner, clean_db) -> None:
    await make_runner().run()
    assert rows(
        clean_db,
        'SELECT p.slug, s.revision, s.value FROM project_state s JOIN projects p ON p.id = s.project_id '
        "WHERE s.key = 'vlm_external_ack'",
    ) == [
        (
            'demo',
            1,
            {
                'acked_refs': {
                    'env@0123456789ab': '2026-01-03T00:00:00+00:00',
                    'local-vlm@1': '2026-01-02T00:00:00+00:00',
                }
            },
        )
    ]


@pytest.mark.db
@pytest.mark.asyncio
async def test_without_a_vlm_in_the_environment_the_upgrade_stops_and_loads_nothing(
    make_runner, clean_db, monkeypatch
) -> None:
    monkeypatch.setenv('OP_VLM_URL', '')
    with pytest.raises(UnresolvedBuiltinError, match='OP_VLM_URL'):
        await make_runner().run()
    assert rows(clean_db, 'SELECT count(*) FROM projects') == [(0,)]


def test_a_probe_of_the_current_environment_body_is_kept_on_the_stored_revision() -> None:
    body = env_endpoint().body
    record = {'ok': True, 'probed_at': '2026-01-03T00:00:00+00:00'}
    kept = ProbeSrc(
        ENV_NAME,
        ENV_REVISION,
        NOW,
        True,
        {'fingerprint': probe_fingerprint(body), 'record': record},
    )
    stale = ProbeSrc(ENV_NAME, ENV_REVISION, NOW, True, {'fingerprint': 'x' * 16, 'record': record})
    assert upgrade_env.env_endpoint_source(kept, now=NOW).probes == (kept,)
    assert upgrade_env.env_endpoint_source(stale, now=NOW).probes == ()
