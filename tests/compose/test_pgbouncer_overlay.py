"""D-50: the PgBouncer overlay puts the pooler under every pooled DB client and keeps Postgres direct.

Docs: https://www.pgbouncer.org/features.html (transaction mode),
https://docs.docker.com/reference/cli/docker/compose/config/ (merged render).
"""

from __future__ import annotations

import pytest

from compose._helpers import render


OVERLAY = ('docker-compose.yml', 'docker-compose.pgbouncer.yml')
POOLED = (
    'api',
    'jobs-worker',
    'periodic-worker',
    'open-vocab-worker',
    'cluster-worker',
    'curation-auto-label-worker',
    'curation-assist-loop-worker',
    'curation-trainer',
    'model-export-worker',
    'curation-evaluator',
    'jobs-beat',
)


@pytest.mark.parametrize('name', POOLED)
def test_the_pooled_host_is_the_pooler_and_the_direct_host_is_postgres(name: str) -> None:
    env = render({}, *OVERLAY)[name]['environment']
    assert env['OP_DB_HOST'] == 'pgbouncer'
    assert env['OP_DB_DIRECT_HOST'] == 'postgres'
    assert env['OP_DB_POOLER'] == 'pgbouncer'


def test_the_migration_step_stays_on_postgres() -> None:
    env = render({}, *OVERLAY)['db-migrate']['environment']
    assert env['OP_DB_HOST'] == 'postgres'
    assert 'OP_DB_POOLER' not in env


def test_the_pooler_runs_in_transaction_mode_without_a_password_value() -> None:
    pooler = render({}, *OVERLAY)['pgbouncer']
    assert pooler['environment']['POOL_MODE'] == 'transaction'
    assert not any('PASSWORD' in key for key in pooler['environment'])
    assert not pooler.get('ports')
    assert 'no-new-privileges:true' in pooler['security_opt']


def test_the_default_render_has_no_pooler() -> None:
    services = render({})
    assert 'pgbouncer' not in services
    assert services['api']['environment'].get('OP_DB_POOLER') is None
