"""Arbiter socket group, pinned PgBouncer, pg_dump in the image, backup wiring on the periodic worker.

Docs: https://docs.docker.com/reference/compose-file/services/#group_add,
https://www.postgresql.org/docs/current/app-pgdump.html (a dump needs the server's major version).
"""

from __future__ import annotations

import re

from compose._helpers import REPO_ROOT, _load_yaml, render


def test_the_arbiter_worker_joins_the_socket_group() -> None:
    files = ('docker-compose.yml', 'docker-compose.gpu-arbiter.yml')
    assert render({'OP_DOCKER_GID': '131'}, *files)['gpu-arbiter-worker']['group_add'] == ['131']
    assert render({}, *files)['gpu-arbiter-worker']['group_add'] == ['999']


def test_no_other_service_gets_the_socket_group() -> None:
    files = ('docker-compose.yml', 'docker-compose.gpu-arbiter.yml')
    services = render({}, *files)
    assert [n for n, s in services.items() if s.get('group_add')] == ['gpu-arbiter-worker']


def test_pgbouncer_image_is_pinned_by_digest() -> None:
    image = _load_yaml(REPO_ROOT / 'docker-compose.pgbouncer.yml')['services']['pgbouncer']['image']
    assert re.search(r'@sha256:[0-9a-f]{64}\}$', image), image


def test_the_api_image_has_the_postgres_18_client() -> None:
    dockerfile = (REPO_ROOT / 'Dockerfile').read_text()
    base = dockerfile.split(' AS base', 1)[1].split('FROM ', 1)[0]
    assert 'postgresql-client-18' in base
    postgres = _load_yaml(REPO_ROOT / 'docker-compose.yml')['services']['postgres']['image']
    assert postgres.startswith('postgres:18')


def test_the_periodic_worker_can_run_the_backup_tick() -> None:
    service = render({})['periodic-worker']
    env = service['environment']
    assert env['OP_DB_BACKUP_DIR'] == '/backups/postgres'
    assert env['OP_DB_DIRECT_HOST'] == 'postgres'
    assert env['OP_DB_OWNER_PASSWORD_FILE'] == '/run/secrets/op_pg/owner_password'
    targets = {v['target'] for v in service['volumes']}
    assert {'/backups/postgres', '/run/secrets/op_pg/owner_password'} <= targets


def test_only_the_periodic_worker_gets_the_owner_secret() -> None:
    holders = [
        name
        for name, service in render({}).items()
        if any(v['target'].endswith('owner_password') for v in service.get('volumes') or [])
    ]
    assert 'periodic-worker' in holders
    assert 'api' not in holders
    assert 'jobs-worker' not in holders


def test_the_cluster_worker_has_the_app_role_and_queue_secrets() -> None:
    env = render({})['cluster-worker']['environment']
    assert env['OP_DB_APP_PASSWORD_FILE'] == '/run/secrets/op_pg/app_password'
    assert env['OP_QUEUE_PASSWORD_FILE'] == '/run/secrets/op_valkey/password'
