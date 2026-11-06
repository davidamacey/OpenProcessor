"""The Postgres + Valkey + Celery services of the B2 control plane (offline: raw compose YAML).

Docs: https://docs.docker.com/reference/compose-file/services/ (depends_on, deploy, healthcheck),
https://docs.docker.com/compose/how-tos/use-secrets/ (secrets as read-only files).
"""

from __future__ import annotations

import re
from typing import Any

import pytest

from compose._helpers import _load_compose, _services


NEW = (
    'postgres',
    'valkey',
    'db-migrate',
    'jobs-worker',
    'periodic-worker',
    'open-vocab-worker',
    'jobs-beat',
    'flower',
)
CELERY_WORKERS = ('jobs-worker', 'periodic-worker', 'open-vocab-worker')
# Python services that run the application image (db-migrate is the one that migrates).
PYTHON_SERVICES = (
    'api',
    'jobs-worker',
    'periodic-worker',
    'open-vocab-worker',
    'jobs-beat',
    'flower',
    'curation-detection-worker',
    'curation-vlm-worker',
    'curation-auto-label-worker',
    'curation-assist-loop-worker',
    'curation-cluster-refresh',
)
# Celery's worker_soft_shutdown_timeout (02b section 7.1); the grace period must outlast it.
SOFT_SHUTDOWN_S = 30
MIN_GRACE_S = SOFT_SHUTDOWN_S + 30


def _svc(name: str) -> dict[str, Any]:
    return _services()[name]


def _command(name: str) -> str:
    cmd = _svc(name).get('command') or []
    return cmd if isinstance(cmd, str) else ' '.join(str(c) for c in cmd)


def _env(name: str) -> dict[str, str]:
    env = _svc(name).get('environment') or {}
    if isinstance(env, dict):
        return {k: str(v) for k, v in env.items()}
    return {e.split('=', 1)[0]: e.split('=', 1)[1] if '=' in e else '' for e in env}


def _depends(name: str) -> dict[str, str]:
    dep = _svc(name).get('depends_on') or {}
    if isinstance(dep, list):
        return dict.fromkeys(dep, 'service_started')
    return {k: v.get('condition', 'service_started') for k, v in dep.items()}


def _mem_limit(name: str) -> str:
    return str(_svc(name)['deploy']['resources']['limits']['memory'])


@pytest.mark.parametrize('name', NEW)
def test_new_services_exist_and_have_no_container_name(name: str) -> None:
    assert name in _services()
    assert 'container_name' not in _svc(name), 'D-54: new services are scalable, no fixed name'


def test_postgres_volume_connections_and_tcp_healthcheck() -> None:
    pg = _svc('postgres')
    assert 'openprocessor-pg:/var/lib/postgresql' in pg['volumes']
    assert re.search(r'max_connections=\$\{OP_PG_MAX_CONNECTIONS:-400\}', _command('postgres'))
    assert 'password_encryption=scram-sha-256' in _command('postgres')
    probe = ' '.join(pg['healthcheck']['test'])
    assert 'pg_isready' in probe
    assert '-h 127.0.0.1' in probe, 'the socket-only first-init server must not count as ready'
    assert pg['image'].startswith('postgres:18.6-trixie@sha256:')
    assert 'ports' not in pg


def test_valkey_is_durable_bounded_and_unpublished() -> None:
    vk = _svc('valkey')
    cmd = _command('valkey')
    assert '--appendonly yes' in cmd
    assert '--maxmemory-policy noeviction' in cmd
    assert '--maxmemory ${OP_VALKEY_MAXMEMORY:-1gb}' in cmd
    assert 'ports' not in vk
    assert 'openprocessor-queue:/data' in vk['volumes']
    assert vk['image'].startswith('valkey/valkey:9.1.3-alpine3.24@sha256:')
    probe = ' '.join(vk['healthcheck']['test'])
    assert 'REDISCLI_AUTH' in probe, 'the password must not be in argv'


def test_only_flower_publishes_a_port_and_on_loopback_by_default() -> None:
    for name in NEW:
        ports = _svc(name).get('ports')
        if name == 'flower':
            assert ports
            assert all(
                str(p).startswith('${OP_UI_BIND_ADDRESS:-${OP_BIND_ADDRESS:-127.0.0.1}}:')
                for p in ports
            )
            assert _svc(name)['profiles'] == ['monitoring']
        else:
            assert not ports, name


@pytest.mark.parametrize('name', CELERY_WORKERS)
def test_celery_workers_remap_sigterm_and_have_a_grace_period(name: str) -> None:
    assert _env(name)['REMAP_SIGTERM'] == 'SIGQUIT'
    grace = str(_svc(name)['stop_grace_period'])
    match = re.fullmatch(r'(?:\$\{OP_STOP_GRACE_JOBS:-(\d+)\}|(\d+))s', grace)
    assert match, grace
    assert int(match.group(1) or match.group(2)) >= MIN_GRACE_S


def test_jobs_worker_recycles_children_on_memory_and_count() -> None:
    cmd = _command('jobs-worker')
    assert '-Q jobs' in cmd
    assert '--max-memory-per-child=${OP_JOBS_MAX_MEMORY_PER_CHILD_KB:-8000000}' in cmd
    assert '--max-tasks-per-child=50' in cmd
    assert '--concurrency=${OP_JOBS_CONCURRENCY:-4}' in cmd


def test_no_worker_shares_the_api_prometheus_multiproc_dir() -> None:
    api_dir = _env('api')['PROMETHEUS_MULTIPROC_DIR']
    for name in (*CELERY_WORKERS, 'jobs-beat'):
        theirs = _env(name).get('PROMETHEUS_MULTIPROC_DIR')
        assert theirs != api_dir, name
    # a worker that sets one has its own tmpfs for it
    for name in CELERY_WORKERS:
        d = _env(name).get('PROMETHEUS_MULTIPROC_DIR')
        if d:
            assert any(str(t).startswith(d) for t in _svc(name).get('tmpfs', [])), name


@pytest.mark.parametrize('name', PYTHON_SERVICES)
def test_python_services_wait_for_migration_and_the_stores(name: str) -> None:
    dep = _depends(name)
    assert dep.get('db-migrate') == 'service_completed_successfully'
    assert dep.get('postgres') == 'service_healthy'
    assert dep.get('valkey') == 'service_healthy'
    assert dep.get('opensearch') == 'service_healthy'
    assert dep.get('triton-server') == 'service_healthy'


def test_db_migrate_waits_on_postgres_and_opensearch_and_runs_once() -> None:
    svc = _svc('db-migrate')
    dep = _depends('db-migrate')
    assert dep == {'postgres': 'service_healthy', 'opensearch': 'service_healthy'}
    assert svc['restart'] == 'no'
    assert _command('db-migrate') == 'python -m scripts.db.migrate upgrade'


def test_api_replicas_come_from_one_env_knob() -> None:
    assert _svc('api')['deploy']['replicas'] == '${OP_API_REPLICAS:-1}'


@pytest.mark.parametrize(
    ('name', 'var'),
    [
        ('postgres', 'OP_PG_MEM_LIMIT'),
        ('valkey', None),
        ('jobs-worker', 'OP_JOBS_MEM_LIMIT'),
        ('periodic-worker', 'OP_PERIODIC_MEM_LIMIT'),
        ('jobs-beat', 'OP_BEAT_MEM_LIMIT'),
    ],
)
def test_memory_limits_come_from_env(name: str, var: str | None) -> None:
    limit = _mem_limit(name)
    assert '${' in limit
    if var:
        assert var in limit


def test_secrets_are_mounted_read_only_as_directories() -> None:
    for name, mount in (
        ('postgres', './secrets/postgres/super_password:/run/secrets/op_pg/super_password:ro'),
        ('valkey', './secrets/valkey:/run/secrets/op_valkey:ro'),
        ('flower', './secrets/flower:/run/secrets/op_flower:ro'),
        ('db-migrate', './secrets/postgres:/run/secrets/op_pg:ro'),
    ):
        assert mount in _svc(name)['volumes'], name


def test_named_volumes_added_and_jobs_volume_kept_until_stage_6() -> None:
    vols = _load_compose()['volumes']
    assert 'openprocessor-pg' in vols
    assert 'openprocessor-queue' in vols


def test_the_unhealthy_comment_is_in_the_compose_file() -> None:
    from compose._helpers import COMPOSE_PATH

    text = COMPOSE_PATH.read_text()
    assert 'nothing acts on `unhealthy`; `restart:` fires only when the main process exits' in text


def test_valkey_reads_its_conf_from_a_tmpfs_copy_the_dropped_user_can_read() -> None:
    """Spike S0: the image drops to `valkey`, which cannot read a mode-600 file of the host user."""
    vk = _svc('valkey')
    entry = ' '.join(vk['entrypoint'])
    assert 'cat /run/secrets/op_valkey/valkey.conf > /run/valkey/valkey.conf' in entry
    assert 'chown valkey:valkey /run/valkey/valkey.conf' in entry
    assert vk['command'][1] == '/run/valkey/valkey.conf'
    assert any(str(t).startswith('/run/valkey') for t in vk['tmpfs'])


def _overlay(path: str) -> dict[str, Any]:
    from compose._helpers import REPO_ROOT, _load_yaml

    return _load_yaml(REPO_ROOT / path)


DEV_PYTHON = (
    'db-migrate',
    'jobs-worker',
    'periodic-worker',
    'open-vocab-worker',
    'jobs-beat',
    'flower',
)


@pytest.mark.parametrize('name', DEV_PYTHON)
def test_dev_overlay_builds_and_mounts_source_for_the_new_python_services(name: str) -> None:
    svc = _overlay('docker-compose.dev.yml')['services'][name]
    assert svc['build']['target'] == 'api'
    assert './src:/app/src:ro' in svc['volumes']
    assert './scripts:/app/scripts:ro' in svc['volumes']


def test_dev_overlay_says_celery_does_not_hot_reload() -> None:
    from compose._helpers import REPO_ROOT

    text = (REPO_ROOT / 'docker-compose.dev.yml').read_text()
    assert 'Celery does not hot-reload' in text
    assert 'restart jobs-worker' in text


HARNESS_NEW = ('postgres', 'valkey', 'db-migrate', 'jobs-worker', 'periodic-worker', 'jobs-beat')


@pytest.mark.parametrize('name', HARNESS_NEW)
def test_live_harness_runs_the_control_plane_without_fixed_names_or_host_ports(name: str) -> None:
    harness = _overlay('docker/test/compose.yml')
    svc = harness['services'][name]
    assert 'container_name' not in svc, 'the harness scales jobs-worker with --scale'
    assert 'ports' not in svc
    assert harness['name'] == 'op-live-verify'


def test_live_harness_api_waits_for_the_migration_and_never_mounts_the_docker_socket() -> None:
    from compose._helpers import REPO_ROOT

    harness = _overlay('docker/test/compose.yml')
    assert harness['services']['api']['depends_on']['db-migrate']['condition'] == (
        'service_completed_successfully'
    )
    assert 'docker.sock' not in (REPO_ROOT / 'docker/test/compose.yml').read_text().replace(
        '/var/run/docker.sock is NEVER mounted', ''
    )
