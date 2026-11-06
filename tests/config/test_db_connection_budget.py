"""The PostgreSQL connection budget, computed from docker-compose.yml and the pool rules.

The table is 02b section 8: every service that reaches the database, how many processes it
runs (``deploy.replicas`` and the uvicorn / Celery counts in the compose file, with
``${OP_API_REPLICAS:-1}`` the single place the API is scaled) and how many connections each
process holds. The sum must fit ``OP_PG_MAX_CONNECTIONS`` minus the 13 slots kept for
maintenance (10) and ``superuser_reserved_connections`` (3); see PostgreSQL's
``max_connections`` documentation:
https://www.postgresql.org/docs/current/runtime-config-connection.html#GUC-MAX-CONNECTIONS
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from unittest import mock

import pytest
import yaml

from src.config.database import DatabaseSettings
from src.db.engine import CHILD_POOL_SIZE


COMPOSE = Path(__file__).resolve().parents[2] / 'docker-compose.yml'
RESERVED = 13

# Connections one Celery child holds: the sync engine plus the per-task async engine (at most 2).
CELERY_CHILD_CONNECTIONS = CHILD_POOL_SIZE + 2
# Children of the GPU worker images that 02b section 8 counts and the compose file does not
# carry yet (4 = 1 core child + 3 GPU children).
PLANNED_CELERY_CHILDREN = 4
# Sync pool of the non-Celery stream workers and daemons.
STREAM_WORKER_CONNECTIONS = 2
STREAM_WORKERS = (
    'curation-detection-worker',
    'curation-vlm-worker',
    'curation-auto-label-worker',
    'curation-assist-loop-worker',
    'curation-cluster-refresh',
)
BEAT_CONNECTIONS = 2  # one direct (single-beat lock) + a sync pool of 1
MIGRATE_CONNECTIONS = 2  # one direct, one for the OpenSearch step bookkeeping

_VAR = re.compile(r'\$\{([A-Z0-9_]+)(?::-([^}]*))?\}')


def _interpolate(text: str, env: dict[str, str]) -> str:
    return _VAR.sub(lambda m: env.get(m.group(1)) or m.group(2) or '', str(text))


def _services() -> dict[str, dict]:
    return yaml.safe_load(COMPOSE.read_text())['services']


def _db_services(services: dict[str, dict]) -> set[str]:
    def environment(service: dict) -> list[str]:
        env = service.get('environment', [])
        return [f'{k}={v}' for k, v in env.items()] if isinstance(env, dict) else env

    return {n for n, s in services.items() if any(e.startswith('OP_DB_') for e in environment(s))}


def _celery_children(command: str) -> int:
    match = re.search(r'(?:--concurrency[= ]|-c )(\d+)', command)
    return int(match.group(1)) if match else 0


def connection_budget(env: dict[str, str] | None = None) -> dict[str, int]:
    """Connections each part of the deployment may hold, and the limits, under ``env``."""
    env = dict(env or {})
    services = _services()
    api = services['api']
    workers = int(
        _interpolate(
            next(e for e in api['environment'] if e.startswith('WEB_CONCURRENCY=')).split('=')[1],
            env,
        )
    )
    replicas = int(_interpolate(api['deploy']['replicas'], env))
    with mock.patch.dict(os.environ, env):
        total = DatabaseSettings.from_env().api_connections_total
    processes = workers * replicas
    if processes > total:
        msg = f'{workers} workers x {replicas} replicas exceeds OP_DB_API_CONNECTIONS_TOTAL={total}'
        raise AssertionError(msg)
    per_process = max(1, total // processes)

    children = PLANNED_CELERY_CHILDREN
    for service in services.values():
        command = _interpolate(
            ' '.join(service['command'])
            if isinstance(service.get('command'), list)
            else service.get('command', ''),
            env,
        )
        if 'celery' in command and ' worker ' in f' {command} ':
            children += _celery_children(command)

    limit = int(_interpolate('${OP_PG_MAX_CONNECTIONS:-400}', env))
    parts = {
        'api_requests': processes * per_process,
        'api_polls': processes,
        'celery_children': children * CELERY_CHILD_CONNECTIONS,
        'beat': BEAT_CONNECTIONS,
        'stream_workers': len(STREAM_WORKERS) * STREAM_WORKER_CONNECTIONS,
        'db_migrate': MIGRATE_CONNECTIONS,
    }
    parts['total'] = sum(parts.values())
    parts['limit'] = limit - RESERVED
    return parts


def test_every_database_service_is_in_the_table() -> None:
    known = {'api', 'db-migrate', 'jobs-beat', *STREAM_WORKERS}
    unaccounted = []
    for name, service in _services().items():
        if name not in _db_services(_services()):
            continue
        command = service.get('command', '')
        command = ' '.join(command) if isinstance(command, list) else command
        if name not in known and not ('celery' in command and ' worker ' in f' {command} '):
            unaccounted.append(name)
    assert not unaccounted, f'add these services to the connection table: {unaccounted}'


def test_uvicorn_workers_equal_web_concurrency() -> None:
    """The pool share is computed from WEB_CONCURRENCY; uvicorn must start as many workers."""
    api = _services()['api']
    flag = next(a for a in api['command'] if a.startswith('--workers=')).split('=')[1].split()[0]
    web = next(e for e in api['environment'] if e.startswith('WEB_CONCURRENCY=')).split('=')[1]
    assert flag == web


def test_default_compose_fits() -> None:
    budget = connection_budget()
    assert budget['total'] <= budget['limit'], budget


def test_default_total_is_128() -> None:
    assert connection_budget()['api_requests'] == 128


def test_one_replica_gives_each_api_process_four_connections() -> None:
    budget = connection_budget()
    assert budget['api_requests'] // 32 == 4


def test_more_replicas_than_the_total_allows_is_refused() -> None:
    """32 workers x 8 replicas = 256 processes on a total of 128: the floor of 1 would break it."""
    with pytest.raises(AssertionError, match='exceeds OP_DB_API_CONNECTIONS_TOTAL'):
        connection_budget({'OP_API_REPLICAS': '8'})


def test_a_budget_over_the_server_limit_fails() -> None:
    """A raised total with more replicas, or a smaller max_connections, goes over the line."""
    over = connection_budget({'OP_API_REPLICAS': '4', 'OP_DB_API_CONNECTIONS_TOTAL': '512'})
    assert over['total'] > over['limit'], over
    small = connection_budget({'OP_PG_MAX_CONNECTIONS': '200'})
    assert small['total'] > small['limit'], small
