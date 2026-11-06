"""The Celery pool healthcheck gets the node its worker names itself.

``scripts/healthcheck/celery_pool_healthcheck.py <node-or-prefix>`` exits 2 without an
argument, so a bare call leaves a healthy worker permanently unhealthy. The prefix must equal
the ``--hostname <prefix>@%h`` the same service's command passes to ``celery worker``.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

import pytest
import yaml

from compose._helpers import REPO_ROOT


if TYPE_CHECKING:
    from pathlib import Path


SCRIPT = 'celery_pool_healthcheck.py'
COMPOSE_FILES = (
    REPO_ROOT / 'docker-compose.yml',
    REPO_ROOT / 'docker-compose.dev.yml',
    REPO_ROOT / 'docker' / 'test' / 'compose.yml',
)


def _words(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return value.split()
    return [str(v) for v in value]


def _probe_services(path: Path) -> dict[str, dict[str, Any]]:
    services = (yaml.safe_load(path.read_text()) or {}).get('services') or {}
    return {
        name: svc
        for name, svc in services.items()
        if any(SCRIPT in w for w in _words((svc.get('healthcheck') or {}).get('test')))
    }


def _hostname_prefix(svc: dict[str, Any]) -> str:
    match = re.search(r'--hostname[ =](\S+?)@', ' '.join(_words(svc.get('command'))))
    assert match, 'a worker with a pool healthcheck must name itself with --hostname <prefix>@%h'
    return match.group(1)


def test_the_four_celery_workers_run_the_pool_healthcheck() -> None:
    assert set(_probe_services(REPO_ROOT / 'docker-compose.yml')) == {
        'jobs-worker',
        'periodic-worker',
        'open-vocab-worker',
        'cluster-worker',
    }


@pytest.mark.parametrize('path', COMPOSE_FILES, ids=lambda p: p.name)
def test_every_pool_healthcheck_passes_its_workers_node(path: Path) -> None:
    for name, svc in _probe_services(path).items():
        test = _words(svc['healthcheck']['test'])
        argv = test[test.index(next(w for w in test if SCRIPT in w)) + 1 :]
        assert argv, f'{path.name}:{name} calls {SCRIPT} with no node argument'
        assert argv[0].split('@')[0] == _hostname_prefix(svc), f'{path.name}:{name}'
