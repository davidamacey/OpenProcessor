"""A Celery worker's health check pings the node name the worker runs under.

``celery worker --hostname gpu_export@%h`` answers ``inspect ping -d gpu_export@<host>`` only;
a check that pings ``celery@<host>`` waits out its timeout and marks a healthy worker unhealthy.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[2]
FILES = ['docker-compose.yml', 'docker/test/compose.yml']


def _text(value: object) -> str:
    return ' '.join(value) if isinstance(value, list) else str(value or '')


def _pinging_workers() -> list[tuple[str, str, str, str]]:
    rows = []
    for rel in FILES:
        services = yaml.safe_load((ROOT / rel).read_text(encoding='utf-8'))['services']
        for name, service in services.items():
            check = _text((service.get('healthcheck') or {}).get('test'))
            target = re.search(r'inspect ping .*-d \\?"?([\w.-]+)@', check)
            if target:
                rows.append((rel, name, _text(service.get('command')), target.group(1)))
    return rows


@pytest.mark.parametrize(('rel', 'name', 'command', 'pinged'), _pinging_workers())
def test_the_ping_names_the_worker_node(rel: str, name: str, command: str, pinged: str) -> None:
    node = re.search(r'--hostname[ =]([\w.-]+)@', command)
    expected = node.group(1) if node else 'celery'  # Celery's default node name
    assert pinged == expected, f'{rel} {name}: pings {pinged}@, the worker runs as {expected}@'


def test_the_known_pinging_workers_are_covered() -> None:
    names = {(rel, name) for rel, name, *_ in _pinging_workers()}
    assert ('docker-compose.yml', 'model-export-worker') in names
    assert ('docker/test/compose.yml', 'trainer') in names
