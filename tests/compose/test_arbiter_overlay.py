"""D-84 / review R-20 / RC-3: the Docker socket is mounted into the one-process arbiter worker alone.

Docs: https://docs.docker.com/reference/cli/docker/compose/config/ (merged render),
https://docs.celeryq.dev/en/stable/reference/cli.html#celery-worker (``-Q``).
"""

from __future__ import annotations

import re

from compose._helpers import render
from compose.test_consumers_after_b2 import consumers
from src.config.queue import TaskQueues


OVERLAY = ('docker-compose.yml', 'docker-compose.gpu-arbiter.yml')
SOCKET = '/var/run/docker.sock'


def _socket_mounts(services: dict) -> set[str]:
    """Services with a writable socket mount (the monitoring log shipper reads it read-only)."""
    return {
        name
        for name, service in services.items()
        for volume in service.get('volumes') or []
        if SOCKET in (volume.get('source'), volume.get('target')) and not volume.get('read_only')
    }


def test_only_the_arbiter_worker_mounts_the_socket() -> None:
    assert _socket_mounts(render({}, *OVERLAY)) == {'gpu-arbiter-worker'}


def test_the_base_render_has_no_socket_and_no_arbiter_consumer() -> None:
    services = render({})
    assert not _socket_mounts(services)
    assert TaskQueues.ARBITER not in consumers(services)


def test_the_arbiter_worker_consumes_only_the_arbiter_queue() -> None:
    services = render({}, *OVERLAY)
    worker = services['gpu-arbiter-worker']
    command = ' '.join(worker['command'])
    assert re.search(r'-Q arbiter(\s|$)', command)
    assert '-c 1' in command
    assert consumers({'gpu-arbiter-worker': {'command': command}}) == {
        TaskQueues.ARBITER: {'gpu-arbiter-worker'}
    }
    for name, service in services.items():
        if name != 'gpu-arbiter-worker':
            assert 'arbiter' not in consumers({name: {'command': service.get('command') or ''}})


def test_the_arbiter_worker_is_hardened_and_has_no_published_port() -> None:
    worker = render({}, *OVERLAY)['gpu-arbiter-worker']
    assert 'no-new-privileges:true' in worker['security_opt']
    assert worker['cap_drop'] == ['ALL']
    assert not worker.get('ports')
    assert 'container_name' not in worker
