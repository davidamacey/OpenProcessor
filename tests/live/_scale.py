"""The scale-out overlay of the live harness: the replica URLs and the reconfigure call.

``docker/test/compose.scale.yml`` puts an nginx in front of N ``api`` replicas and publishes one
port per replica (replica 1 and 2), so a check can move a client between two named replicas.
Start it as that file's header says; ``reconfigure`` re-creates the api and jobs-worker services
with another worker and replica count.
"""

from __future__ import annotations

import os
import subprocess  # nosec B404 - drives the harness's own compose project
from typing import Any

import httpx

from tests.live.conftest import COMPOSE_FILE, COMPOSE_FILE_ARGS, COMPOSE_PROJECT, REPO_ROOT


SCALE_FILE = COMPOSE_FILE.parent / 'compose.scale.yml'
REPLICA1_URL = os.environ.get('VERIFY_REPLICA1_URL', 'http://localhost:14725')
REPLICA2_URL = os.environ.get('VERIFY_REPLICA2_URL', 'http://localhost:14726')


def require_replicas() -> None:
    """Fail (not skip) with the start command when the overlay is not up."""
    for url in (REPLICA1_URL, REPLICA2_URL):
        try:
            httpx.get(f'{url}/live', timeout=10.0).raise_for_status()
        except httpx.HTTPError as exc:
            raise AssertionError(
                f'replica {url} not reachable ({exc}); start the stack with '
                f'`docker compose -p {COMPOSE_PROJECT} -f docker/test/compose.yml '
                '-f docker/test/compose.scale.yml up -d --wait` and OP_API_REPLICAS=2'
            ) from exc


def reconfigure(*, workers: int, api_replicas: int, jobs_replicas: int) -> None:
    """Re-create ``api`` and ``jobs-worker`` with the given counts and wait until healthy."""
    env = {
        **os.environ,
        'VERIFY_COMPOSE_PROJECT': COMPOSE_PROJECT,
        'API_WORKERS': str(workers),
        'OP_API_REPLICAS': str(api_replicas),
        'OP_JOBS_REPLICAS': str(jobs_replicas),
    }
    base = ['docker', 'compose', '-p', COMPOSE_PROJECT, *COMPOSE_FILE_ARGS]
    subprocess.run(  # nosec B603 - fixed argv, no shell
        [*base, '-f', str(SCALE_FILE), 'up', '-d', '--wait', '--no-build', '--no-deps',
         '--force-recreate', 'api', 'jobs-worker', 'proxy'],
        check=True, capture_output=True, text=True, timeout=400, cwd=str(REPO_ROOT), env=env,
    )  # fmt: skip


def restore_base_stack() -> None:
    """Undo ``reconfigure`` when it ran: drop the proxy and put ``api`` / ``jobs-worker`` back to one container.

    The proxy publishes ``VERIFY_API_PORT``, so a later plain ``up --force-recreate api`` (the API
    restart check in test_live_vlm.py) cannot bind it while the overlay's proxy is still there.
    """
    env = {**os.environ, 'VERIFY_COMPOSE_PROJECT': COMPOSE_PROJECT}
    base = ['docker', 'compose', '-p', COMPOSE_PROJECT, *COMPOSE_FILE_ARGS]
    proxy = subprocess.run(  # nosec B603 - fixed argv, no shell
        [*base, '-f', str(SCALE_FILE), 'ps', '-q', 'proxy'],
        check=True, capture_output=True, text=True, timeout=60, cwd=str(REPO_ROOT), env=env,
    )  # fmt: skip
    if not proxy.stdout.strip():
        return
    for argv in (
        [*base, '-f', str(SCALE_FILE), 'rm', '-sf', 'proxy'],
        [*base, 'up', '-d', '--wait', '--no-build', '--no-deps', '--force-recreate',
         '--scale', 'api=1', '--scale', 'jobs-worker=1', 'api', 'jobs-worker'],
    ):  # fmt: skip
        subprocess.run(  # nosec B603 - fixed argv, no shell
            argv,
            check=True,
            capture_output=True,
            text=True,
            timeout=400,
            cwd=str(REPO_ROOT),
            env=env,
        )


def exec_json(code: str, service: str = 'api', index: int = 1) -> Any:
    """Run ``code`` in replica ``index`` of ``service``; return the last stdout line parsed as JSON."""
    import json

    proc = subprocess.run(  # nosec B603 - fixed argv, no shell
        ['docker', 'compose', '-p', COMPOSE_PROJECT, *COMPOSE_FILE_ARGS, 'exec', '-T',
         '--index', str(index), service, 'python', '-c', code],
        check=True, capture_output=True, text=True, timeout=120, cwd=str(REPO_ROOT),
    )  # fmt: skip
    return json.loads(proc.stdout.strip().splitlines()[-1])
