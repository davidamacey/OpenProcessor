"""Docker interactions of the control-plane live checks: one small module, every call bounded.

Every call goes through ``docker compose -p <project> -f docker/test/compose.yml`` with the
project name of the harness (``VERIFY_COMPOSE_PROJECT``), so a check can only touch the
disposable stack. Docs: https://docs.docker.com/reference/cli/docker/compose/
"""

from __future__ import annotations

import json
import subprocess  # nosec B404 - drives the harness's own compose project
from typing import Any

from tests.live.conftest import COMPOSE_FILE, COMPOSE_PROJECT, REPO_ROOT


DEFAULT_TIMEOUT_S = 120
PG_USER = 'op_super'
PG_DB = 'openprocessor'


def compose(
    *args: str,
    stdin: str | None = None,
    check: bool = True,
    timeout: int = DEFAULT_TIMEOUT_S,
) -> subprocess.CompletedProcess[str]:
    """``docker compose -p NAME -f docker/test/compose.yml ARGS``; raises on a non-zero exit."""
    return subprocess.run(  # nosec B603 - fixed argv, no shell
        ['docker', 'compose', '-p', COMPOSE_PROJECT, '-f', str(COMPOSE_FILE), *args],
        input=stdin,
        check=check,
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(REPO_ROOT),
    )


def service_states() -> dict[str, dict[str, Any]]:
    """``docker compose ps -a`` as ``{service: {State, ExitCode, ...}}`` (one row per service)."""
    out = compose('ps', '-a', '--format', 'json').stdout
    rows = [json.loads(line) for line in out.splitlines() if line.strip()]
    return {row['Service']: row for row in rows}


def kill(service: str) -> None:
    """``docker kill`` every container of the service (SIGKILL: no soft shutdown, no lease release)."""
    compose('kill', service)


def start(service: str) -> None:
    """Start a stopped or killed service container again (same container, same filesystem)."""
    compose('start', service)


def stop(service: str, *, grace_s: int = 10) -> None:
    compose('stop', '-t', str(grace_s), service)


def remove_with_volumes(service: str) -> None:
    """Stop and delete the container and its anonymous volumes (a broker that comes back empty)."""
    compose('rm', '--stop', '--force', '--volumes', service)


def up(service: str) -> None:
    """Create and start one service without touching its dependencies."""
    compose('up', '-d', '--no-deps', '--wait', service, timeout=180)


def psql(sql: str) -> str:
    """Run ``sql`` in the postgres container as the superuser over the local socket; tuples-only."""
    return compose(
        'exec', '-T', 'postgres', 'psql', '-U', PG_USER, '-d', PG_DB, '-At', '-c', sql
    ).stdout.strip()
