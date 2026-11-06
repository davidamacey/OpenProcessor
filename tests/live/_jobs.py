"""Starting infra jobs in the live harness, and reading them back.

No route starts an ``op.infra.*`` job, so a check starts one through the job service from inside the
``api`` container (``docker compose exec -T api python -``): the same ``start_job`` the routes
call, over the container's own database and broker configuration. Reads use the deployment-wide
``GET /curation/jobs`` route (the infra kinds belong to no project).
"""

from __future__ import annotations

import json
import time
from typing import Any

from tests.live._compose import compose


_START = """
import asyncio, json
from src.db.session import async_session_factory
from src.services.jobs.service import start_job

async def main():
    async with async_session_factory()() as session:
        started = await start_job(session, {kind!r}, None, params={params!r})
    print(json.dumps({{'job_id': str(started.job.id)}}))

asyncio.run(main())
"""

# A start whose message never leaves: the commit happened, the send did not (a crash between the
# two). The row stays `queued` with no dispatch record, which is reaper rule c's input.
_START_UNDISPATCHED = """
import asyncio, json
from src.db.session import async_session_factory
from src.services.jobs import service

async def never(*args, **kwargs):
    return None

service._dispatch_after_commit = never

async def main():
    async with async_session_factory()() as session:
        started = await service.start_job(session, {kind!r}, None, params={params!r})
    print(json.dumps({{'job_id': str(started.job.id)}}))

asyncio.run(main())
"""


class StartFailedError(RuntimeError):
    """The in-container start raised (database or queue unavailable...); ``stderr`` says why."""

    def __init__(self, stderr: str) -> None:
        super().__init__(stderr[-2000:])
        self.stderr = stderr


def _run(script: str, kind: str, params: dict[str, Any]) -> str:
    proc = compose(
        'exec', '-T', 'api', 'python', '-', stdin=script.format(kind=kind, params=params),
        check=False, timeout=90,
    )  # fmt: skip
    if proc.returncode != 0:
        raise StartFailedError(proc.stderr)
    return json.loads(proc.stdout.strip().splitlines()[-1])['job_id']


def start_infra_job(kind: str, **params: Any) -> str:
    """Start ``kind`` (``noop`` or ``sleep_noop``) as a deployment-wide job; return its id."""
    return _run(_START, kind, params)


def start_infra_job_undispatched(kind: str, **params: Any) -> str:
    """Commit a queued job but do not send its message (reaper rule c must dispatch it)."""
    return _run(_START_UNDISPATCHED, kind, params)


def get_job(api: Any, job_id: str) -> dict[str, Any] | None:
    """The job from the deployment-wide list; ``None`` when it is not on the first page."""
    resp = api.get('/curation/jobs', params={'limit': 200})
    resp.raise_for_status()
    return next((j for j in resp.json()['items'] if j['id'] == job_id), None)


def wait_for_job(
    api: Any,
    job_id: str,
    statuses: set[str],
    *,
    timeout_s: float,
    interval_s: float = 1.0,
) -> dict[str, Any] | None:
    """Poll the job until its status is in ``statuses`` or ``timeout_s`` passes; the last view."""
    deadline = time.monotonic() + timeout_s
    job = get_job(api, job_id)
    while time.monotonic() < deadline:
        if job is not None and job['status'] in statuses:
            return job
        time.sleep(interval_s)
        job = get_job(api, job_id)
    return job
