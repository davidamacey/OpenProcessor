"""Starting infra jobs in the live harness, and reading them back.

No route starts an ``op.infra.*`` job, so a check starts one through the job service from inside the
``api`` container (``docker compose exec -T api python -``): the same ``start_job`` the routes
call, over the container's own database and broker configuration. Reads use the deployment-wide
``GET /jobs`` route (src/routers/jobs.py; the infra kinds belong to no project).
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


# A job of a project kind: the same start_job the routes call, for the project with this slug.
_START_IN_PROJECT = """
import asyncio, json
from sqlalchemy import text
from src.db.session import async_session_factory
from src.services.jobs.service import start_job

async def main():
    async with async_session_factory()() as session:
        project = (await session.execute(
            text('SELECT id FROM projects WHERE slug = :slug'), {{'slug': {slug!r}}}
        )).scalar_one()
    async with async_session_factory()() as session:
        started = await start_job(session, {kind!r}, project, params={params!r})
    print(json.dumps({{'job_id': str(started.job.id)}}))

asyncio.run(main())
"""

# Sends one task by name to the queue its route names (the beat does this for a tick).
_SEND_TASK = """
from src.tasks.app import app
app.send_task({task!r}, task_id={task_id!r})
"""


# N small noop jobs started back to back, each tagged with the run so the check can find them.
_START_BURST = """
import asyncio, json, time
from src.db.session import async_session_factory
from src.services.jobs.service import start_job

async def main():
    began = time.monotonic()
    for i in range({count}):
        async with async_session_factory()() as session:
            await start_job(session, 'noop', None, params={{'load_run': {run!r}, 'i': i}})
    print(json.dumps({{'seconds': time.monotonic() - began}}))

asyncio.run(main())
"""


# The cancel the family routes call, for a job id (no generic cancel route exists).
_CANCEL = """
import asyncio, json, uuid
from src.db.session import async_session_factory
from src.services.jobs.service import cancel_job

async def main():
    async with async_session_factory()() as session:
        done = await cancel_job(
            session, uuid.UUID({job_id!r}),
            uuid.UUID({project!r}) if {project!r} else None, requested_by='live-test'
        )
    print(json.dumps({{'status': done.job.status, 'changed': done.changed}}))

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


def start_job_in_project(kind: str, *, slug: str = 'default', **params: Any) -> str:
    """Start ``kind`` in project ``slug`` (deployment-wide kinds ignore the project); its id."""
    if kind in ('noop', 'sleep_noop'):
        return _run(_START, kind, params)
    proc = compose(
        'exec', '-T', 'api', 'python', '-',
        stdin=_START_IN_PROJECT.format(kind=kind, slug=slug, params=params),
        check=False, timeout=90,
    )  # fmt: skip
    if proc.returncode != 0:
        raise StartFailedError(proc.stderr)
    return str(json.loads(proc.stdout.strip().splitlines()[-1])['job_id'])


def start_noop_burst(count: int, run: str) -> float:
    """Start ``count`` noop jobs tagged ``params.load_run = run``; the seconds the starts took."""
    proc = compose(
        'exec', '-T', 'api', 'python', '-',
        stdin=_START_BURST.format(count=count, run=run), check=False, timeout=900,
    )  # fmt: skip
    if proc.returncode != 0:
        raise StartFailedError(proc.stderr)
    return float(json.loads(proc.stdout.strip().splitlines()[-1])['seconds'])


def _job_project(job_id: str) -> str:
    return compose(
        'exec', '-T', 'postgres', 'psql', '-U', 'op_super', '-d', 'openprocessor', '-At',
        '-c', f"SELECT coalesce(project_id::text, '') FROM jobs WHERE id = '{job_id}'",
    ).stdout.strip()  # fmt: skip


def cancel_job_in_container(job_id: str) -> dict[str, Any]:
    """Cancel through the job service (what the family cancel routes call): ``status``, ``changed``."""
    proc = compose(
        'exec', '-T', 'api', 'python', '-', stdin=_CANCEL.format(job_id=job_id, project=_job_project(job_id)),
        check=False, timeout=90,
    )  # fmt: skip
    if proc.returncode != 0:
        raise StartFailedError(proc.stderr)
    result: dict[str, Any] = json.loads(proc.stdout.strip().splitlines()[-1])
    return result


def send_task(task: str, task_id: str) -> None:
    """Send the named task to its routed queue with a known id (the log of its consumer shows it)."""
    proc = compose(
        'exec', '-T', 'api', 'python', '-',
        stdin=_SEND_TASK.format(task=task, task_id=task_id), check=False, timeout=90,
    )  # fmt: skip
    if proc.returncode != 0:
        raise StartFailedError(proc.stderr)


def start_infra_job_undispatched(kind: str, **params: Any) -> str:
    """Commit a queued job but do not send its message (reaper rule c must dispatch it)."""
    return _run(_START_UNDISPATCHED, kind, params)


def get_job(api: Any, job_id: str) -> dict[str, Any] | None:
    """The job from the deployment-wide list; ``None`` when it is not on the first page."""
    resp = api.get('/jobs', params={'limit': 200})
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
