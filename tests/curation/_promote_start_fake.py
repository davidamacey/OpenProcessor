"""The promote route queues a job; route tests replace the queueing with a capture of the plan.

``captured`` receives ``plan``, ``payload`` and ``class_id_to_name`` of the validated promote, so a
test asserts what the route decided (class names, remap, gate) without a database or a broker.
"""

from __future__ import annotations

import contextlib
import datetime as dt
import uuid
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

from src.schemas.jobs import JobView


if TYPE_CHECKING:
    import pytest


PROJECT_ID = uuid.uuid4()
JOB_ID = uuid.uuid4()


@contextlib.asynccontextmanager
async def _no_session() -> Any:
    yield None


def no_database(monkeypatch: pytest.MonkeyPatch) -> None:
    """The route's session dependency yields nothing: a test that never reaches the database."""
    monkeypatch.setattr('src.core.db_session.async_session_factory', lambda: _no_session)


def install(monkeypatch: pytest.MonkeyPatch, captured: dict[str, Any]) -> None:
    no_database(monkeypatch)

    async def start_promote(
        _session: Any, _project: Any, plan: Any, payload: Any, **_kw: Any
    ) -> Any:
        captured.update(plan=plan, payload=payload, class_id_to_name=plan.class_id_to_name)
        return SimpleNamespace(job=SimpleNamespace(id=JOB_ID), created=True)

    async def get_job(_session: Any, job_id: Any, *, project: Any) -> JobView:
        plan = captured['plan']
        now = dt.datetime(2026, 1, 1, tzinfo=dt.UTC)
        return JobView(
            id=str(job_id),
            kind='promote',
            project='default',
            status='queued',
            scope=plan.triton_name,
            queue='jobs',
            attempt=0,
            retry_count=0,
            max_attempts=3,
            pause_reason=None,
            pause_requested=False,
            cancel_requested=False,
            progress={},
            progress_updated_at=None,
            result=None,
            error_code=None,
            error_message=None,
            params={'run_job_id': plan.job_id, 'triton_name': plan.triton_name},
            requested_by='api',
            created_at=now,
            queued_at=now,
            started_at=None,
            finished_at=None,
            updated_at=now,
        )

    monkeypatch.setattr('src.services.training.promote_job.start_promote', start_promote)
    monkeypatch.setattr('src.routers.curation.train.promote.job_status.get_job', get_job)
    monkeypatch.setattr('src.routers.curation.train.promote.current_project_id', lambda: PROJECT_ID)


def run_job(captured: dict[str, Any]) -> Any:
    """What the worker does with the queued plan: ``execute_promote`` (the promoter is faked by
    the test; the recorder is never used by the fake)."""
    import asyncio

    from src.services.training.promote_run import execute_promote

    return asyncio.run(execute_promote(captured['plan'], captured['payload'], object()))
