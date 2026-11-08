"""Run a queued label operation in the test process, as the worker would.

The batch relabel and move routes queue a ``label_op`` job and answer 202 (4A-5); tests that
call a route function or the route itself then run the handler body here against the same
fake OpenSearch, so they can still assert on what the job wrote.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

from curation.jobs_a.ctx import FakeCtx
from src.config import get_curation_config
from src.config.project_context import current_project_id
from src.db.repos import jobs as repo
from src.db.session import async_session_factory, project_tx
from src.services.jobs.service import cancel_job


def _body(answer: Any) -> dict[str, Any]:
    """A route function answers a 202 as a ``JSONResponse`` and a dry run as a mapping."""
    if isinstance(answer, dict):
        return answer
    return dict(json.loads(answer.body))


async def run_job(job_id: str, runner: Any) -> Any:
    """Run ``runner(ctx)`` for the queued job, then end the row so its exclusive group is free."""
    project_id = current_project_id()
    async with async_session_factory()() as session, project_tx(session, project_id):
        job = await session.run_sync(lambda sync: repo.get(sync, uuid.UUID(job_id)))
    assert job is not None
    try:
        return await runner(FakeCtx(project_id, job.id, job.params))
    finally:
        async with async_session_factory()() as session:
            await cancel_job(session, job.id, project_id, requested_by='test')


async def written(fake: Any, accepted: dict[str, Any]) -> dict[str, Any]:
    """A 202 body -> run the relabel job -> the shape the synchronous route used to answer."""
    from src.services.curation.label_ops.record import run_relabel

    accepted = _body(accepted)
    if accepted.get('job_id') is None:  # an empty selection queues nothing; a dry run is a read
        return accepted
    result = await run_job(accepted['job_id'], lambda ctx: run_relabel(ctx, fake))
    docs = fake.docs(get_curation_config().items_index)
    return {
        'updated': result['updated'],
        'updated_ids': sorted(
            i for i, d in docs.items() if d.get('label_op_id') == result['op_id']
        ),
        'conflicts': result['conflicts'],
        'op_id': result['op_id'],
    }


async def call_route(route: Any, payload: Any, fake: Any) -> dict[str, Any]:
    """Call a batch label or move route function with a real session, then run its job."""
    async with async_session_factory()() as session:
        accepted = await route(payload, fake, session)
    return await written(fake, accepted)


async def call_raw(route: Any, payload: Any, fake: Any) -> Any:
    """Call a route function with a real session and return what it answers, queued or not."""
    async with async_session_factory()() as session:
        return await route(payload, fake, session)
