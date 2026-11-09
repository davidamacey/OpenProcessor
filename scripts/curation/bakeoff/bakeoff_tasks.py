"""The ``bakeoff`` job consumer: the evaluator image's Celery worker.

``celery -A scripts.curation.bakeoff.bakeoff_tasks worker -Q gpu_eval -c 1`` serves the
``gpu_eval`` queue. The API starts a ``bakeoff`` job (``POST /bakeoff/run``) with the job spec
v2 as its parameters (documented in :mod:`scripts.curation.bakeoff.bakeoff_runner`); the job
wrapper claims the row, so a redelivered message finds it running or finished and is
acknowledged without a second evaluation (https://docs.celeryq.dev/en/stable/userguide/tasks.html,
"Should I use retry or acks_late?"; the claim is the platform's idempotency, 02b section 3.1).
The module is named after its worker, not ``celery_tasks``, which the trainer image owns.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from celery.signals import worker_init

from src.config.retired_env import reject_retired_env
from src.db.repos import bakeoff as bakeoff_repo
from src.tasks.runtime.context import JobResult
from src.tasks.runtime.gpu_app import make_app
from src.tasks.runtime.register import register_job

from .bakeoff_runner import run_job


if TYPE_CHECKING:
    from src.tasks.runtime.context import JobContext


app = make_app(['scripts.curation.bakeoff.bakeoff_tasks'])


def evaluate(spec: dict[str, Any]) -> dict[str, Any]:
    """Score the spec's models on its datasets and return the final ``status.json`` content.

    The seam the tests replace with a fake; the real run drives one subprocess per
    (dataset, model) pair on the evaluator's GPUs.
    """
    return run_job(spec)


async def bakeoff(ctx: JobContext) -> JobResult:
    """Run one bake-off; the results are the files under the spec's ``out_dir``.

    ``bakeoff_runs.results_storage_key`` records that directory. A run that ends in the
    ``error`` state fails the job; a run that scored some pairs and failed others is
    ``completed_with_errors``.
    """
    spec = dict(ctx.params)
    await ctx.progress(0, None, 'evaluating')
    status = await asyncio.to_thread(evaluate, spec)
    out_dir = str(spec['out_dir'])
    async with ctx.session() as session:
        await session.run_sync(lambda sync: bakeoff_repo.set_results_key(sync, ctx.job_id, out_dir))
    if status.get('state') != 'done':
        raise RuntimeError(
            f'bake-off {spec["job_id"]} ended {status.get("state")}: {status.get("error")}'
        )
    failed = list(status.get('failed') or [])
    progress = status.get('progress') or {}
    await ctx.progress(progress.get('done', 0), progress.get('total'), 'evaluating', force=True)
    return JobResult(
        {
            'out_dir': out_dir,
            'completed': len(status.get('completed') or []),
            'failed': len(failed),
        },
        with_errors=bool(failed),
    )


register_job('bakeoff', bakeoff)


@worker_init.connect(weak=False)
def _on_worker_init(**_: Any) -> None:
    from src.db import schema_check

    reject_retired_env()
    schema_check.require_current_schema()
