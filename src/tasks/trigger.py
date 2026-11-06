"""Tasks of the `trigger` family: the auto-label sweep (and, as each moves, the assist loop and the VLM batches).

The OpenSearch client is injected here, never imported by ``src/tasks/runtime/`` (review R-2).
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from src.services.curation.auto_label_run import run_auto_label
from src.services.curation.autolabel.progress import JobProgress, PauseRequestedError, failed_stages
from src.services.curation.job_binding import bound_job_project, opensearch_client, project_record
from src.tasks.runtime.context import Paused
from src.tasks.runtime.register import register_job


if TYPE_CHECKING:
    from src.tasks.runtime.context import JobContext

_FACTORIES = {'opensearch': opensearch_client}


async def autolabel(ctx: JobContext) -> dict[str, Any] | Paused:
    """One auto-label run of the job's project, through the existing run coroutine.

    Progress is flushed once a second by a sibling task. A pause is honoured at a stage start and
    between VLM chunks and returns ``Paused``; the resumed job runs the idempotent run again, and
    its VLM stage selects only what is still unvalidated.
    """
    record = await project_record(ctx)
    params = dict(ctx.params)
    with bound_job_project(ctx, record):
        opensearch = await ctx.client('opensearch')
        progress = JobProgress(ctx)
        stop = asyncio.Event()
        flusher = asyncio.create_task(progress.flush_loop(stop))
        try:
            result = await run_auto_label(opensearch=opensearch, progress=progress, **params)
        except PauseRequestedError:
            return Paused()
        except asyncio.CancelledError:
            if not ctx.cancelled():
                raise
            return {}
        finally:
            stop.set()
            await flusher
            progress.finalize()
            await progress.flush()
    failed = failed_stages(result)
    if failed:
        raise RuntimeError(f'stage(s) failed: {", ".join(failed)}')
    return result if isinstance(result, dict) else {'raw': str(result)}


register_job('autolabel', autolabel, client_factories=_FACTORIES)
