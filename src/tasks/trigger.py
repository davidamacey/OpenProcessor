"""Tasks of the `trigger` family: the auto-label sweep (and, as each moves, the assist loop and the VLM batches).

The OpenSearch client is injected here, never imported by ``src/tasks/runtime/`` (review R-2).
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from src.services.curation import vlm_batch
from src.services.curation.auto_label_run import run_auto_label
from src.services.curation.autolabel.progress import JobProgress, PauseRequestedError, failed_stages
from src.services.curation.job_binding import bound_job_project, opensearch_client, project_record
from src.services.jobs.vlm_batch_sweep import run_sweep
from src.tasks.app import app
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


async def _labeler(params: dict[str, Any], usage_source: str) -> Any:
    from src.services.config_store import get_config_store
    from src.services.curation.autolabel.run_vlm import job_endpoint
    from src.services.labeling.run_labeler import get_vlm_labeler, resolve_pack
    from src.services.labeling.vlm_factory import assert_may_connect

    # The worker's config snapshot starts cold; the pinned revision is read by id.
    await get_config_store().ensure_fresh()
    pack_name = params.get('pack_name')
    endpoint = await job_endpoint(
        vlm=None,
        acknowledge_external=False,
        pack=resolve_pack(pack_name),
        pinned_name=params.get('vlm_endpoint'),
        pinned_revision=params.get('vlm_endpoint_revision'),
        resolved=True,
    )
    if endpoint is not None:
        await assert_may_connect(endpoint)
    return get_vlm_labeler(pack_name, endpoint=endpoint, usage_source=usage_source)


async def vlm_label_batch(ctx: JobContext) -> dict[str, Any]:
    """Label the job's crops against the endpoint and pack pinned when it was queued."""
    record = await project_record(ctx)
    params = dict(ctx.params)
    with bound_job_project(ctx, record):
        labeler = await _labeler(params, 'label_batch')
        return await vlm_batch.run_label_batch(
            await ctx.client('opensearch'), labeler, params['crop_ids']
        )


async def vlm_verify_region_batch(ctx: JobContext) -> dict[str, Any]:
    """Verify the staged region images; the staging prefix goes whatever the outcome."""
    record = await project_record(ctx)
    params = dict(ctx.params)
    try:
        with bound_job_project(ctx, record):
            labeler = await _labeler(params, 'region_verify')
            return await vlm_batch.run_verify_region_batch(labeler, params['items'])
    finally:
        await asyncio.to_thread(vlm_batch.drop_staged, params['batch_id'])


@app.task(
    name='op.periodic.vlm_batch_staging_sweep',
    shared=False,
    soft_time_limit=120,
    time_limit=150,
    ignore_result=True,
)
def vlm_batch_staging_sweep() -> None:
    """Delete staged verify-region images whose job is terminal or long gone."""
    run_sweep()


register_job('autolabel', autolabel, client_factories=_FACTORIES)
register_job('vlm_label_batch', vlm_label_batch, client_factories=_FACTORIES)
register_job('vlm_verify_region_batch', vlm_verify_region_batch, client_factories=_FACTORIES)
