"""Execute an assist run: chunk the sampled ids, ask the VLM, record the outcome.

``start_run`` queues the ``assist_run`` job; a Celery worker runs :func:`run_assist` with the
job context. Progress, the cancel check and the result go through the job row
(:mod:`~src.services.curation.assist.job`); a run that ends in an error is a failed job whose
message is the error.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from src.services.curation.assist import job as assist_job
from src.services.curation.assist.models import (
    AssistResult,
    AssistRunRequest,
    AssistRunState,
    AttributesResult,
    ClassifyResult,
    ResolvedScope,
    SegmentResult,
    VerifyResult,
)
from src.services.curation.assist.prepare import prepare_queued
from src.services.curation.assist.proposal import compute_proposal
from src.services.curation.assist.sample_eval import ask_sample
from src.services.curation.assist.scope_now import current_members
from src.services.curation.assist.segment import requeue_for_run, wait_for_region
from src.services.curation.assist.slot_write import RunMeta
from src.services.curation.class_write_guard import ClassWriteGuard
from src.services.curation.vlm_class_batch import class_context
from src.services.labeling.run_labeler import get_vlm_labeler
from src.services.labeling.vlm_factory import assert_may_connect
from src.utils.timeutil import now_iso


if TYPE_CHECKING:
    import uuid
    from collections.abc import Sequence

    from opensearchpy import AsyncOpenSearch
    from sqlalchemy.ext.asyncio import AsyncSession

    from src.services.curation.assist.chunk import ChunkOutcome
    from src.services.curation.assist.prepare import PreparedRun
    from src.services.curation.assist.proposal_models import Proposal
    from src.tasks.runtime.context import JobContext

CONCURRENCY = 4


async def _scope_members(opensearch: AsyncOpenSearch, prepared: PreparedRun) -> list[str] | None:
    """The members of the (sub-)cluster the run's proposal is about, kept for the
    staleness check: the resolved scope of a cluster target, the context's members
    for a selection made in one, ``None`` for a bare selection or a filter."""
    if prepared.request.target.kind == 'cluster':
        return prepared.scope.members
    current = await current_members(opensearch, prepared.request)
    return None if current is None else [cid for cid, _ in current]


def _meta(ctx: JobContext, created_at: str, labeler: object, prepared: PreparedRun) -> RunMeta:
    identity = labeler.identity  # type: ignore[attr-defined]
    return RunMeta(
        run_id=str(ctx.job_id),
        created_at=created_at,
        vlm_endpoint=identity.endpoint_ref,
        vlm_model=identity.model,
        prompt_pack=prepared.pack.name,
        prompt_pack_revision=prepared.pack_revision,
    )


def _result(
    request_actions: Sequence[str],
    total: ChunkOutcome,
    proposal: Proposal,
    segment: SegmentResult | None,
) -> AssistResult:
    return AssistResult(
        classify=(
            ClassifyResult(
                predicted=total.predicted,
                updated=total.updated,
                failed=total.failed,
                unreadable=total.unreadable,
                locked_suggest_only=total.locked_suggest_only,
            )
            if 'classify' in request_actions
            else None
        ),
        attributes=(
            AttributesResult(items_written=total.items_written, invalid_values=total.invalid_values)
            if 'attributes' in request_actions
            else None
        ),
        verify=(
            VerifyResult(
                boxes=total.verify_boxes,
                accepted=total.verify_accepted,
                rejected=total.verify_rejected,
                locked_skipped=total.verify_locked,
            )
            if 'verify' in request_actions
            else None
        ),
        segment=segment,
        proposal=proposal,
    )


async def _segment_stage(
    ctx: JobContext, opensearch: AsyncOpenSearch, prepared: PreparedRun
) -> SegmentResult | None:
    """Re-queue the run's items for the region worker and wait for it; ``None`` when the
    run does not segment."""
    spec = prepared.request.segment
    if 'segment' not in prepared.request.actions or spec is None:
        return None
    ids = prepared.scope.sampled_ids
    run_id = str(ctx.job_id)
    queued = await requeue_for_run(opensearch, ids, run_id=run_id, verify=spec.verify)
    await ctx.progress(0, queued, assist_job.WAITING_STAGE, force=True)
    result = await wait_for_region(ctx, opensearch, run_id, queued)
    await ctx.progress(0, len(ids), 'asking', force=True)
    return result


async def start_run(
    session: AsyncSession, project: uuid.UUID, opensearch: AsyncOpenSearch, prepared: PreparedRun
) -> AssistRunState:
    """Queue the run (409 ``job_busy`` while another one of the project is active)."""
    job = await assist_job.start(
        session, project, prepared, members=await _scope_members(opensearch, prepared)
    )
    return assist_job.queued_state(job, prepared)


async def run_assist(ctx: JobContext, opensearch: AsyncOpenSearch) -> tuple[dict[str, Any], bool]:
    """Run the queued job to its end: the :class:`AssistResult` dump, and whether any VLM
    request failed or any crop was unreadable (the handler wraps it as a result with errors).

    A cancel ends the run after the chunks in flight (the runtime records ``cancelled``);
    any error fails the job with its message."""
    params = ctx.params
    request = AssistRunRequest.model_validate(params['request'])
    prepared = await prepare_queued(
        opensearch,
        request,
        ResolvedScope.model_validate(params['resolved']),
        sampled_ids=list(params['sampled_ids']),
        box_ids=params.get('box_ids'),
    )
    endpoint = prepared.vlm.endpoint
    if endpoint is not None:
        await assert_may_connect(endpoint)
    labeler = get_vlm_labeler(
        prepared.pack_name, prepared.pack_revision, endpoint=endpoint, usage_source='assist_run'
    )
    vlm_ctx = await class_context(
        labeler, ClassWriteGuard('assist_run'), revision=prepared.pack_revision
    )
    meta = _meta(ctx, now_iso(), labeler, prepared)
    actions = request.actions
    ids = prepared.scope.sampled_ids
    segment = await _segment_stage(ctx, opensearch, prepared)
    done = 0

    async def progress(processed: int) -> None:
        nonlocal done
        done += processed
        await ctx.progress(done, len(ids), 'asking')

    total = await ask_sample(
        opensearch,
        vlm_ctx,
        ids,
        actions=actions,
        box_ids=prepared.scope.box_ids,
        images_per_call=prepared.images_per_call,
        meta=meta,
        semaphore=asyncio.Semaphore(CONCURRENCY),
        should_stop=ctx.cancelled,
        on_chunk=progress,
    )
    nothing_to_ask = bool({'verify', 'segment'} & set(actions)) and not (
        total.failed or total.unreadable
    )
    if (
        ids
        and not ctx.cancelled()
        and total.predicted + total.verify_boxes == 0
        and not nothing_to_ask
    ):
        msg = (
            f'the VLM answered nothing: {total.failed} request(s) failed and '
            f'{total.unreadable} crop image(s) could not be read, of {len(ids)} selected. '
            'Check that the VLM endpoint is reachable and the source images are mounted.'
        )
        raise RuntimeError(msg)
    proposal = compute_proposal(
        total.answers,
        min_agreement=request.min_agreement,
        registry_names=frozenset(vlm_ctx.class_names),
    )
    result = _result(actions, total, proposal, segment).model_dump(mode='json')
    return result, bool(total.failed or total.unreadable)


__all__ = ['CONCURRENCY', 'run_assist', 'start_run']
