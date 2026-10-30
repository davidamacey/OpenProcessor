"""Execute an assist run: chunk the sampled ids, ask the VLM, record the outcome.

The run is an asyncio task in the API process (like a reprocess job). It reads
only the prepared run and the dependencies it is handed; progress, cancel and
the final state go through the run's :class:`~src.services.curation.file_job.FileJob`.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import TYPE_CHECKING

import httpx
from opensearchpy.exceptions import OpenSearchException

from src.core.exceptions import RequestRejectedError
from src.core.logging import get_logger
from src.services.curation.assist.job import create_run
from src.services.curation.assist.models import (
    AssistProgress,
    AssistResult,
    AssistRunState,
    AttributesResult,
    ClassifyResult,
    SegmentResult,
    VerifyResult,
)
from src.services.curation.assist.proposal import compute_proposal
from src.services.curation.assist.sample_eval import ask_sample
from src.services.curation.assist.scope_now import current_members
from src.services.curation.assist.segment import requeue_for_run, wait_for_region
from src.services.curation.assist.slot_write import RunMeta
from src.services.curation.class_write_guard import ClassWriteGuard
from src.services.curation.file_job import FileJob, heartbeat_ticker
from src.services.curation.vlm_class_batch import class_context
from src.services.labeling.run_labeler import get_vlm_labeler
from src.services.labeling.vlm_endpoints import VlmEndpointUnavailableError
from src.services.labeling.vlm_factory import assert_may_connect
from src.utils.timeutil import now_iso


if TYPE_CHECKING:
    from collections.abc import Sequence

    from opensearchpy import AsyncOpenSearch

    from src.services.curation.assist.chunk import ChunkOutcome
    from src.services.curation.assist.prepare import PreparedRun
    from src.services.curation.assist.proposal_models import Proposal

logger = get_logger(__name__)

# What ends a run as ``failed`` with its message: a refused or unreachable VLM, a datastore
# error, an unreadable file. Anything else is a bug and propagates.
_RUN_ERRORS = (
    RuntimeError,
    ValueError,
    OSError,
    OpenSearchException,
    httpx.HTTPError,
    RequestRejectedError,
    VlmEndpointUnavailableError,
)

CONCURRENCY = 4

_tasks: dict[str, asyncio.Task[None]] = {}


async def start_run(opensearch: AsyncOpenSearch, prepared: PreparedRun) -> AssistRunState:
    """Claim the project's single run slot, persist the run and schedule it.
    Raises the 409 ``assist_busy`` refusal when a run is live."""
    created = now_iso()
    state = AssistRunState(
        status='queued',
        created_at=created,
        request=prepared.request,
        resolved=prepared.resolved(),
        progress=AssistProgress(total=len(prepared.scope.sampled_ids)),
    )
    job = create_run(
        state=state.model_dump(mode='json'),
        request=prepared.request.model_dump(mode='json'),
        members=await _scope_members(opensearch, prepared),
    )
    _tasks[job.directory.name] = asyncio.create_task(
        run_assist(job, opensearch, prepared, created_at=created)
    )
    state.run_id = job.directory.name
    return state


async def _scope_members(opensearch: AsyncOpenSearch, prepared: PreparedRun) -> list[str] | None:
    """The members of the (sub-)cluster the run's proposal is about, kept for the
    staleness check: the resolved scope of a cluster target, the context's members
    for a selection made in one, ``None`` for a bare selection or a filter."""
    if prepared.request.target.kind == 'cluster':
        return prepared.scope.members
    current = await current_members(opensearch, prepared.request)
    return None if current is None else [cid for cid, _ in current]


def _meta(job: FileJob, prepared: PreparedRun, created_at: str, labeler: object) -> RunMeta:
    identity = labeler.identity  # type: ignore[attr-defined]
    return RunMeta(
        run_id=job.directory.name,
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
    job: FileJob, opensearch: AsyncOpenSearch, prepared: PreparedRun
) -> SegmentResult | None:
    """Re-queue the run's items for the region worker and wait for it; ``None`` when the
    run does not segment."""
    spec = prepared.request.segment
    if 'segment' not in prepared.request.actions or spec is None:
        return None
    ids = prepared.scope.sampled_ids
    queued = await requeue_for_run(opensearch, ids, run_id=job.directory.name, verify=spec.verify)
    job.update(status='waiting_region', progress={'processed': 0, 'total': queued})
    result = await wait_for_region(job, opensearch, job.directory.name, queued)
    job.update(status='running', progress={'processed': 0, 'total': len(ids)})
    return result


async def run_assist(
    job: FileJob, opensearch: AsyncOpenSearch, prepared: PreparedRun, *, created_at: str
) -> None:
    """Run ``prepared`` to a terminal state; the state file is the result."""
    job.update(status='running', updated_at=now_iso())
    ticker = asyncio.create_task(heartbeat_ticker(job))
    try:
        endpoint = prepared.vlm.endpoint
        if endpoint is not None:
            await assert_may_connect(endpoint)
        labeler = get_vlm_labeler(prepared.pack_name, prepared.pack_revision, endpoint=endpoint)
        ctx = class_context(labeler, ClassWriteGuard('assist_run'), revision=prepared.pack_revision)
        meta = _meta(job, prepared, created_at, labeler)
        actions = prepared.request.actions
        ids = prepared.scope.sampled_ids
        segment = await _segment_stage(job, opensearch, prepared)
        done = 0

        def progress(processed: int) -> None:
            nonlocal done
            done += processed
            job.update(progress={'processed': done, 'total': len(ids)}, updated_at=now_iso())
            job.touch_heartbeat()

        total = await ask_sample(
            opensearch,
            ctx,
            ids,
            actions=actions,
            box_ids=prepared.scope.box_ids,
            images_per_call=prepared.images_per_call,
            meta=meta,
            semaphore=asyncio.Semaphore(CONCURRENCY),
            should_stop=job.cancel_requested,
            on_chunk=progress,
        )
        cancelled = job.cancel_requested()
        nothing_to_ask = bool({'verify', 'segment'} & set(actions)) and not (
            total.failed or total.unreadable
        )
        if (
            ids
            and not cancelled
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
            min_agreement=prepared.request.min_agreement,
            registry_names=frozenset(ctx.class_names),
        )
        status = (
            'cancelled'
            if cancelled
            else 'completed_with_errors'
            if total.failed or total.unreadable
            else 'completed'
        )
        job.update(
            status=status,
            result=_result(actions, total, proposal, segment).model_dump(mode='json'),
            finished_at=now_iso(),
            updated_at=now_iso(),
        )
    except asyncio.CancelledError:
        job.update(status='cancelled', finished_at=now_iso(), updated_at=now_iso())
        raise
    except _RUN_ERRORS as exc:
        logger.error('assist_run_failed', run=job.directory.name, error=str(exc))
        job.update(status='failed', error=str(exc), finished_at=now_iso(), updated_at=now_iso())
    finally:
        ticker.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await ticker


__all__ = ['CONCURRENCY', 'run_assist', 'start_run']
