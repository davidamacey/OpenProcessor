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
from src.services.curation.assist.chunk import ChunkOutcome, process_chunk
from src.services.curation.assist.job import create_run
from src.services.curation.assist.models import (
    AssistProgress,
    AssistResult,
    AssistRunState,
    AttributesResult,
    ClassifyResult,
)
from src.services.curation.assist.proposal import compute_proposal
from src.services.curation.assist.scope_now import current_members
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
    request_actions: Sequence[str], total: ChunkOutcome, proposal: Proposal
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
        proposal=proposal,
    )


def _sum(outcomes: list[ChunkOutcome]) -> ChunkOutcome:
    total = ChunkOutcome()
    for o in outcomes:
        for name in vars(total):
            setattr(total, name, getattr(total, name) + getattr(o, name))
    return total


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
        size = max(1, prepared.images_per_call)
        chunks = [ids[i : i + size] for i in range(0, len(ids), size)]
        sem = asyncio.Semaphore(CONCURRENCY)
        done = 0

        async def gated(chunk: list[str]) -> ChunkOutcome | None:
            nonlocal done
            async with sem:
                if job.cancel_requested():
                    return None
                out = await process_chunk(
                    opensearch,
                    ctx,
                    chunk,
                    classify='classify' in actions,
                    attributes='attributes' in actions,
                    meta=meta,
                )
                done += len(chunk)
                job.update(progress={'processed': done, 'total': len(ids)}, updated_at=now_iso())
                job.touch_heartbeat()
                return out

        results = await asyncio.gather(*[gated(c) for c in chunks])
        total = _sum([r for r in results if r is not None])
        cancelled = job.cancel_requested()
        if ids and not cancelled and total.predicted == 0:
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
            result=_result(actions, total, proposal).model_dump(mode='json'),
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
