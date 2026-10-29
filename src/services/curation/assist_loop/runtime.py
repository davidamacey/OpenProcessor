"""Run one queued loop in the worker: rebuild everything the loop needs from its job directory.

The API wrote the normalized request to ``request.json`` and the endpoints and prompt pack it
pinned into ``state.json``. This resolves exactly those pins by id (the worker's registry
snapshot is its own and starts cold), re-checks that the endpoint may be connected to, and
builds the asker, the budget, the optional audit and the partition reader the driver runs on.
A loop that cannot be set up ends ``failed`` with the reason, like one that fails while
running; a resumed loop goes through the same path and continues from its event log.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from src.core.logging import get_logger
from src.schemas.curation.assist_loop import AssistLoopRequest
from src.services.curation.assist.prepare import images_per_call_for
from src.services.curation.assist.slot_write import RunMeta
from src.services.curation.assist_loop import job
from src.services.curation.assist_loop.audit import AuditJudge, Auditor, AuditParams
from src.services.curation.assist_loop.budget import BudgetLimits, LoopBudget
from src.services.curation.assist_loop.driver import (
    RUN_ERRORS,
    LoopDeps,
    LoopOutcome,
    LoopParams,
    run_loop,
)
from src.services.curation.assist_loop.evaluate import VlmAsker
from src.services.curation.assist_loop.partition import ClusterFieldPartition, PartitionScope
from src.services.curation.assist_loop.store import LoopStore
from src.services.curation.autolabel.run_vlm import job_endpoint
from src.services.curation.class_write_guard import ClassWriteGuard
from src.services.curation.vlm_class_batch import class_context
from src.services.labeling.run_labeler import get_vlm_labeler
from src.services.labeling.vlm_factory import assert_may_connect
from src.utils.timeutil import now_iso


if TYPE_CHECKING:
    from opensearchpy import AsyncOpenSearch

logger = get_logger(__name__)


def _request(store: LoopStore) -> AssistLoopRequest:
    raw = store.read_json('request.json')
    if raw is None:
        msg = f'request.json of {store.loop_id} is missing'
        raise ValueError(msg)
    return AssistLoopRequest.model_validate(raw)


@dataclass(frozen=True)
class _Model:
    labeler: Any
    endpoint: Any

    @property
    def images_per_call(self) -> int:
        return images_per_call_for(self.endpoint)


async def _resolve(
    opensearch: AsyncOpenSearch, pin: dict[str, Any], pack: dict[str, Any]
) -> _Model:
    """A pinned endpoint resolved by id, gated again, and its labeler."""
    endpoint = await job_endpoint(
        opensearch,
        vlm=None,
        acknowledge_external=False,
        pack=None,
        pinned_name=pin.get('name'),
        pinned_revision=pin.get('revision'),
        resolved=True,
    )
    if endpoint is None:
        msg = 'the loop pinned no VLM endpoint'
        raise ValueError(msg)
    await assert_may_connect(endpoint)
    labeler = get_vlm_labeler(pack.get('name'), pack.get('revision'), endpoint=endpoint)
    return _Model(labeler, endpoint)


def _budget(store: LoopStore, request: AssistLoopRequest, images_per_call: int) -> LoopBudget:
    caps = request.budget
    return LoopBudget(
        store,
        BudgetLimits.from_request(
            max_vlm_crops=caps.max_vlm_crops if caps else None,
            max_vlm_requests=caps.max_vlm_requests if caps else None,
            max_wall_s=caps.max_wall_s if caps else None,
            images_per_call=images_per_call,
            vlm_concurrency=request.vlm_concurrency,
            audit_max_crops=request.audit.max_crops,
        ),
    )


def _auditor(
    opensearch: AsyncOpenSearch,
    store: LoopStore,
    budget: LoopBudget,
    request: AssistLoopRequest,
    judge: _Model,
    revision: int | None,
) -> Auditor:
    audit = request.audit
    return Auditor(
        store,
        budget,
        AuditJudge(
            opensearch,
            class_context(judge.labeler, ClassWriteGuard('assist_loop_audit'), revision=revision),
            images_per_call=judge.images_per_call,
            semaphore=asyncio.Semaphore(request.vlm_concurrency),
        ),
        AuditParams(
            rate=audit.rate,
            min_per_batch=audit.min_per_batch,
            floor=audit.floor,
            on_fail=audit.on_fail,
        ),
        seed=request.seed,
    )


def _asker(
    opensearch: AsyncOpenSearch,
    store: LoopStore,
    request: AssistLoopRequest,
    primary: _Model,
    pack: dict[str, Any],
) -> VlmAsker:
    identity = primary.labeler.identity
    revision = pack.get('revision')
    meta = RunMeta(
        run_id=store.loop_id,
        created_at=str(store.job.read().get('created_at') or now_iso()),
        vlm_endpoint=identity.endpoint_ref,
        vlm_model=identity.model,
        prompt_pack=str(pack['name']),
        prompt_pack_revision=revision,
    )
    return VlmAsker(
        opensearch,
        class_context(primary.labeler, ClassWriteGuard('assist_loop'), revision=revision),
        meta,
        images_per_call=primary.images_per_call,
        semaphore=asyncio.Semaphore(request.vlm_concurrency),
    )


async def run_job(opensearch: AsyncOpenSearch, loop_id: str) -> LoopOutcome:
    """Run (or resume) the loop ``loop_id`` of the bound project to a stop; its state file
    carries the outcome."""
    store = LoopStore(job.open_job(loop_id).directory)
    try:
        request = _request(store)
        pinned = store.job.read().get('pinned') or {}
        pack = pinned.get('prompt_pack') or {}
        if not pack.get('name'):
            msg = 'the loop pinned no prompt pack'
            raise ValueError(msg)
        primary = await _resolve(opensearch, pinned.get('vlm') or {}, pack)
        judge = (
            await _resolve(opensearch, pinned['audit_vlm'], pack)
            if pinned.get('audit_vlm')
            else None
        )
        budget = _budget(store, request, primary.images_per_call)
        deps = LoopDeps(
            opensearch=opensearch,
            reader=ClusterFieldPartition(opensearch),
            asker=_asker(opensearch, store, request, primary, pack),
            auditor=(
                _auditor(opensearch, store, budget, request, judge, pack.get('revision'))
                if judge is not None
                else None
            ),
        )
    except RUN_ERRORS as exc:
        logger.error('assist_loop_failed', loop=loop_id, error=str(exc))
        store.job.update(
            status='failed', error=str(exc), finished_at=now_iso(), updated_at=now_iso()
        )
        return LoopOutcome('failed', error=str(exc))
    params = LoopParams(
        seed=request.seed,
        theta=request.gate.theta,
        alpha=request.gate.alpha,
        max_dissent=request.gate.max_dissent,
        max_depth=request.recursion.max_depth,
        branching=request.recursion.branching,
        min_node_size=request.recursion.min_node_size,
        node_concurrency=request.node_concurrency,
        images_per_call=primary.images_per_call,
    )
    scope = PartitionScope(
        cluster_ids=request.scope.cluster_ids, min_root_size=request.scope.min_root_size
    )
    return await run_loop(store, params, deps, budget, scope)


__all__ = ['run_job']
