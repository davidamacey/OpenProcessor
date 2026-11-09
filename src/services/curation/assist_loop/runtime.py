"""Set up one slice of a loop: resolve exactly what the run pinned and build what the driver
runs on (the asker, the budget, the optional audit and the partition reader).

The run row holds the normalized request and the endpoints and prompt pack it pinned. This
resolves those pins by id (the worker's registry snapshot is its own and starts cold),
re-checks that the endpoint may be connected to, and builds the pieces. A loop that cannot be
set up raises, like one that fails while running.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from src.domain.embedding_spaces import Target
from src.schemas.curation.assist_loop import AssistLoopRequest
from src.services.curation.assist.prepare import images_per_call_for
from src.services.curation.assist.slot_write import RunMeta
from src.services.curation.assist_loop.audit import AuditJudge, Auditor, AuditParams
from src.services.curation.assist_loop.driver import LoopDeps, LoopParams
from src.services.curation.assist_loop.evaluate import VlmAsker
from src.services.curation.assist_loop.partition import PartitionReader, PartitionScope
from src.services.curation.autolabel.run_vlm import job_endpoint
from src.services.curation.class_write_guard import ClassWriteGuard
from src.services.curation.vlm_class_batch import class_context
from src.services.embedding_spaces.resolve import embedding_space
from src.services.labeling.run_labeler import get_vlm_labeler
from src.services.labeling.vlm_factory import assert_may_connect
from src.utils.timeutil import now_iso


if TYPE_CHECKING:
    from opensearchpy import AsyncOpenSearch

    from src.db.repos.assist_loop_record import RunRow
    from src.services.curation.assist_loop.budget import LoopBudget
    from src.services.curation.assist_loop.record import LoopRecord


@dataclass(frozen=True)
class _Model:
    labeler: Any
    endpoint: Any

    @property
    def images_per_call(self) -> int:
        return images_per_call_for(self.endpoint)


@dataclass(frozen=True)
class Built:
    deps: LoopDeps
    params: LoopParams
    scope: PartitionScope
    request: AssistLoopRequest


async def _resolve(pin: dict[str, Any], pack: dict[str, Any]) -> _Model:
    """A pinned endpoint resolved by id, gated again, and its labeler."""
    endpoint = await job_endpoint(
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
    labeler = get_vlm_labeler(
        pack.get('name'), pack.get('revision'), endpoint=endpoint, usage_source='assist_loop'
    )
    return _Model(labeler, endpoint)


async def _auditor(
    opensearch: AsyncOpenSearch,
    record: LoopRecord,
    budget: LoopBudget,
    request: AssistLoopRequest,
    judge: _Model,
    revision: int | None,
) -> Auditor:
    audit = request.audit
    return Auditor(
        record,
        budget,
        AuditJudge(
            opensearch,
            await class_context(
                judge.labeler, ClassWriteGuard('assist_loop_audit'), revision=revision
            ),
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


async def _asker(
    opensearch: AsyncOpenSearch,
    run: RunRow,
    request: AssistLoopRequest,
    primary: _Model,
    pack: dict[str, Any],
) -> VlmAsker:
    identity = primary.labeler.identity
    revision = pack.get('revision')
    meta = RunMeta(
        run_id=str(run.id),
        created_at=run.created_at.isoformat() or now_iso(),
        vlm_endpoint=identity.endpoint_ref,
        vlm_model=identity.model,
        prompt_pack=str(pack['name']),
        prompt_pack_revision=revision,
    )
    return VlmAsker(
        opensearch,
        await class_context(primary.labeler, ClassWriteGuard('assist_loop'), revision=revision),
        meta,
        images_per_call=primary.images_per_call,
        semaphore=asyncio.Semaphore(request.vlm_concurrency),
    )


async def build(opensearch: AsyncOpenSearch, record: LoopRecord, budget: LoopBudget) -> Built:
    """Everything the driver runs on, from the run's pins."""
    run = record.run
    request = AssistLoopRequest.model_validate(run.request)
    pinned = run.pinned
    pack = pinned.get('prompt_pack') or {}
    if not pack.get('name'):
        msg = 'the loop pinned no prompt pack'
        raise ValueError(msg)
    primary = await _resolve(pinned.get('vlm') or {}, pack)
    judge = await _resolve(pinned['audit_vlm'], pack) if pinned.get('audit_vlm') else None
    deps = LoopDeps(
        opensearch=opensearch,
        reader=PartitionReader.for_run(
            opensearch, record.sessions, project=run.project_id, pinned=pinned
        ),
        asker=await _asker(opensearch, run, request, primary, pack),
        embedding_field=(await embedding_space(Target.ITEM)).vector_field,
        auditor=(
            await _auditor(opensearch, record, budget, request, judge, pack.get('revision'))
            if judge is not None
            else None
        ),
    )
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
        on_budget=request.on_budget,
    )
    scope = PartitionScope(
        cluster_uids=request.scope.cluster_uids, min_root_size=request.scope.min_root_size
    )
    return Built(deps, params, scope, request)


__all__ = ['Built', 'build']
