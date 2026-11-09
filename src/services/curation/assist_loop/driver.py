"""Run an auto-label loop slice: pin the partition, schedule nodes best-first, stop on the rules.

On its first slice the loop snapshots every root of the scope behind the storage seam and opens
it as a node; a later slice loads the tree from the tables and continues every node where it
was. Open nodes wait in a priority queue keyed by (-eligible members, path): the budget goes
first to the nodes that label the most items per VLM call, in an order fixed for a given tree.
Up to ``node_concurrency`` nodes are worked on at once; their VLM requests share one semaphore.

A slice ends when the queue is empty (``completed``, with the ``stop_reason`` ``done``, or the
budget that left nodes unvisited), on a cancel (``cancelled``), on a pause (``paused``: the
person's, or the loop's own after a failing audit or a reached budget), or at the first node
boundary after ``slice_s`` seconds with work left (``yield``). Requests in flight finish and
their answers are kept; nodes they belong to stay open.
"""

from __future__ import annotations

import asyncio
import heapq
from dataclasses import dataclass
from typing import TYPE_CHECKING

import httpx
from opensearchpy.exceptions import OpenSearchException

from src.core.exceptions import RequestRejectedError
from src.core.logging import get_logger
from src.db.repos.assist_loop_record import iteration_counts
from src.domain.loop_tree import best_first_key
from src.services.curation.assist.scope import membership_digest
from src.services.curation.assist_loop.emit import LoopEvents
from src.services.curation.assist_loop.evaluate import EvalContext, GateParams
from src.services.curation.assist_loop.node_step import (
    LoopRun,
    RunControl,
    admit,
    advance,
    crops_needed,
    finished,
    leave_unvisited,
)
from src.services.curation.assist_loop.partition import PINNED_KIND
from src.services.curation.assist_loop.progress import level_rows, totals
from src.services.curation.assist_loop.tree import VLM
from src.services.curation.assist_loop.undo import undo_loop
from src.services.labeling.vlm_endpoints import VlmEndpointUnavailableError
from src.utils.timeutil import now_iso


if TYPE_CHECKING:
    from opensearchpy import AsyncOpenSearch

    from src.services.curation.assist_loop.audit import Auditor
    from src.services.curation.assist_loop.budget import LoopBudget
    from src.services.curation.assist_loop.evaluate import VlmAsker
    from src.services.curation.assist_loop.partition import PartitionScope, RootReader, RootRef
    from src.services.curation.assist_loop.record import LoopRecord

logger = get_logger(__name__)

# What ends a loop as ``failed`` with its message: a refused or unreachable VLM, a
# datastore error, an unreadable blob. Anything else is a bug and propagates.
RUN_ERRORS = (
    RuntimeError,
    ValueError,
    OSError,
    OpenSearchException,
    httpx.HTTPError,
    RequestRejectedError,
    VlmEndpointUnavailableError,
)
MAX_CONCURRENT_SPLITS = 2
"""A split of a node near the refine cap holds hundreds of MB; at most this many at once."""


@dataclass(frozen=True)
class LoopParams:
    seed: int
    theta: float
    alpha: float
    max_dissent: int
    max_depth: int
    branching: int
    min_node_size: int
    node_concurrency: int
    images_per_call: int
    on_budget: str = 'pause'
    """``pause``: a reached budget pauses the loop (raise the limit, then resume);
    ``complete``: the nodes it cannot cover are left unvisited and the loop completes."""

    @property
    def gate(self) -> GateParams:
        return GateParams(theta=self.theta, alpha=self.alpha, max_dissent=self.max_dissent)


@dataclass
class LoopDeps:
    opensearch: AsyncOpenSearch
    reader: RootReader
    asker: VlmAsker
    auditor: Auditor | None = None
    """``None``: the audit is off (no audit endpoint), which the record states."""


@dataclass(frozen=True)
class LoopOutcome:
    status: str
    """``completed``, ``cancelled``, ``paused`` or ``yield``."""
    stop_reason: str | None = None
    pause_reason: str | None = None
    """``None`` for the person's own pause (the route already stored its reason)."""


async def _pin_roots(
    record: LoopRecord, reader: RootReader, scope: PartitionScope
) -> list[RootRef]:
    if any(n.level == 0 for n in record.tree.nodes.values()):
        return await record.root_snapshots()
    partition_id = await reader.pin()
    roots = await reader.snapshots(scope)
    await record.pin_roots(
        roots,
        {
            'kind': PINNED_KIND,
            'partition_id': str(partition_id),
            'digest': membership_digest([f'{r.key}:{r.version}' for r in roots]),
            'snapshot_at': now_iso(),
            'roots': len(roots),
            'eligible_items': sum(len(r.eligible_ids) for r in roots),
        },
    )
    return roots


async def prepare(
    record: LoopRecord,
    params: LoopParams,
    deps: LoopDeps,
    budget: LoopBudget,
    scope: PartitionScope,
    control: RunControl,
) -> LoopRun:
    """Load the run's tree, pin the roots on the first slice and build the run."""
    await record.load()
    roots = await _pin_roots(record, deps.reader, scope)
    evals = EvalContext(
        record=record,
        budget=budget,
        asker=deps.asker,
        gate=params.gate,
        seed=params.seed,
        should_stop=lambda: False,
    )
    run = LoopRun(
        record=record,
        budget=budget,
        params=params,
        deps=deps,
        roots={root.key: root for root in roots},
        states={m.crop_id: m for root in roots for m in root.members},
        split_slots=asyncio.Semaphore(MAX_CONCURRENT_SPLITS),
        evals=evals,
        control=control,
        events=LoopEvents(record.loop_id),
    )
    evals.should_stop = run.halted
    return run


def _pending(run: LoopRun) -> list[tuple[tuple[int, int, tuple[int, ...]], str]]:
    heap: list[tuple[tuple[int, int, tuple[int, ...]], str]] = []

    def visit(path: str, n: int) -> None:
        record = run.tree.nodes.get(path)
        if record is None or not finished(record):
            heapq.heappush(heap, (best_first_key(n, path), path))
            return
        for child in record.children or []:
            visit(child, record.child_n.get(child, 0))

    for key, root in sorted(run.roots.items()):
        visit(key, len(root.eligible_ids))
    return heap


def _halt_reason(run: LoopRun) -> str | None:
    if run.control.cancelled():
        return 'cancelled'
    if run.halt is not None:
        return run.halt
    if run.control.paused():
        return 'user'
    return None


async def write_progress(run: LoopRun, *, close: bool = False) -> None:
    """Report the loop's progress: the per-level rows and totals to the iteration row, crops
    spent of the limit and the node counts to the job."""
    tree = run.tree
    rows = level_rows(tree)
    summary = totals(tree, run.states.values())
    await run.record.save_counts(
        {'levels': rows, 'totals': summary, 'usage_missing': run.budget.usage_missing}, close=close
    )
    await run.events.nodes(tree, force=close)
    await run.events.budget(run.budget)
    await run.events.warnings(tree, run.budget.usage_missing)
    vlm = run.budget.snapshot().get(VLM, {})
    await run.control.progress(
        vlm.get('crops_spent', 0) + vlm.get('crops_reserved', 0),
        vlm.get('max_crops'),
        'iteration:1',
        {
            'nodes_open': sum(1 for n in tree.nodes.values() if n.step != 'done'),
            'nodes_done': sum(1 for n in tree.nodes.values() if n.step == 'done'),
            'items_propagated': summary['items_propagated'],
            'items_direct': summary['items_direct'],
            'left_for_review': summary['items_left_for_review'],
        },
    )


async def _audit(run: LoopRun, *, final: bool) -> None:
    """Audit the propagated nodes no batch covered yet when a batch is due (or, at the
    end, whatever is left); a failing batch pauses the loop."""
    auditor = run.deps.auditor
    if auditor is None or auditor.exhausted(run.tree):
        return
    if not (auditor.pending(run.tree) if final else auditor.due(run.tree)):
        return
    result = await auditor.run_batch(run.tree)
    await run.events.audit(auditor, run.tree)
    if not result.failing:
        return
    if auditor.params.on_fail == 'rollback':
        await undo_loop(
            run.deps.opensearch,
            loop_id=run.record.loop_id,
            links={n.path: n.label_op_id for n in run.tree.nodes.values() if n.label_op_id},
            node_paths=result.nodes,
            dry_run=False,
        )
    run.halt = 'audit'


async def _drain(running: dict[asyncio.Task[list[str]], str]) -> None:
    for task in running:
        task.cancel()
    await asyncio.gather(*running, return_exceptions=True)


def _claim(run: LoopRun, path: str) -> int | None:
    record = run.tree.nodes.get(path)
    return record.claim if record is not None else None


async def _admit(run: LoopRun, path: str, claims: dict[str, int]) -> tuple[str, str | None]:
    """Whether the head of the queue, node ``path``, may start now: ``('start', None)``,
    ``('wait', None)`` (for a running node to free budget) or ``('refuse', reason)`` with
    the budget (``budget_crops``, ``budget_requests`` or ``budget_time``) that cannot
    cover it.

    Nodes running together must not each take a slice of a budget none of them can finish
    on. A node starts only when the crops and requests it may still need fit beside what
    the running nodes may still need: each running node counts at its claim, the *full*
    crops it can ask for (the top-up the allowed dissent calls for included), less what
    it spent; the claim is recorded on the node row. Alone, a node must only afford
    its first draw to start (its top-up, if it dissents, is then a claim like any other).
    Requests are estimated as the crops over ``images_per_call``, rounded up.

    The queue is best-first and strict: when the head must wait, nodes behind it wait too,
    even a smaller one that would fit, so a large node is never starved by small ones.
    That is intended.

    Reproducibility: the decision reads only the budget's counters, the claims of the
    running nodes and the node's own members, so with ``node_concurrency`` 1 (a node
    starts when the previous one is done) the set of visited nodes is the same in every
    run over the same VLM answers. With more, whether the head starts or waits depends on
    which running nodes have finished by then, which is timing: under a binding budget the
    visited set can differ between runs. A node already admitted on its row is not decided
    again: a later slice restarts it on its recorded claim."""
    await run.ensure_members(path)
    minimum = crops_needed(run, path, with_top_up=False)
    full = crops_needed(run, path, with_top_up=True)
    need = full if claims else minimum
    held = run.tree.node(path).crops
    if need == 0:
        await admit(run, path, held)
        claims[path] = held
        return 'start', None
    outstanding = sum(max(0, total - run.tree.node(other).crops) for other, total in claims.items())
    size = max(1, run.deps.asker.images_per_call)
    crops = outstanding + need
    reason = run.budget.would_refuse(VLM, crops=crops, requests=-(-crops // size))
    if reason is None:
        claim = held + full
        await admit(run, path, claim)
        claims[path] = claim
        return 'start', None
    if claims and reason != 'budget_time':
        return 'wait', None
    return 'refuse', reason


def _budget_pause(run: LoopRun, reason: str | None) -> bool:
    """Whether a refusal pauses the loop (``on_budget: pause``) instead of leaving nodes
    unvisited."""
    if reason is not None and reason.startswith('budget_') and run.params.on_budget == 'pause':
        run.halt = 'budget'
        run.stop_reason = run.stop_reason or reason
        return True
    return False


async def _start_ready(
    run: LoopRun,
    heap: list[tuple[tuple[int, int, tuple[int, ...]], str]],
    resumed: list[tuple[tuple[int, int, tuple[int, ...]], str]],
    claims: dict[str, int],
    running: dict[asyncio.Task[list[str]], str],
) -> None:
    """Start nodes while there is room: resumed ones first, then the head of the queue."""
    while (
        _halt_reason(run) is None
        and not run.control.slice_due()
        and (resumed or heap)
        and len(running) < run.params.node_concurrency
    ):
        if resumed:
            _, path = resumed.pop(0)
            running[asyncio.create_task(advance(run, path))] = path
            continue
        _, path = heap[0]
        admission, reason = await _admit(run, path, claims)
        if admission == 'wait':
            return
        if admission == 'refuse' and _budget_pause(run, reason):
            return
        heapq.heappop(heap)
        if admission == 'refuse':
            await leave_unvisited(run, path, reason or 'budget_crops')
            continue
        running[asyncio.create_task(advance(run, path))] = path


async def schedule(run: LoopRun) -> LoopOutcome:
    """Work the queue until it is empty or a stop rule fires."""
    heap = _pending(run)
    # Nodes a previous slice admitted restart first, on the claim their rows recorded; their
    # claims count against every later admission from the start.
    resumed = sorted(entry for entry in heap if _claim(run, entry[1]) is not None)
    heap = [entry for entry in heap if entry not in resumed]
    heapq.heapify(heap)
    claims = {path: claim for _, path in resumed if (claim := _claim(run, path)) is not None}
    running: dict[asyncio.Task[list[str]], str] = {}
    try:
        while True:
            await _start_ready(run, heap, resumed, claims, running)
            if not running:
                break
            done, _ = await asyncio.wait(running, return_when=asyncio.FIRST_COMPLETED)
            for task in sorted(done, key=lambda t: running[t]):
                path = running.pop(task)
                claims.pop(path, None)
                children = task.result()
                sizes = run.tree.node(path).child_n
                for child in children:
                    if not finished(run.tree.nodes.get(child)):
                        heapq.heappush(heap, (best_first_key(sizes.get(child, 0), child), child))
                await _audit(run, final=False)
                await write_progress(run)
        if _halt_reason(run) is None and not (heap or resumed):
            await _audit(run, final=True)
    except BaseException:
        await _drain(running)
        raise
    halt = _halt_reason(run)
    if halt == 'cancelled':
        return LoopOutcome('cancelled')
    if halt == 'user':
        return LoopOutcome('paused')
    if halt is not None:
        return LoopOutcome('paused', pause_reason=halt)
    if heap or resumed:
        return LoopOutcome('yield')
    return LoopOutcome('completed', stop_reason=run.stop_reason or 'done')


async def finish(run: LoopRun, outcome: LoopOutcome) -> None:
    """What a slice leaves in the tables when it stops: the wall time it used, the per-level
    counts on the iteration and, when the run is over, its stop reason."""
    await run.budget.charge_wall()
    done = outcome.status == 'completed'
    await write_progress(run, close=done)
    if done:
        counts = await run.record.with_session(lambda s: iteration_counts(s, run.record.run_id))
        await run.events.iteration(1, 'closed', counts)
    if outcome.status == 'completed' and outcome.stop_reason:
        await run.record.stop_reason(outcome.stop_reason)


__all__ = [
    'RUN_ERRORS',
    'LoopDeps',
    'LoopOutcome',
    'LoopParams',
    'finish',
    'prepare',
    'schedule',
    'write_progress',
]
