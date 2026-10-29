"""Run an auto-label loop: pin the partition, schedule nodes best-first, stop on the rules.

On its first run the loop snapshots every root of the scope into the job directory (the
pinned partition); a resumed loop reads those snapshots and replays its event log, so it
continues every node where it was. Open nodes wait in a priority queue keyed by
(-eligible members, path): the budget goes first to the nodes that label the most items
per VLM call, in an order fixed for a given tree. Up to ``node_concurrency`` nodes are
worked on at once; their VLM requests share one semaphore.

The loop stops when the queue is empty (``completed``, ``stop_reason`` ``done``, or the
budget that left nodes unvisited), on ``cancel.flag`` (``cancelled``) or ``pause.flag``
(``paused``, ``pause_reason`` ``user``). Requests in flight finish and their answers are
kept; nodes they belong to stay open for a resume. ``state.json`` carries the per-level
record and the budget after every node step.
"""

from __future__ import annotations

import asyncio
import contextlib
import heapq
from dataclasses import dataclass
from typing import TYPE_CHECKING

import httpx
from opensearchpy.exceptions import OpenSearchException

from src.core.exceptions import RequestRejectedError
from src.core.logging import get_logger
from src.domain.loop_tree import best_first_key
from src.services.curation.assist.scope import membership_digest
from src.services.curation.assist_loop.evaluate import EvalContext, GateParams
from src.services.curation.assist_loop.node_step import LoopRun, advance, finished
from src.services.curation.assist_loop.partition import read_snapshot, write_snapshot
from src.services.curation.assist_loop.progress import level_rows, totals
from src.services.curation.assist_loop.store import replay
from src.services.curation.file_job import heartbeat_ticker
from src.services.labeling.vlm_endpoints import VlmEndpointUnavailableError
from src.utils.timeutil import now_iso


if TYPE_CHECKING:
    from opensearchpy import AsyncOpenSearch

    from src.services.curation.assist_loop.budget import LoopBudget
    from src.services.curation.assist_loop.evaluate import VlmAsker
    from src.services.curation.assist_loop.partition import PartitionReader, PartitionScope, RootRef
    from src.services.curation.assist_loop.store import LoopStore

logger = get_logger(__name__)

# What ends a loop as ``failed`` with its message: a refused or unreachable VLM, a
# datastore error, an unreadable job file. Anything else is a bug and propagates.
_RUN_ERRORS = (
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

    @property
    def gate(self) -> GateParams:
        return GateParams(theta=self.theta, alpha=self.alpha, max_dissent=self.max_dissent)


@dataclass
class LoopDeps:
    opensearch: AsyncOpenSearch
    reader: PartitionReader
    asker: VlmAsker


@dataclass(frozen=True)
class LoopOutcome:
    status: str
    """``completed``, ``cancelled``, ``paused`` or ``failed``."""
    stop_reason: str | None = None
    pause_reason: str | None = None
    error: str | None = None


async def _pin_roots(
    store: LoopStore, reader: PartitionReader, scope: PartitionScope
) -> list[RootRef]:
    lines = store.read_roots()
    if lines is not None:
        return [read_snapshot(store.directory, line['root']) for line in lines]
    roots = await reader.roots(scope)
    lines = [write_snapshot(store.directory, root) for root in roots]
    store.write_roots(lines)
    store.job.update(
        partition={
            'kind': 'cluster_field',
            'digest': membership_digest([f'{x["root"]}:{x["digest"]}' for x in lines]),
            'snapshot_at': now_iso(),
            'roots': len(roots),
            'eligible_items': sum(len(r.eligible_ids) for r in roots),
        }
    )
    return roots


async def _prepare(
    store: LoopStore,
    params: LoopParams,
    deps: LoopDeps,
    budget: LoopBudget,
    scope: PartitionScope,
) -> LoopRun:
    roots = await _pin_roots(store, deps.reader, scope)
    tree = replay(store.events())
    store.listeners.append(tree.apply)
    evals = EvalContext(
        store=store,
        budget=budget,
        asker=deps.asker,
        gate=params.gate,
        seed=params.seed,
        should_stop=lambda: False,
    )
    run = LoopRun(
        store=store,
        budget=budget,
        params=params,
        deps=deps,
        tree=tree,
        roots={root.key: root for root in roots},
        states={m.crop_id: m for root in roots for m in root.members},
        split_slots=asyncio.Semaphore(MAX_CONCURRENT_SPLITS),
        evals=evals,
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
    if run.store.cancel_requested():
        return 'cancelled'
    if run.halt is not None:
        return run.halt
    if run.store.pause_requested():
        return 'user'
    return None


def write_progress(run: LoopRun, **fields: object) -> None:
    """Rewrite the loop's record in ``state.json`` (and touch the heartbeat)."""
    run.store.job.update(
        levels=level_rows(run.tree),
        totals=totals(run.tree, run.states.values()),
        budget={**run.budget.snapshot(), 'wall_s': round(run.budget.wall_s(), 3)},
        updated_at=now_iso(),
        **fields,
    )
    run.store.job.touch_heartbeat()


async def _drain(running: dict[asyncio.Task[list[str]], str]) -> None:
    for task in running:
        task.cancel()
    await asyncio.gather(*running, return_exceptions=True)


async def schedule(run: LoopRun) -> LoopOutcome:
    """Work the queue until it is empty or a stop rule fires."""
    heap = _pending(run)
    running: dict[asyncio.Task[list[str]], str] = {}
    try:
        while True:
            while _halt_reason(run) is None and heap and len(running) < run.params.node_concurrency:
                _, path = heapq.heappop(heap)
                running[asyncio.create_task(advance(run, path))] = path
            if not running:
                break
            done, _ = await asyncio.wait(running, return_when=asyncio.FIRST_COMPLETED)
            for task in sorted(done, key=lambda t: running[t]):
                path = running.pop(task)
                children = task.result()
                sizes = run.tree.nodes[path].child_n
                for child in children:
                    if not finished(run.tree.nodes.get(child)):
                        heapq.heappush(heap, (best_first_key(sizes.get(child, 0), child), child))
                write_progress(run)
    except BaseException:
        await _drain(running)
        raise
    halt = _halt_reason(run)
    if halt == 'cancelled':
        return LoopOutcome('cancelled')
    if halt is not None:
        return LoopOutcome('paused', pause_reason=halt)
    return LoopOutcome('completed', stop_reason=run.stop_reason or 'done')


def _final_fields(outcome: LoopOutcome) -> dict[str, object]:
    fields: dict[str, object] = {
        'status': outcome.status,
        'stop_reason': outcome.stop_reason,
        'pause_reason': outcome.pause_reason,
        'error': outcome.error,
    }
    if outcome.status in ('completed', 'cancelled', 'failed'):
        fields['finished_at'] = now_iso()
    return fields


async def run_loop(
    store: LoopStore,
    params: LoopParams,
    deps: LoopDeps,
    budget: LoopBudget,
    scope: PartitionScope,
) -> LoopOutcome:
    """Run (or resume) the loop in ``store`` to a stop; the outcome is also in its state."""
    store.job.update(
        status='running',
        started_at=store.job.read().get('started_at') or now_iso(),
        stop_reason=None,
        pause_reason=None,
        error=None,
        finished_at=None,
    )
    ticker = asyncio.create_task(heartbeat_ticker(store.job))
    try:
        run = await _prepare(store, params, deps, budget, scope)
        outcome = await schedule(run)
        write_progress(run, **_final_fields(outcome))
    except _RUN_ERRORS as exc:
        logger.error('assist_loop_failed', loop=store.loop_id, error=str(exc))
        outcome = LoopOutcome('failed', error=str(exc))
        store.job.update(**_final_fields(outcome), updated_at=now_iso())
    finally:
        ticker.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await ticker
    return outcome


__all__ = ['LoopDeps', 'LoopOutcome', 'LoopParams', 'run_loop', 'schedule', 'write_progress']
