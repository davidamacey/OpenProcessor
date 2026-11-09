"""One node's way through the loop: evaluate, then propagate, split or leave for review.

Every step is committed to the node's row before the next begins (the asks and answers,
``gated`` for a pass, ``propagating`` once the write starts, ``done`` with the decision), so a
resumed node continues at the step it was in. A ``pass`` that turns out to be ``stale`` at
write time ends with that decision instead.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from src.domain.loop_stats import required_sample
from src.domain.loop_tree import parse_path, root_path
from src.services.curation.assist.scope import membership_digest
from src.services.curation.assist_loop.evaluate import NodeInput, evaluate_node
from src.services.curation.assist_loop.propagate import propagate_node
from src.services.curation.assist_loop.splitter import fetch_embeddings, split_node
from src.services.curation.class_label import ItemLabel
from src.services.curation.class_registry_names import registry_item_class


if TYPE_CHECKING:
    import asyncio
    from collections.abc import Awaitable, Callable

    from src.services.curation.assist.proposal import ClassAnswer
    from src.services.curation.assist_loop.budget import LoopBudget
    from src.services.curation.assist_loop.driver import LoopDeps, LoopParams
    from src.services.curation.assist_loop.emit import LoopEvents
    from src.services.curation.assist_loop.evaluate import EvalContext, Verdict
    from src.services.curation.assist_loop.partition import MemberState, RootRef
    from src.services.curation.assist_loop.record import LoopRecord
    from src.services.curation.assist_loop.tree import LoopTree, NodeRecord

LOOP_WRITER = 'assist_loop'
"""Prefix of the writer id of every class the loop propagates (``assist_loop:<loop_id>``)."""


@dataclass
class LoopRun:
    """Everything a node step reads and writes."""

    record: LoopRecord
    budget: LoopBudget
    params: LoopParams
    deps: LoopDeps
    roots: dict[str, RootRef]
    states: dict[str, MemberState]
    split_slots: asyncio.Semaphore
    evals: EvalContext
    control: RunControl
    events: LoopEvents
    stop_reason: str | None = None
    """The first budget that left a node unvisited."""
    halt: str | None = None
    """Set by the audit to stop scheduling (``audit``)."""
    _members: dict[str, list[str]] = field(default_factory=dict)

    @property
    def tree(self) -> LoopTree:
        return self.record.tree

    def halted(self) -> bool:
        return self.halt is not None or self.control.stopping()

    async def ensure_members(self, path: str) -> None:
        if path in self._members:
            return
        if path in self.roots:
            self._members[path] = self.roots[path].member_ids
        else:
            self._members[path] = await self.record.snapshots.read_ids(
                self.tree.node(path).snapshot_key
            )

    def members(self, path: str) -> list[str]:
        """Every member of the node (a child holds its eligible and ground-truth ones); the
        node's blob must have been read with :meth:`ensure_members`."""
        return self._members[path]

    def eligible(self, ids: list[str]) -> list[str]:
        return sorted(i for i in ids if self.states[i].state == 'eligible')

    def label(self, class_id: int, class_name: str) -> ItemLabel:
        loop_id = self.record.loop_id
        return ItemLabel.propagated(
            class_id=class_id,
            class_name=class_name,
            run_id=loop_id,
            writer=f'{LOOP_WRITER}:{loop_id}',
        )


@dataclass(frozen=True)
class RunControl:
    """What the job tells the loop: stop for a cancel or a pause, or hand over after a slice."""

    cancelled: Callable[[], bool]
    paused: Callable[[], bool]
    slice_due: Callable[[], bool]
    progress: Callable[[int, int | None, str, dict[str, Any]], Awaitable[None]]

    def stopping(self) -> bool:
        return self.cancelled() or self.paused()


def finished(record: NodeRecord | None) -> bool:
    if record is None or record.decision is None:
        return False
    kind = record.decision['kind']
    if kind == 'pass':
        return record.propagated is not None
    return True


def _inherited(
    run: LoopRun, path: str, eligible: list[str]
) -> tuple[dict[str, ClassAnswer], set[str]]:
    """Answers ancestors of ``path`` got for its eligible members, and the members they
    asked without a usable answer."""
    reused: dict[str, ClassAnswer] = {}
    failed: set[str] = set()
    ancestor = run.tree.node(path).parent
    while ancestor is not None:
        upper = run.tree.node(ancestor)
        for crop_id in eligible:
            if crop_id in upper.answers and crop_id not in reused:
                reused[crop_id] = upper.answers[crop_id]
        failed |= upper.failed
        ancestor = upper.parent
    return reused, failed


def _view(run: LoopRun, record: NodeRecord) -> NodeInput:
    ids = run.members(record.path)
    eligible = run.eligible(ids)
    truth = [
        name for i in ids if run.states[i].truth and (name := run.states[i].class_name) is not None
    ]
    reused, failed = _inherited(run, record.path, eligible)
    return NodeInput(
        record=record,
        eligible=eligible,
        truth=truth,
        reused=reused,
        known_failed=frozenset((failed & set(eligible)) - set(reused)),
    )


def crops_needed(run: LoopRun, path: str, *, with_top_up: bool) -> int:
    """New VLM crops node ``path`` still needs to reach a verdict: the first draw the gate
    needs with no dissent, or (``with_top_up``) the larger one the allowed dissent needs.
    What ancestors already answered and what the node already holds is not counted."""
    eligible = run.eligible(run.members(path))
    if not eligible:
        return 0
    gate = run.params.gate
    dissent = gate.max_dissent if with_top_up else 0
    need = required_sample(len(eligible), gate.theta, gate.alpha_stage, dissent)
    record = run.tree.node(path)
    held = set(record.answers) | record.failed
    reused, _ = _inherited(run, path, eligible)
    return max(0, need - len(held | set(reused)))


async def admit(run: LoopRun, path: str, claim: int) -> None:
    """Record that node ``path`` was admitted against ``claim`` crops in total."""
    requests = math.ceil(claim / max(1, run.params.images_per_call))
    await run.record.admit(run.tree.node(path), claim, requests)


async def leave_unvisited(run: LoopRun, path: str, reason: str) -> None:
    """Record that node ``path`` was not worked on because ``reason`` (a budget) cannot
    cover it, before any crop was spent on it."""
    record = run.tree.node(path)
    run.stop_reason = run.stop_reason or reason
    await run.record.decide(record, 'unvisited', reason=reason, sampled=0)


def _verdict_fields(verdict: Verdict) -> dict[str, Any]:
    return {
        'class_name': verdict.class_name,
        'agree': verdict.agree,
        'answered': verdict.answered,
        'dissent': verdict.dissent,
        'need': verdict.need,
        'bound': verdict.bound,
        'vetoed': verdict.vetoed,
        'distribution': [list(pair) for pair in verdict.distribution],
        'new_class_proposals': [list(pair) for pair in verdict.new_class_proposals],
    }


def _decide_kind(run: LoopRun, record: NodeRecord) -> str:
    """Where a node that did not pass goes."""
    if record.level >= run.params.max_depth:
        return 'review_leaf'
    if record.n < run.params.min_node_size:
        return 'review_small'
    return 'split'


async def _split_children(run: LoopRun, record: NodeRecord) -> tuple[list[list[str]], int]:
    """The node's children by embedding (each with at least one eligible member) and the
    number of members left unplaced. Fewer than two such children is no split."""
    ids = run.members(record.path)
    members = sorted(i for i in ids if run.states[i].state == 'eligible' or run.states[i].truth)
    async with run.split_slots:
        vectors = await fetch_embeddings(
            run.deps.opensearch, members, field=run.deps.embedding_field
        )
        result = await split_node(
            members,
            vectors,
            branching=run.params.branching,
            seed=run.params.seed,
            path=record.path,
        )
    labelable = [c for c in result.children if run.eligible(c)]
    return (labelable if len(labelable) >= 2 else []), len(result.unplaced)


async def _decide(run: LoopRun, record: NodeRecord) -> bool:
    """Evaluate and record the node's decision; ``False`` when it was stopped (left open)."""
    node = _view(run, record)
    evaluation = await evaluate_node(run.evals, node)
    verdict = evaluation.verdict
    common: dict[str, Any] = {'sampled': evaluation.sampled}
    rec = run.record
    if not node.eligible:
        await rec.decide(record, 'review_small', reason='no_eligible_members', **common)
        return True
    if verdict is None:
        if evaluation.unvisited is None:
            return False
        run.stop_reason = run.stop_reason or evaluation.unvisited
        await rec.decide(record, 'unvisited', reason=evaluation.unvisited, **common)
        return True
    common.update(_verdict_fields(verdict))
    if verdict.passed:
        await rec.gate_pass(record, **common)
        return True
    kind = _decide_kind(run, record)
    if kind != 'split':
        await rec.decide(record, kind, **common)
        return True
    children, unplaced = await _split_children(run, record)
    if not children:
        await rec.decide(record, 'review_leaf', reason='no_split', **common)
        return True
    await rec.split(
        record,
        children,
        run.eligible,
        membership_digest,
        unplaced=unplaced,
        **common,
    )
    return True


async def _propagate(run: LoopRun, record: NodeRecord) -> None:
    rec = run.record
    found = await registry_item_class(str((record.decision or {}).get('class_name') or ''))
    if found is None:
        await rec.restate(record, 'stale', reason='class_not_in_registry')
        return
    node = _view(run, record)
    resuming = record.propagate_begun
    if not resuming:
        await rec.begin_propagate(record)
    result = await propagate_node(
        run.deps.opensearch,
        run.deps.reader,
        root_key=root_path(parse_path(record.path)[0]),
        eligible=node.eligible,
        answers={**node.reused, **record.answers},
        tokens={i: run.states[i].token for i in node.eligible},
        label=run.label(*found),
        job_id=rec.job_id,
        resuming=resuming,
    )
    if result.stale:
        await rec.restate(record, 'stale', reason='drift', drift=result.drift)
        return
    await rec.finish_propagate(
        record,
        class_name=found[1],
        written_ids=result.written_ids,
        skipped=result.skipped,
        drift=result.drift,
        label_op_id=result.label_op_id,
    )


async def advance(run: LoopRun, path: str) -> list[str]:
    """Take node ``path`` as far as it goes; the children to schedule next."""
    await run.ensure_members(path)
    record = run.tree.node(path)
    if record.step in ('open', 'admitted') and not await _decide(run, record):
        return []
    kind = (record.decision or {}).get('kind')
    if kind == 'pass' and record.propagated is None and not run.halted():
        await _propagate(run, record)
    return list(record.children or []) if kind == 'split' else []


__all__ = [
    'LOOP_WRITER',
    'LoopRun',
    'RunControl',
    'admit',
    'advance',
    'crops_needed',
    'finished',
    'leave_unvisited',
]
