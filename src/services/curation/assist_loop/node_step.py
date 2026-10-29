"""One node's way through the loop: open, evaluate, then propagate, split or leave for review.

Every step is logged before the next begins (``node_open``, the evaluation's ``reserve``
and ``answers``, ``decision``, then ``children`` or ``propagate_begin`` and
``propagated``), so a resumed node continues at the step it was in. A ``pass`` that turns
out to be ``stale`` at write time gets a second ``decision``; the last one counts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from src.domain.loop_tree import child_path, parent_path, parse_path, path_level, root_path
from src.services.curation.assist.scope import membership_digest
from src.services.curation.assist_loop.evaluate import NodeInput, evaluate_node
from src.services.curation.assist_loop.propagate import propagate_node
from src.services.curation.assist_loop.splitter import fetch_embeddings, split_node
from src.services.curation.class_label import ItemLabel
from src.services.curation.class_registry_names import registry_item_class


if TYPE_CHECKING:
    import asyncio

    from src.services.curation.assist.proposal import ClassAnswer
    from src.services.curation.assist_loop.budget import LoopBudget
    from src.services.curation.assist_loop.driver import LoopDeps, LoopParams
    from src.services.curation.assist_loop.evaluate import EvalContext, Verdict
    from src.services.curation.assist_loop.partition import MemberState, RootRef
    from src.services.curation.assist_loop.store import LoopStore, LoopTree, NodeRecord

LOOP_WRITER = 'assist_loop'
"""Prefix of the writer id of every class the loop propagates (``assist_loop:<loop_id>``)."""
_EVENT_KEYS = ('seq', 'at', 'event')


@dataclass
class LoopRun:
    """Everything a node step reads and writes."""

    store: LoopStore
    budget: LoopBudget
    params: LoopParams
    deps: LoopDeps
    tree: LoopTree
    roots: dict[str, RootRef]
    states: dict[str, MemberState]
    split_slots: asyncio.Semaphore
    evals: EvalContext
    stop_reason: str | None = None
    """The first budget that left a node unvisited."""
    halt: str | None = None
    """Set by the audit to stop scheduling (``audit``)."""
    _members: dict[str, list[str]] = field(default_factory=dict)

    def halted(self) -> bool:
        return (
            self.halt is not None or self.store.cancel_requested() or self.store.pause_requested()
        )

    def members(self, path: str) -> list[str]:
        """Every member of the node (a child holds its eligible and ground-truth ones)."""
        if path not in self._members:
            if path in self.roots:
                self._members[path] = self.roots[path].member_ids
            else:
                self._members[path] = self.store.read_node_members(path)
        return self._members[path]

    def eligible(self, ids: list[str]) -> list[str]:
        return sorted(i for i in ids if self.states[i].state == 'eligible')

    def label(self, class_id: int, class_name: str) -> ItemLabel:
        loop_id = self.store.loop_id
        return ItemLabel.propagated(
            class_id=class_id,
            class_name=class_name,
            run_id=loop_id,
            writer=f'{LOOP_WRITER}:{loop_id}',
        )


def finished(record: NodeRecord | None) -> bool:
    if record is None or record.decision is None:
        return False
    kind = record.decision['kind']
    if kind == 'pass':
        return record.propagated is not None
    if kind == 'split':
        return record.children is not None
    return True


def _open(run: LoopRun, path: str) -> NodeRecord:
    ids = run.members(path)
    run.store.append(
        'node_open',
        path=path,
        parent=parent_path(path),
        n=len(run.eligible(ids)),
        digest=membership_digest(ids),
        level=path_level(path),
    )
    return run.tree.nodes[path]


def _view(run: LoopRun, record: NodeRecord) -> NodeInput:
    ids = run.members(record.path)
    eligible = run.eligible(ids)
    truth = [
        name for i in ids if run.states[i].truth and (name := run.states[i].class_name) is not None
    ]
    reused: dict[str, ClassAnswer] = {}
    failed: set[str] = set()
    ancestor = parent_path(record.path)
    while ancestor is not None:
        upper = run.tree.nodes[ancestor]
        for crop_id in eligible:
            if crop_id in upper.answers and crop_id not in reused:
                reused[crop_id] = upper.answers[crop_id]
        failed |= upper.failed
        ancestor = parent_path(ancestor)
    return NodeInput(
        record=record,
        eligible=eligible,
        truth=truth,
        reused=reused,
        known_failed=frozenset((failed & set(eligible)) - set(reused)),
    )


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
        vectors = await fetch_embeddings(run.deps.opensearch, members)
        result = await split_node(
            members,
            vectors,
            branching=run.params.branching,
            seed=run.params.seed,
            path=record.path,
        )
    labelable = [c for c in result.children if run.eligible(c)]
    return (labelable if len(labelable) >= 2 else []), len(result.unplaced)


def _log_children(
    run: LoopRun, record: NodeRecord, children: list[list[str]], unplaced: int
) -> None:
    entries = []
    for index, ids in enumerate(children):
        path = child_path(record.path, index)
        entries.append(
            {
                'path': path,
                'n': len(run.eligible(ids)),
                'file': run.store.write_node_members(path, ids),
            }
        )
    run.store.append('children', path=record.path, children=entries, unplaced=unplaced)


def _decision(run: LoopRun, record: NodeRecord, kind: str, **fields: Any) -> None:
    run.store.append(
        'decision',
        path=record.path,
        kind=kind,
        level=record.level,
        n=record.n,
        wall_s=round(run.budget.wall_s(), 3),
        **fields,
    )


async def _decide(run: LoopRun, record: NodeRecord) -> bool:
    """Evaluate and record the node's decision; ``False`` when it was stopped (left open)."""
    node = _view(run, record)
    evaluation = await evaluate_node(run.evals, node)
    verdict = evaluation.verdict
    common = {'sampled': evaluation.sampled}
    if not node.eligible:
        _decision(run, record, 'review_small', reason='no_eligible_members', **common)
        return True
    if verdict is None:
        if evaluation.unvisited is None:
            return False
        run.stop_reason = run.stop_reason or evaluation.unvisited
        _decision(run, record, 'unvisited', reason=evaluation.unvisited, **common)
        return True
    common.update(_verdict_fields(verdict))
    if verdict.passed:
        _decision(run, record, 'pass', **common)
        return True
    kind = _decide_kind(run, record)
    if kind != 'split':
        _decision(run, record, kind, **common)
        return True
    children, unplaced = await _split_children(run, record)
    if not children:
        _decision(run, record, 'review_leaf', reason='no_split', **common)
        return True
    _decision(run, record, 'split', **common)
    _log_children(run, record, children, unplaced)
    return True


async def _resume_split(run: LoopRun, record: NodeRecord) -> None:
    children, unplaced = await _split_children(run, record)
    if children:
        _log_children(run, record, children, unplaced)
    else:
        _decision(run, record, 'review_leaf', reason='no_split', sampled=0)


def _restate(run: LoopRun, record: NodeRecord, kind: str, **fields: Any) -> None:
    previous = {k: v for k, v in (record.decision or {}).items() if k not in _EVENT_KEYS}
    previous.update(kind=kind, wall_s=round(run.budget.wall_s(), 3), **fields)
    run.store.append('decision', **previous)


async def _propagate(run: LoopRun, record: NodeRecord) -> None:
    found = registry_item_class(str((record.decision or {}).get('class_name') or ''))
    if found is None:
        _restate(run, record, 'stale', reason='class_not_in_registry')
        return
    node = _view(run, record)
    resuming = record.propagate_begun
    if not resuming:
        run.store.append(
            'propagate_begin', path=record.path, class_name=found[1], n_eligible=len(node.eligible)
        )
    result = await propagate_node(
        run.deps.opensearch,
        run.deps.reader,
        root_key=root_path(parse_path(record.path)[0]),
        eligible=node.eligible,
        answers={**node.reused, **record.answers},
        tokens={i: run.states[i].token for i in node.eligible},
        label=run.label(*found),
        resuming=resuming,
    )
    if result.stale:
        _restate(run, record, 'stale', reason='drift', drift=result.drift)
        return
    run.store.append(
        'propagated',
        path=record.path,
        class_name=found[1],
        written=len(result.written_ids),
        written_ids=result.written_ids,
        skipped=result.skipped,
        drift=result.drift,
    )


async def advance(run: LoopRun, path: str) -> list[str]:
    """Take node ``path`` as far as it goes; the children to schedule next."""
    record = run.tree.nodes.get(path) or _open(run, path)
    if record.decision is None and not await _decide(run, record):
        return []
    kind = (record.decision or {}).get('kind')
    if kind == 'pass' and record.propagated is None and not run.halted():
        await _propagate(run, record)
    elif kind == 'split' and record.children is None:
        await _resume_split(run, record)
    return list(record.children or []) if (record.decision or {}).get('kind') == 'split' else []


__all__ = ['LOOP_WRITER', 'LoopRun', 'advance', 'finished']
