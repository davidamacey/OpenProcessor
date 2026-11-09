"""The incremental path: claim pending members, re-score them, regroup only what they touch.

Components are closed under the edges: the members of the claimed members' current groups, their
new neighbours, and (by repeating the edge read until nothing new appears) every member joined to
them. Groups untouched by that set are copied into the new run unchanged (same lineage), so an
incremental run equals a full build on the same state for the components it touched.

A pending member whose vector is gone is not an error when its reason is ``deleted``; for any
other reason it is counted, its edges are dropped, and its pending row stays so the next tick
retries it until its row is older than ``MISSING_MAX_AGE``: then the row is dropped and the id
is reported in ``RescoreOutcome.expired`` (a terminal "missing", not a retry every tick forever).
The run is written only when every claimed row was handled or kept.

The core level (no project, no build step) starts from the pending rows: with no current run the
first pass is a ``full`` run over the claimed members and what they join; a project level
refuses to start incrementally.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from src.db.repos import near_dup as repo, near_dup_pending as pending_repo
from src.domain import near_dup as rules
from src.services.near_duplicates import candidates
from src.services.near_duplicates.grouping import copied, group_edges, with_lineage
from src.services.near_duplicates.runs import Env, RunSummary, write_and_flip
from src.services.near_duplicates.vectors import MemberDoc, fetch_members


if TYPE_CHECKING:
    from collections.abc import Mapping

CLAIM_BATCH = 5000
MISSING_MAX_AGE = dt.timedelta(hours=24)


class NoCurrentRunError(RuntimeError):
    """An incremental run needs a parent: the level must be built in full first."""


@dataclass(frozen=True)
class RescoreOutcome:
    summary: RunSummary | None
    claimed: int
    missing_vectors: int
    regrouped: frozenset[str] = frozenset()
    """The members whose groups were recomputed (the claimed ones and everything joined to them)."""
    expired: tuple[str, ...] = ()
    """Pending members dropped for staying without a vector past ``MISSING_MAX_AGE``."""


async def _leak_partitions(env: Env) -> dict[str, list[str]]:
    """Item level: each image mapped to the images of its current ``leak`` group."""
    async with env.scope() as session:
        run = await repo.current_run(session, env.project_id, 'image')
        groups = [] if run is None else await repo.load_groups(session, run.id, 'leak')
    return {image: list(g.members) for g in groups for image in g.members}


async def _new_edges(env: Env, members: Mapping[str, MemberDoc]) -> list[tuple[str, str, float]]:
    plan = env.plan
    vectors = {m: d.vector for m, d in members.items()}
    if not plan.is_item:
        return await candidates.knn_edges(
            env.client,
            index=plan.index,
            field=plan.field,
            members=vectors,
            k=plan.k,
            min_cosine=plan.lowest_cut,
        )
    partitions = await _leak_partitions(env)
    scoped = {m: v for m, v in vectors.items() if members[m].image_id in partitions}

    def narrowed(member: str) -> dict[str, Any]:
        return {'terms': {'image_id': partitions[members[member].image_id or '']}}

    def accept(member: str, hit: dict[str, Any]) -> bool:
        mine = members[member]
        if hit.get('image_id') == mine.image_id:
            return False
        return not (
            plan.same_detector_class and hit.get('detector_class_name') != mine.detector_class
        )

    return await candidates.knn_edges(
        env.client,
        index=plan.index,
        field=plan.field,
        members=scoped,
        k=plan.k,
        min_cosine=plan.lowest_cut,
        extra_source=('image_id', 'detector_class_name'),
        filter_for=narrowed,
        accept=accept,
    )


async def _closure(
    env: Env, seeds: set[str], decisions: Mapping[tuple[str, str], str]
) -> tuple[set[str], list[tuple[str, str, float]]]:
    """The members reachable from ``seeds`` over stored edges and ``same`` decisions."""
    reached = set(seeds)
    frontier = set(seeds)
    edges: dict[tuple[str, str], float] = {}
    while frontier:
        async with env.scope() as session:
            found = await repo.load_edges(
                session,
                env.project_id,
                env.plan.level,
                env.plan.space_key,
                min_score=env.plan.lowest_cut,
                touching=sorted(frontier),
            )
        edges.update(((a, b), s) for a, b, s in found)
        step = {n for a, b, _ in found for n in (a, b)}
        for (a, b), decision in decisions.items():
            if decision == 'same' and (a in frontier or b in frontier):
                step.update((a, b))
        frontier = step - reached
        reached |= frontier
    return reached, [(a, b, s) for (a, b), s in edges.items()]


async def _affected(
    env: Env,
    claimed: list[str],
    parent_groups: tuple[list[repo.StoredGroup], list[repo.StoredGroup]],
    decisions: Mapping[tuple[str, str], str],
) -> tuple[set[str], list[tuple[str, str, float]]]:
    """The members to regroup and every stored edge among them: claimed members, whole parent
    groups they touch, and everything joined to either, repeated until nothing new joins."""
    affected = set(claimed)
    while True:
        affected, edges = await _closure(env, affected, decisions)
        grown = set(affected)
        for groups in parent_groups:
            for group in _touching(groups, affected):
                grown.update(group.members)
        if grown == affected:
            return affected, edges
        affected = grown


def _touching(groups: list[repo.StoredGroup], nodes: set[str]) -> list[repo.StoredGroup]:
    return [g for g in groups if nodes.intersection(g.members)]


async def rescore(env: Env) -> RescoreOutcome:
    """One incremental pass over up to ``CLAIM_BATCH`` pending members of the level."""
    assert env.job_id is not None  # a rescore is always a job
    async with env.scope() as session:
        parent = await repo.current_run(session, env.project_id, env.plan.level)
        if parent is None and env.plan.level != rules.CORE:
            msg = f'the {env.plan.level} level has no current run; build it in full first'
            raise NoCurrentRunError(msg)
        claimed = await pending_repo.claim_pending(
            session, env.project_id, env.plan.level, env.job_id, CLAIM_BATCH
        )
    if not claimed:
        return RescoreOutcome(None, 0, 0)
    claimed_ids = sorted(c.member_id for c in claimed)
    live = [c.member_id for c in claimed if c.reason != 'deleted']
    members = await fetch_members(
        env.client, index=env.plan.index, field=env.plan.field, ids=live, item=env.plan.is_item
    )
    gone = sorted(set(live) - members.keys())
    edges = await _new_edges(env, members)
    async with env.scope() as session:
        await repo.replace_edges(
            session, env.project_id, env.plan.level, env.plan.space_key, claimed_ids, edges
        )
        decisions = await repo.load_decisions(session, env.project_id, env.plan.level)
        parent_dedup = [] if parent is None else await repo.load_groups(session, parent.id, 'dedup')
        parent_leak = [] if parent is None else await repo.load_groups(session, parent.id, 'leak')
    affected, scoped_edges = await _affected(
        env, claimed_ids, (parent_dedup, parent_leak), decisions
    )
    grouped = await group_edges(
        env.client,
        env.plan,
        scoped_edges,
        {p: d for p, d in decisions.items() if p[0] in affected or p[1] in affected},
        members,
    )
    split = len(grouped.dedup)
    async with env.scope() as session:
        fresh = await repo.new_ids(session, 2 * len(scoped_edges) + 2)
    dedup = with_lineage(grouped.dedup, _touching(parent_dedup, affected), fresh[:split])
    leak = with_lineage(grouped.leak, _touching(parent_leak, affected), fresh[split:])
    dedup += [copied(g) for g in parent_dedup if not affected.intersection(g.members)]
    leak += [copied(g) for g in parent_leak if not affected.intersection(g.members)]
    summary = await write_and_flip(
        env,
        mode='full' if parent is None else 'incremental',
        parent_run_id=None if parent is None else parent.id,
        dedup=dedup,
        leak=leak,
        scored=len(members),
        oversized=grouped.oversized_split,
    )
    async with env.scope() as session:
        await pending_repo.drain_claimed(
            session, env.job_id, [m for m in claimed_ids if m not in set(gone)]
        )
        expired = await pending_repo.expire_claimed(session, env.job_id, gone, MISSING_MAX_AGE)
    return RescoreOutcome(summary, len(claimed_ids), len(gone), frozenset(affected), tuple(expired))
