"""The full build of one level: every vector, exact kNN, stored edges, groups, a current run.

Image level: exact FAISS search over every vector of the images index (``k`` neighbours, the
lowest cut). Item level: the same search inside each image ``leak`` group of the current image
run only (scope ``image_leak_groups``): two crops are compared only when their frames are
near-duplicates of each other, never two crops of one frame, and only within a detector class
when the profile says so.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.db.repos import near_dup as repo
from src.services.near_duplicates import candidates
from src.services.near_duplicates.grouping import group_edges, with_lineage
from src.services.near_duplicates.runs import Env, RunSummary, write_and_flip
from src.services.near_duplicates.vectors import (
    MemberDoc,
    scan_items_of_images,
    scan_members,
    stack,
)


if TYPE_CHECKING:
    from collections.abc import Callable

    from src.db.repos.near_dup import RunRow


class NoImageRunError(RuntimeError):
    """The item level reads the image level's leak groups; there is no current image run."""


async def _all_members(env: Env) -> dict[str, MemberDoc]:
    found: dict[str, MemberDoc] = {}
    async for page in scan_members(
        env.client, index=env.plan.index, field=env.plan.field, item=env.plan.is_item
    ):
        found.update((m.member_id, m) for m in page)
    return found


def _image_edges(env: Env, members: dict[str, MemberDoc]) -> list[tuple[str, str, float]]:
    ids, mat = stack(members.values())
    return candidates.exact_edges(ids, mat, k=env.plan.k, min_cosine=env.plan.lowest_cut)


async def _item_scope(env: Env) -> tuple[RunRow, list[list[str]]]:
    async with env.scope() as session:
        run = await repo.current_run(session, env.project_id, 'image')
        if run is None:
            msg = 'the item level needs a current image run (its leak groups are the scope)'
            raise NoImageRunError(msg)
        groups = await repo.load_groups(session, run.id, 'leak')
    return run, [list(g.members) for g in groups]


def _pair_allowed(env: Env, a: MemberDoc, b: MemberDoc) -> bool:
    if a.image_id == b.image_id:
        return False
    return not (env.plan.same_detector_class and a.detector_class != b.detector_class)


async def _item_edges(env: Env) -> tuple[dict[str, MemberDoc], list[tuple[str, str, float]]]:
    _, image_groups = await _item_scope(env)
    members: dict[str, MemberDoc] = {}
    edges: list[tuple[str, str, float]] = []
    for image_ids in image_groups:
        pool = await scan_items_of_images(
            env.client, index=env.plan.index, field=env.plan.field, image_ids=image_ids
        )
        ids, mat = stack(pool)
        by_id = {m.member_id: m for m in pool}
        members.update(by_id)
        edges.extend(
            (a, b, s)
            for a, b, s in candidates.exact_edges(
                ids, mat, k=len(ids), min_cosine=env.plan.lowest_cut
            )
            if _pair_allowed(env, by_id[a], by_id[b])
        )
    return members, edges


async def score_full(env: Env) -> tuple[dict[str, MemberDoc], int]:
    """Replace every stored edge of the level with freshly computed ones; returns the members
    read and how many were scored."""
    if env.plan.is_item:
        members, edges = await _item_edges(env)
    else:
        members = await _all_members(env)
        edges = _image_edges(env, members)
    async with env.scope() as session:
        await repo.clear_edges(session, env.project_id, env.plan.level, env.plan.space_key)
        await repo.replace_edges(
            session, env.project_id, env.plan.level, env.plan.space_key, [], edges
        )
    return members, len(members)


async def group_stored(
    env: Env, *, scored: int, known: dict[str, MemberDoc] | None = None
) -> RunSummary:
    """Group the level's stored edges and flip the result current (a ``full`` run)."""
    async with env.scope() as session:
        edges = await repo.load_edges(
            session,
            env.project_id,
            env.plan.level,
            env.plan.space_key,
            min_score=env.plan.lowest_cut,
        )
        decisions = await repo.load_decisions(session, env.project_id, env.plan.level)
        parent = await repo.current_run(session, env.project_id, env.plan.level)
        parent_dedup = [] if parent is None else await repo.load_groups(session, parent.id, 'dedup')
        parent_leak = [] if parent is None else await repo.load_groups(session, parent.id, 'leak')
        fresh = await repo.new_ids(session, 2 * len(edges) + 2)
    grouped = await group_edges(env.client, env.plan, edges, decisions, known)
    split = len(grouped.dedup)
    return await write_and_flip(
        env,
        mode='full',
        parent_run_id=None if parent is None else parent.id,
        dedup=with_lineage(grouped.dedup, parent_dedup, fresh[:split]),
        leak=with_lineage(grouped.leak, parent_leak, fresh[split:]),
        scored=scored,
        oversized=grouped.oversized_split,
    )


async def full_build(
    env: Env, *, cancelled: Callable[[], bool] = lambda: False
) -> RunSummary | None:
    """Score every member, then group and flip; ``None`` when ``cancelled()`` between the two."""
    members, scored = await score_full(env)
    if cancelled():
        return None
    return await group_stored(env, scored=scored, known=members)
