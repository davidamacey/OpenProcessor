"""Groups from edges: components per purpose, then each group's representative and cosines.

The pure rules are in ``src/domain/near_dup.py``; this module adds the vectors they need. A
member with an edge but no vector any more is a stale edge: the run refuses (fail closed) rather
than group a member nobody can describe.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from src.db.repos.near_dup import GroupToWrite
from src.domain import near_dup as rules
from src.domain.group_lineage import match_groups
from src.services.near_duplicates.vectors import MemberDoc, fetch_members


if TYPE_CHECKING:
    from collections.abc import Mapping

    from opensearchpy import AsyncOpenSearch

    from src.db.repos.near_dup import StoredGroup
    from src.services.near_duplicates.plan import LevelPlan


class StaleEdgeError(RuntimeError):
    """An edge names a member that has no vector in the index."""


@dataclass(frozen=True)
class Grouped:
    """The groups of one run: member sets per purpose, with the size-guard split count."""

    dedup: list[rules.Group]
    leak: list[rules.Group]
    oversized_split: int


def _purpose_edges(
    edges: list[tuple[str, str, float]],
    cut: float,
    decisions: Mapping[tuple[str, str], str],
) -> dict[tuple[str, str], float]:
    return rules.effective_edges(edges, cut, decisions)


async def group_edges(
    client: AsyncOpenSearch,
    plan: LevelPlan,
    edges: list[tuple[str, str, float]],
    decisions: Mapping[tuple[str, str], str],
    known: Mapping[str, MemberDoc] | None = None,
) -> Grouped:
    """Components at the dedup and the leak cut; ``known`` spares a re-read of vectors in hand."""
    per_purpose = {'dedup': _purpose_edges(edges, plan.dedup_cosine, decisions)}
    if plan.has_leak:
        per_purpose['leak'] = _purpose_edges(edges, plan.leak_cosine, decisions)
    components = {p: rules.group_members(e, plan.max_group_size) for p, e in per_purpose.items()}
    wanted = {m for c in components.values() for g in c.groups for m in g}
    docs = dict(known or {})
    missing = wanted - docs.keys()
    if missing:
        docs.update(
            await fetch_members(
                client,
                index=plan.index,
                field=plan.field,
                ids=missing,
                item=plan.is_item,
            )
        )
    lost = wanted - docs.keys()
    if lost:
        msg = f'{len(lost)} grouped members have no vector (first: {sorted(lost)[0]})'
        raise StaleEdgeError(msg)
    described = {
        p: [
            rules.describe_group(g, np.stack([docs[m].vector for m in g]), per_purpose[p])
            for g in c.groups
        ]
        for p, c in components.items()
    }
    return Grouped(
        described['dedup'],
        described.get('leak', []),
        max(c.oversized_split for c in components.values()),
    )


def with_lineage(
    groups: list[rules.Group],
    parent: list[StoredGroup],
    fresh: list[uuid.UUID],
) -> list[GroupToWrite]:
    """Rows to write: a group overlapping a parent group (Jaccard >= 0.5) keeps its lineage id,
    the others take one of the ``fresh`` ids (at least ``len(groups)`` of them)."""
    new = {str(i): g.members for i, g in enumerate(groups)}
    old = {str(i): g.members for i, g in enumerate(parent)}
    matched = match_groups(old, new).by_new
    out: list[GroupToWrite] = []
    for i, group in enumerate(groups):
        inherited = matched[str(i)].inherited
        carried_id = str(parent[int(inherited)].lineage_id) if inherited is not None else None
        out.append(
            GroupToWrite(
                lineage_id=uuid.UUID(carried_id) if carried_id else fresh[i],
                representative_id=group.representative,
                members=group.members,
                cosine_to_rep=group.cosine_to_rep,
                min_edge_cosine=group.min_edge_cosine,
                min_cosine_to_rep=group.min_cosine_to_rep,
            )
        )
    return out


def copied(group: StoredGroup) -> GroupToWrite:
    """An untouched parent group, carried into an incremental run with its lineage."""
    return GroupToWrite(
        lineage_id=group.lineage_id,
        representative_id=group.representative_id,
        members=group.members,
        cosine_to_rep=group.cosine_to_rep,
        min_edge_cosine=group.min_edge_cosine,
        min_cosine_to_rep=group.min_cosine_to_rep,
    )
