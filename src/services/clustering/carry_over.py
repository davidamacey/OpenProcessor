"""Cluster identity carry-over and anchor pinning for one run (research clustering plan 5.1).

A new run's clusters are matched to the parent partition's by member Jaccard with Hungarian
assignment (``src.domain.group_lineage``); a matched cluster keeps its ``cluster_uid``. Anchored
identities (a person curated them) are pinned afterwards: their members stay in them, and the number
of members the pin had to override is reported as ``anchored_moves`` so the run is kept out of
auto-activation (plan J3).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from sqlalchemy import text

from src.domain.group_lineage import match_groups
from src.services.clustering.partitions import ClusterLineage, mint_uids


if TYPE_CHECKING:
    import uuid
    from collections.abc import Collection, Mapping, Sequence

    from sqlalchemy.ext.asyncio import AsyncSession


@dataclass(frozen=True)
class CarryOver:
    uid_of_label: dict[int, uuid.UUID]
    """The identity of every fitted cluster (label >= 0)."""
    lineage: dict[uuid.UUID, ClusterLineage]
    pinned: dict[str, uuid.UUID] = field(default_factory=dict)
    """Members the anchor pin moved back into their anchored identity (item key to uid)."""
    stability: float = 1.0
    """Share of the parent's clusters whose identity a new cluster inherited."""
    counts: dict[str, int] = field(default_factory=dict)
    """Parent clusters by what became of them, plus ``new`` and ``dropped``."""


async def parent_memberships(
    session: AsyncSession, parent_partition_id: uuid.UUID
) -> dict[uuid.UUID, set[str]]:
    rows = await session.execute(
        text(
            'SELECT cluster_uid, item_key FROM cluster_memberships '
            'WHERE partition_id = :p AND cluster_uid IS NOT NULL'
        ),
        {'p': parent_partition_id},
    )
    out: dict[uuid.UUID, set[str]] = defaultdict(set)
    for uid, key in rows:
        out[uid].add(key)
    return out


async def anchored_uids(session: AsyncSession, uids: Collection[uuid.UUID]) -> set[uuid.UUID]:
    if not uids:
        return set()
    rows = await session.execute(
        text(
            'SELECT cluster_uid FROM cluster_identities '
            'WHERE cluster_uid = ANY(:uids) AND anchored_at IS NOT NULL'
        ),
        {'uids': list(uids)},
    )
    return {row[0] for row in rows}


def _group_by_label(keys: Sequence[str], labels: Sequence[int]) -> dict[int, list[str]]:
    groups: dict[int, list[str]] = defaultdict(list)
    for key, label in zip(keys, labels, strict=True):
        if label >= 0:
            groups[int(label)].append(key)
    return groups


async def carry_over(
    session: AsyncSession,
    *,
    keys: Sequence[str],
    labels: Sequence[int],
    parent_partition_id: uuid.UUID | None,
) -> CarryOver:
    """Identities for the fitted clusters (``labels`` >= 0; -1 is unassigned)."""
    new_groups = _group_by_label(keys, labels)
    old_groups: Mapping[uuid.UUID, set[str]] = (
        await parent_memberships(session, parent_partition_id) if parent_partition_id else {}
    )
    result = match_groups(old_groups, new_groups)
    fresh = [label for label, lin in result.by_new.items() if lin.inherited is None]
    minted = iter(await mint_uids(session, len(fresh)))
    uid_of_label: dict[int, uuid.UUID] = {}
    lineage: dict[uuid.UUID, ClusterLineage] = {}
    for label, lin in result.by_new.items():
        inherited = lin.inherited
        uid = inherited if inherited is not None else next(minted)
        uid_of_label[int(label)] = uid
        lineage[uid] = ClusterLineage(lin.transition, tuple(lin.predecessors))
    inheriting = {lin.inherited for lin in result.by_new.values() if lin.inherited is not None}
    counts = {
        'parent_clusters': len(old_groups),
        'inherited': len(inheriting),
        'new': sum(1 for lin in result.by_new.values() if lin.transition == 'new'),
        'split': sum(1 for lin in result.by_new.values() if lin.transition == 'split'),
        'merged': sum(1 for lin in result.by_new.values() if lin.transition == 'merged'),
        'dropped': len(result.dropped),
    }
    stability = len(inheriting) / len(old_groups) if old_groups else 1.0
    pinned = await _pin_anchors(session, keys, labels, uid_of_label, lineage, old_groups)
    return CarryOver(uid_of_label, lineage, pinned, stability, counts)


async def _pin_anchors(
    session: AsyncSession,
    keys: Sequence[str],
    labels: Sequence[int],
    uid_of_label: dict[int, uuid.UUID],
    lineage: dict[uuid.UUID, ClusterLineage],
    old_groups: Mapping[uuid.UUID, set[str]],
) -> dict[str, uuid.UUID]:
    """Members of an anchored identity that the fit put elsewhere, back into their identity."""
    anchored = await anchored_uids(session, old_groups)
    if not anchored:
        return {}
    present = {key: int(label) for key, label in zip(keys, labels, strict=True)}
    pinned: dict[str, uuid.UUID] = {}
    for uid in sorted(anchored):
        for key in sorted(old_groups[uid]):
            label = present.get(key)
            if label is None:
                continue
            if label < 0 or uid_of_label.get(label) != uid:
                pinned[key] = uid
        if any(u == uid for u in pinned.values()):
            lineage.setdefault(uid, ClusterLineage('survived', (uid,)))
    return pinned
