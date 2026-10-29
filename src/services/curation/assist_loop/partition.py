"""The loop's pinned partition: what the roots are, who is in them, and the snapshot.

The partition is read through :class:`PartitionReader`; :class:`ClusterFieldPartition` is
the only implementation and the only reader of ``cluster_id`` in the loop. A versioned
partition (stable cluster identities) implements the same two calls, so the driver, the
gate and the writer do not change.

At start every root's members are snapshotted into the job directory with the lock-rule
state and class-state token each had: the loop works on that pinned set, never on what the
cluster field says later. ``root_of`` is how it notices an item that has since moved.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol

from src.clients.curation_opensearch.crops import mget_crops
from src.config.curation_accessors import items_index
from src.domain.loop_tree import parse_path, root_path
from src.services.curation.assist.scope import membership_digest
from src.services.curation.class_write_guard import (
    CLASS_GUARD_SOURCE_FIELDS,
    class_state_token,
    class_write_locked,
)
from src.services.curation.cluster_ids import RESIDUAL_CLUSTER_ID_OFFSET
from src.services.curation.reprocess_targets import scan_items


if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from opensearchpy import AsyncOpenSearch

MemberStateName = Literal['eligible', 'locked', 'holdout', 'excluded']
_PAGE = 500
_SNAPSHOT_DIR = 'members'


@dataclass(frozen=True)
class PartitionScope:
    cluster_ids: Sequence[int] | None
    """``None`` = every candidate cluster."""
    min_root_size: int


@dataclass(frozen=True)
class MemberState:
    crop_id: str
    state: MemberStateName
    """``eligible`` may be asked and written; the others never are (a locked member is
    ground truth for the veto)."""
    class_name: str | None
    token: tuple[Any, ...]
    """:func:`~src.services.curation.class_write_guard.class_state_token` at snapshot."""


@dataclass(frozen=True)
class RootRef:
    key: str
    cluster_id: int
    version: str
    """Digest of the member ids: equal members, equal version."""
    members: list[MemberState]
    """Every member at snapshot time, sorted by id."""

    @property
    def member_ids(self) -> list[str]:
        return [m.crop_id for m in self.members]

    @property
    def eligible_ids(self) -> list[str]:
        return [m.crop_id for m in self.members if m.state == 'eligible']


class PartitionReader(Protocol):
    async def roots(self, scope: PartitionScope) -> list[RootRef]: ...

    async def root_of(self, ids: Sequence[str]) -> dict[str, str]:
        """Root key each id belongs to now; an id in no root is absent."""
        ...


def member_state(crop_id: str, source: dict[str, Any]) -> MemberState:
    state: MemberStateName
    if source.get('test_holdout'):
        state = 'holdout'
    elif source.get('class_excluded'):
        state = 'excluded'
    elif class_write_locked(source):
        state = 'locked'
    else:
        state = 'eligible'
    return MemberState(
        crop_id=crop_id,
        state=state,
        class_name=source.get('class_name') or None,
        token=class_state_token(source),
    )


class ClusterFieldPartition:
    """Roots are clusters named by the stored ``cluster_id``."""

    def __init__(self, opensearch: AsyncOpenSearch) -> None:
        self._os = opensearch

    async def _sizes(self, scope: PartitionScope) -> list[tuple[int, int]]:
        if scope.cluster_ids is None:
            query: dict[str, Any] = {'range': {'cluster_id': {'gte': RESIDUAL_CLUSTER_ID_OFFSET}}}
        else:
            query = {'terms': {'cluster_id': list(scope.cluster_ids)}}
        sizes: list[tuple[int, int]] = []
        after: dict[str, Any] | None = None
        while True:
            composite: dict[str, Any] = {
                'size': _PAGE,
                'sources': [{'cluster_id': {'terms': {'field': 'cluster_id'}}}],
            }
            if after is not None:
                composite['after'] = after
            resp = await self._os.search(
                index=items_index(),
                body={'size': 0, 'query': query, 'aggs': {'roots': {'composite': composite}}},
            )
            agg = resp['aggregations']['roots']
            buckets = agg.get('buckets') or []
            sizes.extend((int(b['key']['cluster_id']), int(b['doc_count'])) for b in buckets)
            after = agg.get('after_key')
            if not buckets or after is None:
                return [(cid, n) for cid, n in sizes if n >= scope.min_root_size]

    async def _snapshot(self, cluster_id: int) -> RootRef:
        found = await scan_items(
            self._os,
            {'term': {'cluster_id': cluster_id}},
            index=items_index(),
            includes=[*CLASS_GUARD_SOURCE_FIELDS, 'test_holdout'],
        )
        members = sorted((member_state(cid, src) for cid, src in found), key=lambda m: m.crop_id)
        return RootRef(
            key=root_path(cluster_id),
            cluster_id=cluster_id,
            version=membership_digest([m.crop_id for m in members]),
            members=members,
        )

    async def roots(self, scope: PartitionScope) -> list[RootRef]:
        return [await self._snapshot(cid) for cid, _ in await self._sizes(scope)]

    async def root_of(self, ids: Sequence[str]) -> dict[str, str]:
        docs = await mget_crops(self._os, list(ids), source_includes=['cluster_id'])
        out: dict[str, str] = {}
        for crop_id, doc in docs.items():
            cluster_id = (doc.get('_source') or {}).get('cluster_id')
            if isinstance(cluster_id, int) and not isinstance(cluster_id, bool) and cluster_id >= 0:
                out[crop_id] = root_path(cluster_id)
        return out


def _snapshot_file(job_dir: Path, key: str) -> Path:
    _, indices = parse_path(key)
    if indices:
        msg = f'not a root path: {key!r}'
        raise ValueError(msg)
    return job_dir / _SNAPSHOT_DIR / f'{key}.json'


def snapshot_line(root: RootRef) -> dict[str, Any]:
    """The ``roots.jsonl`` entry that points at a root's snapshot file."""
    return {
        'root': root.key,
        'cluster_id': root.cluster_id,
        'n': len(root.members),
        'digest': root.version,
        'members_file': f'{_SNAPSHOT_DIR}/{root.key}.json',
    }


def write_snapshot(job_dir: Path, root: RootRef) -> dict[str, Any]:
    """Persist ``root`` under ``job_dir`` (atomic rename: a snapshot is whole or absent)
    and return its ``roots.jsonl`` entry."""
    target = _snapshot_file(job_dir, root.key)
    target.parent.mkdir(parents=True, exist_ok=True)
    body = {
        'root': root.key,
        'cluster_id': root.cluster_id,
        'version': root.version,
        'members': [
            {'id': m.crop_id, 'state': m.state, 'class_name': m.class_name, 'token': list(m.token)}
            for m in root.members
        ],
    }
    tmp = target.with_suffix('.tmp')
    tmp.write_text(json.dumps(body), encoding='utf-8')
    tmp.replace(target)
    return snapshot_line(root)


def read_snapshot(job_dir: Path, key: str) -> RootRef:
    """The snapshot of root ``key``; one that is missing or unreadable is an error, never
    an empty root."""
    target = _snapshot_file(job_dir, key)
    try:
        raw = json.loads(target.read_text(encoding='utf-8'))
        return RootRef(
            key=raw['root'],
            cluster_id=int(raw['cluster_id']),
            version=raw['version'],
            members=[
                MemberState(m['id'], m['state'], m['class_name'], tuple(m['token']))
                for m in raw['members']
            ],
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        msg = f'snapshot of {key} is missing or unreadable: {exc}'
        raise ValueError(msg) from exc


__all__ = [
    'ClusterFieldPartition',
    'MemberState',
    'PartitionReader',
    'PartitionScope',
    'RootRef',
    'member_state',
    'read_snapshot',
    'snapshot_line',
    'write_snapshot',
]
