"""The loop's pinned partition: what the roots are, who is in them, and the snapshot.

The loop's clusters are the serving partition's (research clustering plan 7.2). A root is one
cluster of the partition the run pinned: named by its stable ``cluster_uid`` and, in node paths and
on the wire, by the ``display_no`` the cluster cards show (``r<display_no>``; the number names a
cluster only within one partition, which is why the run pins the partition id). The partition
and its cluster list are read from PostgreSQL, where a partition is immutable once ready; the
served members and their class state are read from the ``items`` index, which carries the
partition fields the projection writes (``cluster_partition_id``, ``cluster_uid``,
``cluster_state``). Only ``assigned`` members are in a root: a ``review`` member never propagates.

At start every root's members are snapshotted behind the storage seam with the lock-rule state
and class-state token each had: the loop works on that pinned set, never on what the items say
later. ``root_of`` is how it notices an item that has since moved to another cluster or been
re-projected by another partition.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol

from sqlalchemy import text

from src.clients.curation_opensearch.crops import mget_crops
from src.clients.occ import is_locked_class
from src.config.curation_accessors import items_index
from src.domain.loop_tree import root_path
from src.services.clustering.assign import serving_partition_id
from src.services.curation.assist.scope import membership_digest
from src.services.curation.class_write_guard import (
    CLASS_GUARD_SOURCE_FIELDS,
    class_state_token,
    class_write_locked,
)
from src.services.curation.reprocess_targets import scan_items


if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from contextlib import AbstractAsyncContextManager

    from opensearchpy import AsyncOpenSearch
    from sqlalchemy.ext.asyncio import AsyncSession

MemberStateName = Literal['eligible', 'locked', 'holdout', 'excluded']
type SessionFactory = Callable[[], AbstractAsyncContextManager[AsyncSession]]
"""Opens a session scoped to the run's project (the job's ``ctx.session``, or a route's
``project_tx``)."""
ASSIGNED = 'assigned'
PINNED_KIND = 'serving_partition'
_PAGE = 500
_ELIGIBLE: dict[str, Any] = {
    'filter': {
        'bool': {
            'must_not': [
                {'term': {'test_holdout': True}},
                {'term': {'class_excluded': True}},
                {'term': {'class_validated': True}},
                {'wildcard': {'class_source': '*human*'}},
            ]
        }
    }
}
"""The query form of :func:`member_state`'s ``eligible``: not frozen, not shelved, not validated
by anyone and not a person's class (a validated import is ``class_validated``)."""
_CLUSTERS = text(
    'SELECT cluster_uid, size, display_no FROM partition_clusters WHERE partition_id = :p '
    'ORDER BY display_no'
)


class NoServingPartitionError(RuntimeError):
    """The project serves no partition of the target: there are no clusters to work on."""


@dataclass(frozen=True)
class PartitionScope:
    cluster_uids: Sequence[str] | None
    """``None`` = every cluster of the partition."""
    min_root_size: int


@dataclass(frozen=True)
class RootSize:
    """A candidate root counted in the store, without reading its members."""

    cluster_uid: str
    display_no: int
    members: int
    eligible: int
    """Members the loop could ask about and write (the snapshot's ``eligible`` ones)."""

    @property
    def key(self) -> str:
        return root_path(self.display_no)


@dataclass(frozen=True)
class MemberState:
    crop_id: str
    state: MemberStateName
    """``eligible`` may be asked and written; the others never are (a locked member is
    ground truth for the veto)."""
    class_name: str | None
    token: tuple[Any, ...]
    """:func:`~src.services.curation.class_write_guard.class_state_token` at snapshot."""
    truth: bool
    """A class a person or a validated import set: ground truth for the veto. A class
    validated by an automated writer is locked but is not evidence."""


@dataclass(frozen=True)
class RootRef:
    display_no: int
    cluster_uid: str
    version: str
    """Digest of the member ids: equal members, equal version."""
    members: list[MemberState]
    """Every member at snapshot time, sorted by id."""

    @property
    def key(self) -> str:
        return root_path(self.display_no)

    @property
    def member_ids(self) -> list[str]:
        return [m.crop_id for m in self.members]

    @property
    def eligible_ids(self) -> list[str]:
        return [m.crop_id for m in self.members if m.state == 'eligible']


class RootReader(Protocol):
    async def pin(self) -> uuid.UUID:
        """Pin the serving partition for this run; its id."""
        ...

    async def survey(self, scope: PartitionScope) -> list[RootSize]:
        """The roots of ``scope`` with their sizes (what a plan shows); reads no members."""
        ...

    async def snapshots(self, scope: PartitionScope) -> list[RootRef]: ...

    async def root_of(self, ids: Sequence[str]) -> dict[str, str]:
        """Root key each id belongs to now; an id in no root is absent."""
        ...


@dataclass(frozen=True)
class ClusterRoot:
    """One cluster of a partition, as the loop sees a root."""

    partition_id: uuid.UUID
    cluster_uid: uuid.UUID
    size: int
    display_no: int


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
        truth=state == 'locked' and is_locked_class(source),
    )


class PartitionReader:
    """The loop's reader of the serving partition.

    A reader is bound to one project and target and, once :meth:`pin` has run (or a resumed
    slice built it with :meth:`for_run`), to one partition: every later read is of that
    partition, whatever the project serves by then.
    """

    def __init__(
        self,
        opensearch: AsyncOpenSearch,
        sessions: SessionFactory,
        *,
        project: uuid.UUID,
        target: str = 'item',
        partition_id: uuid.UUID | None = None,
    ) -> None:
        self._os = opensearch
        self._sessions = sessions
        self._project = project
        self._target = target
        self.partition_id = partition_id
        self._display: dict[str, int] | None = None

    @classmethod
    def for_run(
        cls,
        opensearch: AsyncOpenSearch,
        sessions: SessionFactory,
        *,
        project: uuid.UUID,
        pinned: Mapping[str, Any],
        target: str = 'item',
    ) -> PartitionReader:
        """The reader of a run: bound to the partition its first slice pinned, if it has run."""
        found = (pinned.get('partition') or {}).get('partition_id')
        return cls(
            opensearch,
            sessions,
            project=project,
            target=target,
            partition_id=None if found is None else uuid.UUID(str(found)),
        )

    @staticmethod
    async def roots(
        session: AsyncSession,
        *,
        project: uuid.UUID,
        target: str,
        partition_id: uuid.UUID | None = None,
    ) -> list[ClusterRoot]:
        """The clusters of ``partition_id`` (default: the serving partition of ``target``, class
        level), in display order."""
        if partition_id is None:
            partition_id = await serving_partition_id(session, project=project, target=target)
        if partition_id is None:
            return []
        rows = await session.execute(_CLUSTERS, {'p': partition_id})
        return [ClusterRoot(partition_id, uid, size, display_no) for uid, size, display_no in rows]

    async def pin(self) -> uuid.UUID:
        async with self._sessions() as session:
            found = await serving_partition_id(session, project=self._project, target=self._target)
        if found is None:
            msg = f'the project serves no {self._target} partition: run clustering first'
            raise NoServingPartitionError(msg)
        self.partition_id = found
        self._display = None
        return found

    def _pinned(self) -> uuid.UUID:
        if self.partition_id is None:
            msg = 'the reader has no pinned partition'
            raise RuntimeError(msg)
        return self.partition_id

    async def _display_nos(self) -> dict[str, int]:
        """Display number of each cluster uid of the pinned partition (immutable once ready,
        so read once)."""
        if self._display is None:
            partition_id = self._pinned()
            async with self._sessions() as session:
                clusters = await self.roots(
                    session, project=self._project, target=self._target, partition_id=partition_id
                )
            self._display = {str(c.cluster_uid): c.display_no for c in clusters}
        return self._display

    def _served(self, *extra: dict[str, Any]) -> dict[str, Any]:
        """Items the pinned partition serves as assigned members of a cluster."""
        return {
            'bool': {
                'filter': [
                    {'term': {'cluster_partition_id': str(self._pinned())}},
                    {'term': {'cluster_state': ASSIGNED}},
                    *extra,
                ]
            }
        }

    async def survey(self, scope: PartitionScope) -> list[RootSize]:
        display = await self._display_nos()
        wanted = (
            []
            if scope.cluster_uids is None
            else [{'terms': {'cluster_uid': [*scope.cluster_uids]}}]
        )
        sizes: list[RootSize] = []
        after: dict[str, Any] | None = None
        while True:
            composite: dict[str, Any] = {
                'size': _PAGE,
                'sources': [{'cluster_uid': {'terms': {'field': 'cluster_uid'}}}],
            }
            if after is not None:
                composite['after'] = after
            resp = await self._os.search(
                index=items_index(),
                body={
                    'size': 0,
                    'query': self._served(*wanted),
                    'aggs': {'roots': {'composite': composite, 'aggs': {'eligible': _ELIGIBLE}}},
                },
            )
            agg = resp['aggregations']['roots']
            buckets = agg.get('buckets') or []
            sizes.extend(
                RootSize(
                    uid,
                    display[uid],
                    int(b['doc_count']),
                    int(b['eligible']['doc_count']),
                )
                for b in buckets
                if (uid := str(b['key']['cluster_uid'])) in display
            )
            after = agg.get('after_key')
            if not buckets or after is None:
                kept = [s for s in sizes if s.members >= scope.min_root_size]
                return sorted(kept, key=lambda s: s.display_no)

    async def _snapshot(self, size: RootSize) -> RootRef:
        found = await scan_items(
            self._os,
            self._served({'term': {'cluster_uid': size.cluster_uid}}),
            index=items_index(),
            includes=[*CLASS_GUARD_SOURCE_FIELDS, 'test_holdout'],
        )
        members = sorted((member_state(cid, src) for cid, src in found), key=lambda m: m.crop_id)
        return RootRef(
            display_no=size.display_no,
            cluster_uid=size.cluster_uid,
            version=membership_digest([m.crop_id for m in members]),
            members=members,
        )

    async def snapshots(self, scope: PartitionScope) -> list[RootRef]:
        return [await self._snapshot(size) for size in await self.survey(scope)]

    async def root_of(self, ids: Sequence[str]) -> dict[str, str]:
        partition_id = str(self._pinned())
        display = await self._display_nos()
        docs = await mget_crops(
            self._os,
            list(ids),
            source_includes=['cluster_partition_id', 'cluster_uid', 'cluster_state'],
        )
        out: dict[str, str] = {}
        for crop_id, doc in docs.items():
            src = doc.get('_source') or {}
            number = display.get(str(src.get('cluster_uid')))
            if (
                src.get('cluster_partition_id') == partition_id
                and src.get('cluster_state') == ASSIGNED
                and number is not None
            ):
                out[crop_id] = root_path(number)
        return out


__all__ = [
    'ASSIGNED',
    'PINNED_KIND',
    'ClusterRoot',
    'MemberState',
    'NoServingPartitionError',
    'PartitionReader',
    'PartitionScope',
    'RootReader',
    'RootRef',
    'RootSize',
    'SessionFactory',
    'member_state',
]
