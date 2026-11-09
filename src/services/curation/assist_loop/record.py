"""The loop's durable record: every step the driver takes is committed to the ``assist_loop_*``
tables through here before the next step starts.

A slice builds a :class:`LoopRecord` from the run row, loads the tree from the tables
(:meth:`LoopRecord.load`) and goes on from it. Each method is one short transaction on the
job's own engine (``ctx.session``) and carries the job's fencing token, so a slice that lost
its lease writes nothing and raises ``LostFenceError``. The in-memory tree is updated only
after the rows are committed, so it never runs ahead of the database.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

from src.db.repos import assist_loop_record as rows
from src.domain.loop_tree import child_path
from src.services.curation.assist_loop.tree import (
    ASK_PURPOSES,
    LoopTree,
    NodeRecord,
    build_tree,
    decision_columns,
)


if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from contextlib import AbstractAsyncContextManager

    from sqlalchemy.ext.asyncio import AsyncSession
    from sqlalchemy.orm import Session

    from src.services.curation.assist.proposal import ClassAnswer
    from src.services.curation.assist_loop.partition import RootRef
    from src.services.curation.assist_loop.snapshots import SnapshotStore

ITERATION = 1
"""Only one iteration exists until the review loop (v0.8.0) opens more."""


class LoopRecord:
    """One run's rows, for the slice of the job that owns them."""

    def __init__(
        self,
        sessions: Callable[[], AbstractAsyncContextManager[AsyncSession]],
        run: rows.RunRow,
        *,
        token: int,
        snapshots: SnapshotStore,
    ) -> None:
        self.sessions = sessions
        self.run = run
        self.run_id = run.id
        self.job_id = run.job_id
        self.token = token
        self.snapshots = snapshots
        self.tree = LoopTree()

    @property
    def loop_id(self) -> str:
        return str(self.run_id)

    async def _do[T](self, fn: Callable[[Session], T]) -> T:
        async with self.sessions() as session:
            return await session.run_sync(fn)

    async def _fenced[T](self, fn: Callable[[Session, uuid.UUID, int], T]) -> T:
        return await self._do(lambda s: fn(s, self.job_id, self.token))

    # ------------------------------------------------------------------ load

    async def load(self) -> LoopTree:
        """Rebuild the tree from the tables: the open nodes, their answers and the pending asks."""

        def read(session: Session) -> LoopTree:
            counts = rows.iteration_counts(session, self.run_id)
            return build_tree(
                rows.nodes(session, self.run_id),
                rows.selections(session, self.run_id),
                rows.proposals(session, self.run_id, role='primary'),
                counts=counts,
            )

        self.tree = await self._do(read)
        return self.tree

    # ------------------------------------------------------------------ roots

    async def pin_roots(self, roots: Sequence[RootRef], partition: Mapping[str, Any]) -> None:
        """Snapshot every root behind the storage seam and open it as a node."""
        new: list[rows.NewNode] = []
        for root in roots:
            key = await self.snapshots.write_root(root)
            new.append(
                rows.NewNode(
                    path=root.key,
                    parent_path=None,
                    level=0,
                    member_count=len(root.members),
                    eligible_count=len(root.eligible_ids),
                    membership_digest=root.version,
                    snapshot_storage_key=key,
                )
            )
        pinned = {**self.run.pinned, 'partition': dict(partition)}

        def write(session: Session, job: uuid.UUID, token: int) -> None:
            rows.insert_nodes(
                session, self.run_id, job_id=job, token=token, iteration=ITERATION, nodes=new
            )
            rows.set_pinned(session, self.run_id, job_id=job, token=token, pinned=pinned)

        await self._fenced(write)
        await self.load()

    async def root_snapshots(self) -> list[RootRef]:
        keys = [n.snapshot_key for n in self.tree.nodes.values() if n.level == 0]
        return [await self.snapshots.read_root(key) for key in sorted(keys)]

    # ------------------------------------------------------------------ one node

    async def admit(self, node: NodeRecord, claim: int, requests: int) -> None:
        if node.step == 'open':

            def write(session: Session, job: uuid.UUID, token: int) -> None:
                rows.update_node(
                    session,
                    self.run_id,
                    node.path,
                    job_id=job,
                    token=token,
                    step='admitted',
                    admitted_crops=claim,
                    admitted_requests=requests,
                )

            await self._fenced(write)
            node.step = 'admitted'
        node.claim = claim

    async def select(self, node: NodeRecord, purpose: str, ids: Sequence[str]) -> None:
        """Put ``ids`` on record as asked at ``node`` before the request is sent."""
        if purpose not in ASK_PURPOSES:
            raise ValueError(f'not an ask purpose: {purpose}')
        known = len(node.answers) + len(node.failed) + len(node.pending)
        new = [
            rows.SelectionRow(i, purpose, node.path, ITERATION, known + rank)
            for rank, i in enumerate(ids)
            if i not in node.answers and i not in node.failed and i not in node.pending
        ]
        if not new:
            return

        def write(session: Session, job: uuid.UUID, token: int) -> None:
            rows.insert_selections(session, self.run_id, job_id=job, token=token, rows=new)

        await self._fenced(write)
        node.pending.extend(s.item_id for s in new)
        node.crops += len(new)

    async def record_answers(
        self,
        node: NodeRecord,
        request_id: uuid.UUID,
        chunk: Sequence[str],
        answers: Mapping[str, ClassAnswer],
        *,
        request_failed: frozenset[str] | set[str] = frozenset(),
        proposer_ref: str = '',
    ) -> None:
        """The outcome of one request: each id of the chunk is answered, unreadable (the VLM
        replied with nothing usable) or failed (the call itself failed)."""
        batch: list[rows.ProposalRow] = []
        for item in chunk:
            answer = answers.get(item)
            if answer is not None:
                named = answer.class_name or None
                batch.append(
                    rows.ProposalRow(
                        item,
                        'primary',
                        request_id,
                        'answered',
                        named,
                        None if named else (answer.proposed or None),
                        proposer_ref,
                    )
                )
            else:
                outcome = 'failed' if item in request_failed else 'unreadable'
                batch.append(
                    rows.ProposalRow(item, 'primary', request_id, outcome, None, None, proposer_ref)
                )

        def write(session: Session, job: uuid.UUID, token: int) -> None:
            rows.insert_proposals(session, self.run_id, job_id=job, token=token, rows=batch)

        await self._fenced(write)
        for row in batch:
            if row.item_id in node.pending:
                node.pending.remove(row.item_id)
            if row.outcome == 'answered':
                node.answers[row.item_id] = answers[row.item_id]
            else:
                node.failed.add(row.item_id)

    async def forget_failed(self, node: NodeRecord, ids: Sequence[str]) -> None:
        """An outage, not a verdict: the failed asks are forgotten so a resume asks them again."""

        def write(session: Session, job: uuid.UUID, token: int) -> None:
            rows.delete_failed_proposals(session, self.run_id, job_id=job, token=token, ids=ids)

        await self._fenced(write)
        for item in ids:
            if item in node.failed:
                node.failed.discard(item)
                node.pending.append(item)

    # ------------------------------------------------------------------ decisions

    async def _set(self, node: NodeRecord, **columns: Any) -> None:
        def write(session: Session, job: uuid.UUID, token: int) -> None:
            rows.update_node(session, self.run_id, node.path, job_id=job, token=token, **columns)

        await self._fenced(write)

    async def decide(self, node: NodeRecord, kind: str, **fields: Any) -> None:
        """A decision that ends the node: review_leaf, review_small, unvisited, stale."""
        columns = decision_columns(fields)
        await self._set(node, step='done', decision=kind, **columns)
        node.step = 'done'
        node.decision = {'kind': kind, **fields}

    async def gate_pass(self, node: NodeRecord, **fields: Any) -> None:
        """The gate passed: the node waits for its propagation (step ``gated``)."""
        await self._set(node, step='gated', **decision_columns(fields))
        node.step = 'gated'
        node.decision = {'kind': 'pass', **fields}

    async def begin_propagate(self, node: NodeRecord) -> None:
        await self._set(node, step='propagating')
        node.step = 'propagating'
        node.propagate_begun = True

    async def finish_propagate(
        self,
        node: NodeRecord,
        *,
        class_name: str,
        written_ids: Sequence[str],
        skipped: Mapping[str, int],
        drift: float,
        label_op_id: str | None,
    ) -> None:
        await self.snapshots.write_ids('written', node.path, written_ids)
        stored = {**skipped, 'drift': drift}
        await self._set(
            node,
            step='done',
            decision='pass',
            items_propagated=len(written_ids),
            skipped=stored,
            label_op_id=None if label_op_id is None else uuid.UUID(label_op_id),
        )
        node.step = 'done'
        node.propagated = {
            'written': len(written_ids),
            'class_name': class_name,
            'drift': drift,
            'skipped': dict(skipped),
        }
        node.label_op_id = label_op_id

    async def written_ids(self, node: NodeRecord) -> list[str]:
        if node.propagated is None or not node.propagated['written']:
            return []
        return await self.snapshots.read_ids(self.snapshots.key('written', node.path))

    async def restate(self, node: NodeRecord, kind: str, **fields: Any) -> None:
        """A passed node that turned out stale at write time: it ends with the new decision."""
        previous = {k: v for k, v in (node.decision or {}).items() if k != 'kind'}
        await self.decide(node, kind, **{**previous, **fields})

    async def split(
        self,
        node: NodeRecord,
        children: Sequence[list[str]],
        eligible: Callable[[list[str]], Sequence[str]],
        digest: Callable[[list[str]], str],
        **fields: Any,
    ) -> None:
        """The node's ``split`` decision and its children, in one transaction."""
        new: list[rows.NewNode] = []
        for index, ids in enumerate(children):
            path = child_path(node.path, index)
            key = await self.snapshots.write_ids('nodes', path, ids)
            new.append(
                rows.NewNode(
                    path=path,
                    parent_path=node.path,
                    level=node.level + 1,
                    member_count=len(ids),
                    eligible_count=len(eligible(ids)),
                    membership_digest=digest(ids),
                    snapshot_storage_key=key,
                )
            )
        columns = decision_columns(fields)

        def write(session: Session, job: uuid.UUID, token: int) -> None:
            rows.split_node(
                session,
                self.run_id,
                node.path,
                job_id=job,
                token=token,
                iteration=ITERATION,
                decision_columns=columns,
                children=new,
            )

        await self._fenced(write)
        node.step = 'done'
        node.decision = {'kind': 'split', **fields}
        node.children = [n.path for n in new]
        node.child_n = {n.path: n.eligible_count for n in new}
        for n in new:
            self.tree.nodes[n.path] = NodeRecord(
                path=n.path,
                parent=node.path,
                level=n.level,
                n=n.eligible_count,
                member_count=n.member_count,
                digest=n.membership_digest,
                snapshot_key=n.snapshot_storage_key,
            )

    async def record_audit_chunk(
        self,
        request_id: uuid.UUID,
        chunk: Sequence[tuple[str, str]],
        answers: Mapping[str, str | None],
    ) -> None:
        """One audit call: each ``(item, node)`` is selected for the audit, and the judge's
        class is its ``audit`` proposal (``unreadable`` when it named none)."""
        selected = [
            rows.SelectionRow(item, 'audit', node, ITERATION, rank)
            for rank, (item, node) in enumerate(chunk)
        ]
        proposed = [
            rows.ProposalRow(
                item,
                'audit',
                request_id,
                'answered' if answers.get(item) else 'unreadable',
                answers.get(item),
                None,
            )
            for item, _ in chunk
        ]

        def write(session: Session, job: uuid.UUID, token: int) -> None:
            rows.insert_selections(session, self.run_id, job_id=job, token=token, rows=selected)
            rows.insert_proposals(session, self.run_id, job_id=job, token=token, rows=proposed)

        await self._fenced(write)

    async def save_counts(self, counts: Mapping[str, Any], *, close: bool = False) -> None:
        """Merge ``counts`` (the per-level rows, the totals) into the iteration row, which the
        status routes read; ``close`` ends the iteration."""

        def write(session: Session, job: uuid.UUID, token: int) -> None:
            merged = {**rows.iteration_counts(session, self.run_id), **counts}
            rows.set_iteration_counts(
                session, self.run_id, job_id=job, token=token, counts=merged, close=close
            )

        await self._fenced(write)

    async def stop_reason(self, reason: str) -> None:
        def write(session: Session, job: uuid.UUID, token: int) -> None:
            rows.set_stop_reason(session, self.run_id, job_id=job, token=token, reason=reason)

        await self._fenced(write)

    async def set_audit(self, audit: Mapping[str, Any]) -> None:
        def write(session: Session, job: uuid.UUID, token: int) -> None:
            counts = rows.iteration_counts(session, self.run_id)
            counts['audit'] = dict(audit)
            rows.set_iteration_counts(session, self.run_id, job_id=job, token=token, counts=counts)

        await self._fenced(write)
        self.tree.audit = dict(audit)

    async def with_session[T](self, fn: Callable[[Session], T]) -> T:
        """A read or an unfenced-by-design write in the job's project scope."""
        return await self._do(fn)


__all__ = ['ITERATION', 'LoopRecord']
