"""What the loop routes serve: the stored state as the wire model, the nodes of the tree the
event log describes, and the member ids of one node."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.domain.loop_tree import parse_path, path_level
from src.schemas.curation.assist_loop import (
    AssistLoopState,
    LoopMembersPage,
    LoopNewClassVote,
    LoopNode,
    LoopNodePage,
    LoopVote,
)
from src.services.curation.assist_loop import job
from src.services.curation.assist_loop.partition import read_snapshot
from src.services.curation.assist_loop.store import LoopStore, NodeRecord, replay
from src.services.rejections import rejection


if TYPE_CHECKING:
    from collections.abc import Callable


def loop_state(state: dict[str, Any]) -> AssistLoopState:
    return AssistLoopState.model_validate(state)


def open_store(loop_id: str) -> LoopStore:
    """The store of an existing loop (404 otherwise)."""
    if job.read_state(loop_id) is None:
        raise rejection(404, 'not_found', f'no auto-label loop {loop_id}')
    return LoopStore(job.open_job(loop_id).directory)


def _order(path: str) -> tuple[int, int, tuple[int, ...]]:
    cluster, indices = parse_path(path)
    return (path_level(path), cluster, indices)


def _node(record: NodeRecord) -> LoopNode:
    decision = record.decision or {}
    return LoopNode(
        path=record.path,
        level=record.level,
        parent=record.parent,
        n=record.n,
        sampled=int(decision.get('sampled', len(record.answers) + len(record.failed))),
        answered=int(decision.get('answered', len(record.answers))),
        decision=decision.get('kind'),
        reason=decision.get('reason'),
        class_name=decision.get('class_name'),
        agree=decision.get('agree'),
        bound=decision.get('bound'),
        distribution=[LoopVote(class_name=c, votes=v) for c, v in decision.get('distribution', [])],
        new_class_proposals=[
            LoopNewClassVote(name=c, votes=v) for c, v in decision.get('new_class_proposals', [])
        ],
        items_propagated=int((record.propagated or {}).get('written', 0)),
    )


def _page[T](
    items: list[T], key: Callable[[T], Any], after: Any, limit: int
) -> tuple[list[T], T | None]:
    """``items`` (in ``key`` order) after the cursor, at most ``limit``, and the last one
    returned when more remain."""
    rest = [x for x in items if after is None or key(x) > after]
    page = rest[:limit]
    return page, (page[-1] if len(rest) > limit else None)


def node_page(
    store: LoopStore, *, level: int | None, decision: str | None, after: str | None, limit: int
) -> LoopNodePage:
    nodes = [
        _node(r)
        for r in replay(store.events()).nodes.values()
        if (level is None or r.level == level)
        and (decision is None or (r.decision or {}).get('kind') == decision)
    ]
    nodes.sort(key=lambda n: _order(n.path))
    try:
        cursor = None if after is None else _order(after)
    except ValueError:
        raise rejection(422, 'invalid_request', f'{after!r} is not a node path') from None
    page, last = _page(nodes, lambda n: _order(n.path), cursor, limit)
    return LoopNodePage(nodes=page, next_after=last.path if last else None)


def member_page(store: LoopStore, path: str, *, after: str | None, limit: int) -> LoopMembersPage:
    if path not in replay(store.events()).nodes:
        raise rejection(404, 'not_found', f'loop {store.loop_id} has no node {path}')
    try:
        _, indices = parse_path(path)
        ids = (
            read_snapshot(store.directory, path).member_ids
            if not indices
            else store.read_node_members(path)
        )
    except ValueError as exc:
        raise rejection(404, 'not_found', str(exc)) from exc
    page, last = _page(sorted(ids), lambda i: i, after, limit)
    return LoopMembersPage(crop_ids=page, next_after=last)


__all__ = ['loop_state', 'member_page', 'node_page', 'open_store']
