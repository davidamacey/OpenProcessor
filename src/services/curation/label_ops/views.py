"""What the list route serves: each record with the number of items that still
carry its label, in the wire shape."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.config.curation_accessors import items_index
from src.domain.label_ops import LABEL_OP_ID_RE, current_query
from src.services.curation.label_ops.store import (
    LABEL_OPS_KEEP,
    LabelOpRecord,
    ServedStatus,
    list_ops,
    served_status,
)
from src.services.rejections import rejection


if TYPE_CHECKING:
    from collections.abc import Iterable

_COUNT_CHUNK = 50
"""Operations counted per ``filters`` aggregation."""


async def current_counts(opensearch: Any, records: Iterable[LabelOpRecord]) -> dict[str, int]:
    """Items still carrying each operation's label: one ``filters`` aggregation per chunk."""
    recs = list(records)
    counts: dict[str, int] = {}
    for start in range(0, len(recs), _COUNT_CHUNK):
        chunk = recs[start : start + _COUNT_CHUNK]
        resp = await opensearch.search(
            index=items_index(),
            body={
                'size': 0,
                'track_total_hits': False,
                'aggs': {
                    'ops': {
                        'filters': {
                            'filters': {r.op_id: current_query(r.fingerprint) for r in chunk}
                        }
                    }
                },
            },
        )
        buckets = resp['aggregations']['ops']['buckets']
        counts.update({r.op_id: int(buckets[r.op_id]['doc_count']) for r in chunk})
    return counts


def label_op_wire(record: LabelOpRecord, current: int, status: ServedStatus) -> dict[str, Any]:
    """The one storage-to-wire mapping of an operation (:class:`LabelOp`); ``status`` is
    :func:`~src.services.curation.label_ops.store.served_status`."""
    return {
        'op_id': record.op_id,
        'kind': record.kind,
        'at': record.at,
        'finished_at': record.finished_at,
        'class_name': record.class_name,
        'target_cluster_id': record.target_cluster_id,
        'target_kind': record.target_kind,
        'label_source': record.label_source,
        'target': record.target,
        'selected': record.selected,
        'count': record.count,
        'conflicts': record.conflicts,
        'current': current,
        'status': status,
        'undone': bool(record.undo_passes),
        'undoable': status not in ('writing', 'undoing') and current > 0,
        'last_undo': record.undo_passes[-1] if record.undo_passes else None,
    }


async def list_label_ops(
    opensearch: Any, *, limit: int, undoable_only: bool, after: str | None
) -> dict[str, Any]:
    """Newest first by ``op_id``, older than ``after`` (exclusive). With
    ``undoable_only`` the next ``limit`` undoable operations. ``next_after`` is
    the page's last ``op_id`` only when a further operation exists."""
    if after is not None and not LABEL_OP_ID_RE.fullmatch(after):
        raise rejection(422, 'invalid_request', f'{after!r} is not a label operation id')
    if not undoable_only:
        window = await list_ops(limit=limit + 1, after=after)
        page = window[:limit]
        counts = await current_counts(opensearch, page)
        return _list_body(
            [label_op_wire(r, counts[r.op_id], await served_status(r)) for r in page],
            limit,
            page[-1].op_id if len(window) > limit else None,
        )
    served: list[dict[str, Any]] = []
    cursor = after
    while len(served) <= limit:
        chunk = await list_ops(limit=_COUNT_CHUNK, after=cursor)
        if not chunk:
            break
        cursor = chunk[-1].op_id
        counts = await current_counts(opensearch, chunk)
        for record in chunk:
            wire = label_op_wire(record, counts[record.op_id], await served_status(record))
            if wire['undoable'] and len(served) <= limit:
                served.append(wire)
    page_items = served[:limit]
    return _list_body(page_items, limit, page_items[-1]['op_id'] if len(served) > limit else None)


def _list_body(items: list[dict[str, Any]], limit: int, next_after: str | None) -> dict[str, Any]:
    return {'items': items, 'limit': limit, 'retention': LABEL_OPS_KEEP, 'next_after': next_after}
