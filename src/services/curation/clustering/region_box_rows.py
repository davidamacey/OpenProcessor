"""Region boxes as clustering rows: read them, write cluster fields back.

Region clustering works on *boxes*, not items: one row per box that has a
vector. Rows are the join of ``region_boxes`` and ``region_box_embeddings``
(:func:`~src.services.curation.region_box_embeddings.join_box_vectors`);
a cluster result goes back onto the box itself (``cluster_id`` /
``cluster_subid`` / ``cluster_distance`` inside ``region_boxes``).

Every write is an OCC read-modify-write over the live item
(:func:`write_box_edits`): the edit is a pure function of the box *as it
is now*, so a box a human moved, re-stated or deleted while the model was
fitting is never overwritten with a stale decision, and the list, counts
and ``region_revision`` go through
:func:`~src.services.curation.region_boxes.boxes_write_fields` like every
other box writer. A version conflict re-merges against the new version
rather than dropping the write.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from src.clients.occ import occ_skip_on_conflict_bulk
from src.clients.occ_locks import is_locked_box
from src.config import get_region_fields
from src.core.logging import get_logger
from src.services.curation.region_box_embeddings import box_vector_source_includes, join_box_vectors
from src.services.curation.region_boxes import (
    RegionBox,
    box_query,
    boxes_write_fields,
    read_boxes,
    without_cluster,
)


if TYPE_CHECKING:
    from collections.abc import Sequence

    from opensearchpy import AsyncOpenSearch

    from src.config.region_fields import RegionFields


logger = get_logger(__name__)

_CONFLICT_RETRIES = 3
_SCROLL_PAGE = 2000


@dataclass(frozen=True)
class RegionBoxRow:
    """One clustering row: a box, its vector and its item's class name."""

    crop_id: str
    box: RegionBox
    vector: list[float]
    class_name: str | None = None

    @property
    def box_id(self) -> str:
        return self.box.box_id


def box_state_clause(states: Sequence[str], F: RegionFields | None = None) -> dict[str, Any]:
    """The per-box clause ``state in states`` (one nested clause, so any
    further per-box filter ANDed into it matches the same box)."""
    F = F or get_region_fields()
    return {'terms': {f'{F.boxes}.{F.boxes_state}': list(states)}}


def box_filter(
    states: Sequence[str],
    *,
    cluster_id: int | None = None,
    F: RegionFields | None = None,
) -> dict[str, Any]:
    """The one nested clause selecting boxes in ``states`` (and, when given,
    in ``cluster_id``) -- both conditions on the same box."""
    F = F or get_region_fields()
    must: list[dict[str, Any]] = [box_state_clause(states, F)]
    if cluster_id is not None:
        must.append({'term': {f'{F.boxes}.cluster_id': cluster_id}})
    return box_query({'bool': {'filter': must}}, F)


def has_vector_clause(F: RegionFields | None = None) -> dict[str, Any]:
    """Items that carry at least one stored box vector."""
    F = F or get_region_fields()
    return {
        'nested': {
            'path': F.box_embeddings,
            'query': {'exists': {'field': f'{F.box_embeddings}.embedding'}},
        }
    }


async def count_boxes(
    client: AsyncOpenSearch, *, index: str, states: Sequence[str], cluster_id: int | None = None
) -> int:
    """The number of *boxes* (not items) in ``states`` / ``cluster_id``."""
    F = get_region_fields()
    must: list[dict[str, Any]] = [box_state_clause(states, F)]
    if cluster_id is not None:
        must.append({'term': {f'{F.boxes}.cluster_id': cluster_id}})
    resp = await client.search(
        index=index,
        body={
            'size': 0,
            'track_total_hits': False,
            'aggs': {
                'boxes': {
                    'nested': {'path': F.boxes},
                    'aggs': {'matching': {'filter': {'bool': {'filter': must}}}},
                }
            },
        },
    )
    boxes = (resp.get('aggregations') or {}).get('boxes') or {}
    return int((boxes.get('matching') or {}).get('doc_count', 0))


async def scroll_box_rows(
    client: AsyncOpenSearch,
    *,
    index: str,
    states: Sequence[str],
    cluster_id: int | None = None,
    filters: Sequence[dict[str, Any]] = (),
    must_not: Sequence[dict[str, Any]] = (),
) -> list[RegionBoxRow]:
    """Every box in ``states`` (and ``cluster_id``) with a valid stored
    vector, from the items matching ``filters`` / not matching ``must_not``.

    A box whose vector is missing or stale (the box moved since it was
    computed) is not a row: the backfill / worker re-embeds it first.
    """
    F = get_region_fields()
    query = {
        'bool': {
            'filter': [
                box_filter(states, cluster_id=cluster_id, F=F),
                has_vector_clause(F),
                *filters,
            ],
            'must_not': list(must_not),
        }
    }
    body = {
        'size': _SCROLL_PAGE,
        'query': query,
        '_source': [F.boxes, *box_vector_source_includes(F), 'class_name'],
    }
    rows: list[RegionBoxRow] = []
    resp = await client.search(index=index, body=body, scroll='5m')
    scroll_id = resp.get('_scroll_id')
    hits = resp['hits']['hits']
    try:
        while hits:
            for hit in hits:
                src = hit.get('_source') or {}
                for box, vector in join_box_vectors(src, F):
                    if box.state not in states:
                        continue
                    if cluster_id is not None and box.cluster_id != cluster_id:
                        continue
                    rows.append(
                        RegionBoxRow(
                            crop_id=hit['_id'],
                            box=box,
                            vector=vector,
                            class_name=src.get('class_name'),
                        )
                    )
            resp = await client.scroll(scroll_id=scroll_id, scroll='5m')
            scroll_id = resp.get('_scroll_id')
            hits = resp['hits']['hits']
    finally:
        if scroll_id:
            try:
                await client.clear_scroll(scroll_id=scroll_id)
            except Exception as exc:
                logger.warning('curation_clear_scroll_failed', error=str(exc))
    return rows


def human_final_clauses(F: RegionFields | None = None) -> list[tuple[str, str, Any]]:
    """The item-level clauses that make a region verdict final for an
    automated re-cluster: a human label source or verifier, or a validated
    set. One list drives both the query side (:func:`human_final_must_not`)
    and the write-side check (:func:`item_is_human_final`)."""
    F = F or get_region_fields()
    return [
        (F.label_source, 'eq', 'human'),
        (F.verifier, 'eq', 'human'),
        (F.validated, 'eq', True),
    ]


def item_is_human_final(current: dict[str, Any], F: RegionFields | None = None) -> bool:
    return any(current.get(field) == value for field, _op, value in human_final_clauses(F))


def human_final_must_not(F: RegionFields | None = None) -> list[dict[str, Any]]:
    """Query clauses excluding the items :func:`item_is_human_final` matches."""
    return [{'term': {field: value}} for field, _op, value in human_final_clauses(F)]


BoxEdit = Callable[[RegionBox], RegionBox]


async def write_box_edits(
    client: AsyncOpenSearch,
    *,
    index: str,
    edits: dict[str, dict[str, BoxEdit]],
    respect_human: bool,
    item_fields: Callable[[Sequence[RegionBox]], dict[str, Any]] | None = None,
    writer_id: str = 'region_clustering',
) -> dict[str, int]:
    """Apply per-box edits onto the live items: ``edits[crop_id][box_id]``
    is a pure function of the box as it is now (it re-checks its own
    preconditions and returns the box unchanged when they no longer hold).

    ``respect_human`` adds the automated-writer guard: an item whose verdict
    is human-final (:func:`item_is_human_final`) is skipped whole, and a
    locked box (a human's or an import's, ``is_locked_box``) is never
    edited. ``item_fields`` supplies extra item-level fields from the new
    box list (e.g. the re-derived status).

    A conflicting write is re-merged against the new version, up to
    :data:`_CONFLICT_RETRIES` times. Returns the counts, each named for its
    unit: ``items_written`` (items updated), ``boxes_changed`` (boxes whose
    stored value changed, in the items written), ``items_unchanged`` (items
    where no edit applied), ``items_conflicted`` (items dropped after the
    retries) and ``items_errored``.
    """
    F = get_region_fields()
    unchanged: set[str] = set()
    changed: dict[str, int] = {}

    def merger(crop_id: str, current: dict[str, Any]) -> dict[str, Any]:
        # Re-run on every conflict retry, so the tallies are keyed by item:
        # the last merge of an item is the one that counts.
        changed.pop(crop_id, None)
        unchanged.discard(crop_id)
        if respect_human and item_is_human_final(current, F):
            unchanged.add(crop_id)
            return {}
        by_box = edits[crop_id]
        boxes = read_boxes(current, F)
        new_boxes: list[RegionBox] = []
        for box in boxes:
            edit = by_box.get(box.box_id)
            if edit is None or (respect_human and is_locked_box(box)):
                new_boxes.append(box)
                continue
            new_boxes.append(edit(box))
        if new_boxes == boxes:
            unchanged.add(crop_id)
            return {}
        changed[crop_id] = sum(1 for old, new in zip(boxes, new_boxes, strict=True) if old != new)
        # A cluster-only write (partition, refine) changes nothing an open
        # editor sees, so it must not invalidate its expected revision.
        visible = any(
            without_cluster(old) != without_cluster(new)
            for old, new in zip(boxes, new_boxes, strict=True)
        )
        doc = dict(boxes_write_fields(new_boxes, current_src=current, F=F, bump_revision=visible))
        doc['updated_at'] = datetime.now(UTC).isoformat()
        if item_fields is not None:
            doc.update(item_fields(new_boxes))
        return doc

    pending = list(edits)
    items_written = 0
    errored: set[str] = set()
    for _attempt in range(_CONFLICT_RETRIES):
        if not pending:
            break
        result = await occ_skip_on_conflict_bulk(
            client, doc_ids=pending, merger=merger, index=index, writer_id=writer_id
        )
        items_written += int(result['updated'])
        errored.update(str(e.get('doc_id')) for e in result['errors'])
        pending = list(result['skipped_ids'])
    if errored:
        logger.warning('region_box_edit_errors', writer=writer_id, errors=len(errored))
    if pending:
        logger.warning('region_box_edit_conflicts_dropped', writer=writer_id, items=len(pending))
    not_written = errored.union(pending)
    return {
        'items_written': items_written,
        'boxes_changed': sum(n for crop_id, n in changed.items() if crop_id not in not_written),
        'items_unchanged': len(unchanged),
        'items_conflicted': len(pending),
        'items_errored': len(errored),
    }


def with_cluster(cluster_id: int, distance: float | None, *, only_states: Sequence[str]) -> BoxEdit:
    """Place a box in ``cluster_id`` at ``distance`` (clearing a sub-id from
    a previous partition) -- only while it is still in one of ``only_states``."""

    def edit(box: RegionBox) -> RegionBox:
        if box.state not in only_states:
            return box
        return dataclasses.replace(
            box, cluster_id=cluster_id, cluster_subid=None, cluster_distance=distance
        )

    return edit


__all__ = [
    'BoxEdit',
    'RegionBoxRow',
    'box_filter',
    'box_state_clause',
    'count_boxes',
    'has_vector_clause',
    'human_final_clauses',
    'human_final_must_not',
    'item_is_human_final',
    'scroll_box_rows',
    'with_cluster',
    'write_box_edits',
]
