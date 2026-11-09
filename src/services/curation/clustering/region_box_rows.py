"""Region boxes as rows: read the boxes with a vector, write automated box edits back.

Region clustering and the false-positive matcher work on *boxes*, not items: one row per box
that has a vector. Rows are the join of ``region_boxes`` and ``region_box_embeddings``
(:func:`~src.services.curation.region_box_embeddings.join_box_vectors`).

Every write is an OCC read-modify-write over the live item
(:func:`write_box_edits`): the edit is a pure function of the box *as it
is now*, so a box a human moved, re-stated or deleted while the model was
fitting is never overwritten with a stale decision, and the list, counts
and ``region_revision`` go through
:func:`~src.domain.region_boxes.boxes_write_fields` like every
other box writer. A version conflict re-merges against the new version
rather than dropping the write.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from opensearchpy.exceptions import OpenSearchException

from src.clients.occ import occ_skip_on_conflict_bulk
from src.clients.occ_locks import is_locked_box
from src.config import get_region_fields
from src.core.logging import get_logger
from src.domain.region_boxes import RegionBox, box_query, boxes_write_fields, read_boxes
from src.services.curation.region_box_embeddings import box_vector_source_includes, join_box_vectors
from src.services.embedding_spaces.readers import vector_source_excludes


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


def box_filter(states: Sequence[str], *, F: RegionFields | None = None) -> dict[str, Any]:
    """The one nested clause selecting boxes in ``states``."""
    F = F or get_region_fields()
    return box_query({'bool': {'filter': [box_state_clause(states, F)]}}, F)


def has_vector_clause(F: RegionFields | None = None) -> dict[str, Any]:
    """Items that carry at least one stored box vector."""
    F = F or get_region_fields()
    return {
        'nested': {
            'path': F.box_embeddings,
            'query': {'exists': {'field': f'{F.box_embeddings}.embedding'}},
        }
    }


async def scroll_box_rows(
    client: AsyncOpenSearch,
    *,
    index: str,
    states: Sequence[str],
    filters: Sequence[dict[str, Any]] = (),
    must_not: Sequence[dict[str, Any]] = (),
) -> list[RegionBoxRow]:
    """Every box in ``states`` with a valid stored
    vector, from the items matching ``filters`` / not matching ``must_not``.

    A box whose vector is missing or stale (the box moved since it was
    computed) is not a row: the backfill / worker re-embeds it first.

    Rows come back sorted by (crop id, box id), not in scroll order: the scroll
    order is the shard and ingest order, and a k-means fit picks its start by row
    position, so the same boxes in another order gave another partition (ARI 0.553
    on the 3,000-item set). The fit is a function of the box set.
    """
    F = get_region_fields()
    query = {
        'bool': {
            'filter': [
                box_filter(states, F=F),
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
            except OpenSearchException as exc:
                logger.warning('curation_clear_scroll_failed', error=str(exc))
    rows.sort(key=lambda r: (r.crop_id, r.box_id))
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
    writer_id: str = 'region_fp',
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
        doc = dict(boxes_write_fields(new_boxes, current_src=current, F=F))
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
            client,
            vector_excludes=await vector_source_excludes(),
            doc_ids=pending,
            merger=merger,
            index=index,
            writer_id=writer_id,
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


__all__ = [
    'BoxEdit',
    'RegionBoxRow',
    'box_filter',
    'box_state_clause',
    'has_vector_clause',
    'human_final_clauses',
    'human_final_must_not',
    'item_is_human_final',
    'scroll_box_rows',
    'write_box_edits',
]
