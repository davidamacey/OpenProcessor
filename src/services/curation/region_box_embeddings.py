"""Per-box region embeddings (``region_box_embeddings``).

One vector per embeddable box, stored in a sibling nested field of the box
list -- ``[{box_id, bbox_norm, embedding}]`` (``bbox_norm`` is the geometry
the vector was computed from) -- not inside ``region_boxes``: every item
read that feeds the wire excludes vectors, and a human edit that read the
list without them and wrote it back would otherwise delete every embedding.
Box writers therefore never touch this field; only the worker's embed
stage and the backfill script write it, through :func:`write_box_embeddings`.

Readers join the two lists by ``box_id`` (:func:`join_box_vectors`); an
entry whose box is gone, or whose box moved since the vector was computed,
is ignored there, and the next write for that item prunes or replaces it
(:func:`merge_box_embeddings`).

Which boxes are embedded: ``accepted`` (the clustering pool) and
``false_positive`` (the FP matcher's input); never ``rejected`` or
``proposed``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.config.region_fields import RegionFields, get_region_fields
from src.config.region_state import RegionStatus
from src.core.logging import get_logger
from src.services.curation.region_box_edits import same_box
from src.services.curation.region_boxes import RegionBox, read_boxes


if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from opensearchpy import AsyncOpenSearch

logger = get_logger(__name__)

EMBEDDED_BOX_STATES: tuple[str, ...] = (
    'accepted',
    RegionStatus.FALSE_POSITIVE.value,
)


def embeddable(boxes: Iterable[RegionBox]) -> list[RegionBox]:
    """The boxes that carry a vector: accepted and false-positive ones."""
    return [b for b in boxes if b.state in EMBEDDED_BOX_STATES]


def entry_for(box: RegionBox, vector: Sequence[float]) -> dict[str, Any]:
    """The stored entry for ``box``'s vector. It records the geometry the
    vector was computed from, so a box a human later moves is recognised as
    having a stale vector (:func:`current_vectors`)."""
    return {'box_id': box.box_id, 'bbox_norm': list(box.bbox_norm), 'embedding': list(vector)}


def current_vectors(src: dict[str, Any], F: RegionFields | None = None) -> dict[str, list[float]]:
    """``{box_id: vector}`` for every box of ``src`` whose stored vector is
    still valid: the entry exists and was computed from the box's present
    geometry. An orphan entry (no such box) and a stale one (the box moved)
    are skipped. ``src`` may carry the entries without their vectors (an
    ``_source`` include of ``box_id`` / ``bbox_norm`` only) when the caller
    just needs the ids."""
    F = F or get_region_fields()
    entries = {
        e['box_id']: e
        for e in src.get(F.box_embeddings) or []
        if isinstance(e, dict) and e.get('box_id')
    }
    out: dict[str, list[float]] = {}
    for box in read_boxes(src, F):
        entry = entries.get(box.box_id)
        if entry is not None and same_box(entry.get('bbox_norm'), box.bbox_norm):
            out[box.box_id] = entry.get('embedding')  # type: ignore[assignment]
    return out


def missing_boxes(src: dict[str, Any], F: RegionFields | None = None) -> list[RegionBox]:
    """The embeddable boxes of ``src`` with no valid stored vector."""
    F = F or get_region_fields()
    have = current_vectors(src, F)
    return [b for b in embeddable(read_boxes(src, F)) if b.box_id not in have]


def merge_box_embeddings(
    existing: Sequence[dict[str, Any]] | None,
    new: Sequence[dict[str, Any]],
    live_ids: set[str],
) -> list[dict[str, Any]]:
    """The stored list after writing ``new`` entries: an entry for a box id
    in ``new`` is replaced, an entry whose box is no longer in ``live_ids``
    is dropped (orphan pruning), the rest are kept, in order."""
    replaced = {e['box_id'] for e in new}
    kept = [
        e
        for e in existing or []
        if isinstance(e, dict) and e.get('box_id') in live_ids and e.get('box_id') not in replaced
    ]
    return [*kept, *(e for e in new if e['box_id'] in live_ids)]


def join_box_vectors(
    src: dict[str, Any], F: RegionFields | None = None
) -> list[tuple[RegionBox, list[float]]]:
    """``(box, vector)`` for every box of ``src`` with a valid stored
    vector (:func:`current_vectors`), in box order."""
    F = F or get_region_fields()
    vectors = current_vectors(src, F)
    return [(b, vectors[b.box_id]) for b in read_boxes(src, F) if vectors.get(b.box_id) is not None]


_WRITE_ATTEMPTS = 3


async def write_box_embeddings(
    client: AsyncOpenSearch,
    *,
    index: str,
    by_crop: dict[str, list[dict[str, Any]]],
) -> dict[str, int]:
    """Write per-box vectors onto their items: ``by_crop`` maps a crop id to
    its new ``[{box_id, embedding}]`` entries.

    Each item is read with its box list and current entries, merged with
    :func:`merge_box_embeddings` against the boxes that exist *now* (so an
    entry for a box that was deleted or never landed is dropped, and a
    race with a box write can never leave a vector for a box that is not
    there), and written back conditionally. A version conflict re-reads and
    retries; the box list itself is never written. Returns
    ``{'written', 'skipped', 'errors'}``.
    """
    from src.clients.curation_opensearch import mget_crops

    F = get_region_fields()
    pending = dict(by_crop)
    counts = {'written': 0, 'skipped': 0, 'errors': 0}
    for attempt in range(_WRITE_ATTEMPTS):
        if not pending:
            break
        docs = await mget_crops(
            client, list(pending), index=index, source_includes=[F.boxes, F.box_embeddings]
        )
        body: list[dict[str, Any]] = []
        order: list[str] = []
        for crop_id, entries in pending.items():
            doc = docs.get(crop_id)
            if doc is None:
                counts['skipped'] += 1
                continue
            src = doc.get('_source') or {}
            live = {b.box_id for b in read_boxes(src, F)}
            merged = merge_box_embeddings(src.get(F.box_embeddings), entries, live)
            body.append(
                {
                    'update': {
                        '_index': index,
                        '_id': crop_id,
                        'if_seq_no': doc['_seq_no'],
                        'if_primary_term': doc['_primary_term'],
                    }
                }
            )
            body.append({'doc': {F.box_embeddings: merged}})
            order.append(crop_id)
        retry: dict[str, list[dict[str, Any]]] = {}
        if body:
            try:
                resp = await client.bulk(body=body, refresh=False)
            except Exception as exc:
                logger.warning('box_embeddings_bulk_failed', error=str(exc))
                counts['errors'] += len(order)
                break
            for crop_id, item in zip(order, resp.get('items') or [], strict=True):
                action = item.get('update') or {}
                status = action.get('status')
                if status in (200, 201):
                    counts['written'] += 1
                elif status == 409 and attempt < _WRITE_ATTEMPTS - 1:
                    retry[crop_id] = pending[crop_id]
                elif status == 409:
                    counts['skipped'] += 1
                else:
                    counts['errors'] += 1
                    logger.warning(
                        'box_embeddings_write_failed', crop_id=crop_id, error=action.get('error')
                    )
        pending = retry
    return counts


__all__ = [
    'EMBEDDED_BOX_STATES',
    'current_vectors',
    'embeddable',
    'entry_for',
    'join_box_vectors',
    'merge_box_embeddings',
    'missing_boxes',
    'write_box_embeddings',
]
