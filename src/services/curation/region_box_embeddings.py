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
is ignored there, and any later embedding write for that item prunes or
replaces it (:func:`merge_box_embeddings`). A human edit that deletes or
moves a box prunes right away (:func:`prune_box_embeddings`).

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


def box_vector_source_includes(F: RegionFields | None = None) -> list[str]:
    """The ``_source`` includes that read the per-box vectors back.

    A search (or scroll) whose ``_source`` names the nested field itself
    (``region_box_embeddings``) gets the number ``1`` instead of each vector on
    an index with derived source (OpenSearch 3.6); the leaf paths return the
    real values. Every search that needs the vectors uses this list."""
    F = F or get_region_fields()
    return [f'{F.box_embeddings}.{leaf}' for leaf in ('box_id', 'bbox_norm', 'embedding')]


def embeddable(boxes: Iterable[RegionBox]) -> list[RegionBox]:
    """The boxes that carry a vector: accepted and false-positive ones."""
    return [b for b in boxes if b.state in EMBEDDED_BOX_STATES]


def entry_for(box: RegionBox, vector: Sequence[float]) -> dict[str, Any]:
    """The stored entry for ``box``'s vector. It records the geometry the
    vector was computed from, so a box a human later moves is recognised as
    having a stale vector (:func:`current_vectors`)."""
    return {'box_id': box.box_id, 'bbox_norm': list(box.bbox_norm), 'embedding': list(vector)}


def is_current(entry: dict[str, Any], box: RegionBox) -> bool:
    """True when ``entry`` is ``box``'s vector, computed from the box's
    present geometry: the one staleness rule for readers and writers."""
    return entry.get('box_id') == box.box_id and same_box(entry.get('bbox_norm'), box.bbox_norm)


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
        if entry is not None and is_current(entry, box):
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
    live: Iterable[RegionBox],
) -> list[dict[str, Any]]:
    """The stored list after writing ``new`` entries against the ``live``
    boxes: an entry for a box id in ``new`` is replaced; an entry whose box
    is gone or has moved since its vector was computed (:func:`is_current`)
    is dropped; the rest are kept, in order. A new entry for a box that is
    gone or moved is dropped too."""
    boxes = {b.box_id: b for b in live}

    def valid(entry: Any) -> bool:
        if not isinstance(entry, dict):
            return False
        box = boxes.get(str(entry.get('box_id')))
        return box is not None and is_current(entry, box)

    replaced = {e['box_id'] for e in new}
    kept = [e for e in existing or [] if valid(e) and e['box_id'] not in replaced]
    return [*kept, *(e for e in new if valid(e))]


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
    entry for a box that was deleted, moved or never landed is dropped, and
    a race with a box write can never leave a vector for a box that is not
    there), and written back conditionally when the merge changed the
    stored list. A version conflict re-reads and retries; the box list
    itself is never written. Returns item counts ``{'written', 'unchanged',
    'skipped', 'errors'}`` (``skipped``: item missing or a conflict that
    outlasted the retries).
    """
    from src.clients.curation_opensearch.crops import mget_crops

    F = get_region_fields()
    pending = dict(by_crop)
    counts = {'written': 0, 'unchanged': 0, 'skipped': 0, 'errors': 0}
    for attempt in range(_WRITE_ATTEMPTS):
        if not pending:
            break
        docs = await mget_crops(
            client,
            list(pending),
            index=index,
            source_includes=[F.boxes, *box_vector_source_includes(F)],
        )
        body: list[dict[str, Any]] = []
        order: list[str] = []
        for crop_id, entries in pending.items():
            doc = docs.get(crop_id)
            if doc is None:
                counts['skipped'] += 1
                continue
            src = doc.get('_source') or {}
            stored = src.get(F.box_embeddings) or []
            merged = merge_box_embeddings(stored, entries, read_boxes(src, F))
            if merged == stored:
                counts['unchanged'] += 1
                continue
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


async def prune_box_embeddings(
    client: AsyncOpenSearch, *, index: str, crop_ids: Iterable[str]
) -> None:
    """Drop the entries of ``crop_ids`` whose box a human just deleted or
    moved. Hygiene, not correctness (readers already ignore such an entry),
    so a failure is logged and never fails the edit that triggered it."""
    ids = list(crop_ids)
    if not ids:
        return
    try:
        await write_box_embeddings(client, index=index, by_crop={c: [] for c in ids})
    except Exception as exc:
        logger.warning('box_embeddings_prune_failed', error=str(exc), n=len(ids))


__all__ = [
    'EMBEDDED_BOX_STATES',
    'box_vector_source_includes',
    'current_vectors',
    'embeddable',
    'entry_for',
    'is_current',
    'join_box_vectors',
    'merge_box_embeddings',
    'missing_boxes',
    'prune_box_embeddings',
    'write_box_embeddings',
]
