"""A person moves or resizes an item's box, and undoes it.

One write, one concurrency check: the revision is compared inside the OCC merger
(:func:`~src.clients.occ.occ_update_one`), against the document the conditional write is made on,
so two editors holding the same revision cannot both win. The vector of the old box is dropped in
the same write, so a failed re-embed never leaves it on a moved box. ``bbox_norm`` is written
here and in :mod:`~src.services.curation.image_item_write`, nowhere else for a person.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from opensearchpy.exceptions import NotFoundError

from src.clients.occ import occ_skip_on_conflict_bulk, occ_update_one
from src.clients.occ_locks import has_human_geometry
from src.config import get_curation_config, get_region_fields
from src.core.exceptions import RequestRejectedError
from src.domain.item_geometry import (
    GEOMETRY_SOURCE_HUMAN,
    ItemBoxError,
    box_area_norm,
    validate_item_box,
)
from src.domain.region_boxes import boxes_write_fields, read_boxes
from src.schemas.curation.config_common import error_body
from src.services.curation.edit_history import (
    EDIT_HISTORY_FIELD,
    GEOMETRY_UNDO_WRITER,
    GEOMETRY_WRITER,
    REGION_REVISION_BEFORE,
    EditKind,
    find_edit_undo,
    record_edit,
    with_dropped_regions,
)
from src.services.curation.edit_undo import edit_undo_update
from src.services.curation.item_embed_inline import EmbedOutcome, embed_item_inline
from src.services.curation.region_box_embeddings import prune_box_embeddings
from src.services.curation.reprocess_targets import items_by_terms
from src.services.curation.source_image_cache import evict_crop_cache
from src.services.curation.wire import serialize_item
from src.services.detection.geometry import box_area
from src.services.embedding_spaces.readers import (
    item_state_fields,
    item_vector_resets,
    vector_source_excludes,
)
from src.utils.timeutil import now_iso


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from opensearchpy import AsyncOpenSearch

#: A region box counts as inside the item box up to this slack.
INSIDE_TOLERANCE = 1e-4


class GeometryConflictError(Exception):
    """The caller's revision is not the stored one; ``current`` is the stored item."""

    def __init__(self, current: dict[str, Any]) -> None:
        super().__init__('geometry_conflict')
        self.current = current


class RegionsOutsideItemError(Exception):
    """Region boxes would fall outside the new item box and were not listed to be dropped."""

    def __init__(self, box_ids: list[str]) -> None:
        super().__init__('regions_outside_item')
        self.box_ids = box_ids


class RegionsChangedError(Exception):
    """The edit to undo dropped region boxes and a region write has happened since."""


class NoGeometryToUndoError(Exception):
    """The item has no un-undone geometry edit on record."""


@dataclass(frozen=True)
class GeometryResult:
    source: dict[str, Any]
    """The stored item after the write and the embed (no vectors)."""
    embed: EmbedOutcome


def _inside(inner: Sequence[float], outer: Sequence[float]) -> bool:
    slack = INSIDE_TOLERANCE
    return (
        inner[0] >= outer[0] - slack
        and inner[1] >= outer[1] - slack
        and inner[2] <= outer[2] + slack
        and inner[3] <= outer[3] + slack
    )


def checked_box(bbox: Sequence[float]) -> tuple[float, float, float, float]:
    """``bbox`` validated, or the 422 the API answers (``box_out_of_image``, ``degenerate_box``,
    ``box_too_small`` with ``min_side``)."""
    try:
        return validate_item_box(bbox)
    except ItemBoxError as exc:
        raise RequestRejectedError(422, error_body(exc.code, exc.message, **exc.fields)) from exc  # type: ignore[arg-type]


def geometry_edit_merger(
    bbox: Sequence[float],
    *,
    expected_revision: int,
    drop_region_box_ids: Sequence[str],
    resets: dict[str, Any],
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """The OCC merger of a box edit. Raises :class:`GeometryConflictError` on a stale revision and
    :class:`RegionsOutsideItemError` when region boxes would be left outside the new box.
    ``resets`` are the vector and state fields the move clears."""
    new_box = list(checked_box(bbox))
    drop = set(drop_region_box_ids)

    def _merge(current: dict[str, Any]) -> dict[str, Any]:
        revision = int(current.get('geometry_revision') or 0)
        if revision != expected_revision:
            raise GeometryConflictError(current)
        F = get_region_fields()
        boxes = read_boxes(current, F)
        outside = {b.box_id for b in boxes if not _inside(b.bbox_norm, new_box)}
        unlisted = sorted(outside - drop)
        if unlisted:
            raise RegionsOutsideItemError(unlisted)
        now = now_iso()
        history = record_edit(current, kind=EditKind.GEOMETRY, writer=GEOMETRY_WRITER, now=now)
        if drop & outside:
            history = with_dropped_regions(history, current)
        update: dict[str, Any] = {
            'bbox_norm': new_box,
            'crop_area_norm': box_area_norm(new_box),
            'geometry_source': GEOMETRY_SOURCE_HUMAN,
            'geometry_revision': revision + 1,
            EDIT_HISTORY_FIELD: history,
            'updated_at': now,
            **resets,
        }
        if not has_human_geometry(current) and not current.get('detector_bbox_norm'):
            # The machine's own box, kept on the first move so a re-detect cannot re-create it.
            update['detector_bbox_norm'] = current.get('bbox_norm')
        if drop & outside:
            kept = [b for b in boxes if b.box_id not in drop]
            update.update(boxes_write_fields(kept, F=F, current_src=current))
        return update

    return _merge


def geometry_undo_merger(
    resets: dict[str, Any], *, expected_revision: int
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """The OCC merger of an undo: the geometry before the newest un-undone edit, the revision
    moved forward (an undo is a change a stale editor must see), the vector cleared. The revision
    is checked on the document written, so a move that landed after the editor looked is a
    :class:`GeometryConflictError`, never undone unseen."""

    def _merge(current: dict[str, Any]) -> dict[str, Any]:
        if int(current.get('geometry_revision') or 0) != expected_revision:
            raise GeometryConflictError(current)
        entry = find_edit_undo(current.get(EDIT_HISTORY_FIELD), EditKind.GEOMETRY)
        before = (entry or {}).get('state', {}).get(REGION_REVISION_BEFORE)
        if before is not None and int(current.get(get_region_fields().revision) or 0) != before + 1:
            raise RegionsChangedError
        update = edit_undo_update(current, EditKind.GEOMETRY, GEOMETRY_UNDO_WRITER)
        if update is None:
            raise NoGeometryToUndoError
        return {**update, **resets}

    return _merge


def _not_found(crop_id: str) -> RequestRejectedError:
    return RequestRejectedError(
        404, error_body('crop_not_found', f'crop not found: {crop_id}', crop_ids=[crop_id])
    )


async def _conflict(crop_id: str, current: dict[str, Any]) -> RequestRejectedError:
    item = serialize_item(current, crop_id, state_fields=await item_state_fields())
    return RequestRejectedError(
        409,
        error_body(
            'geometry_conflict',
            'the box was changed by someone else; reload the item and try again',
            current_geometry_revision=int(current.get('geometry_revision') or 0),
            item=item,
        ),
    )


async def rerank_image_items(opensearch: AsyncOpenSearch, image_id: str) -> None:
    """Rank the image's items by box area, largest first (ingest's rule, ties by id). A machine
    field: a concurrent writer wins and the next rank pass fixes it."""
    cfg = get_curation_config()
    rows = await items_by_terms(
        opensearch, 'image_id', [image_id], index=cfg.items_index, includes=['bbox_norm']
    )
    boxed = [(cid, src['bbox_norm']) for cid, src in rows if len(src.get('bbox_norm') or []) == 4]
    ordered = sorted(boxed, key=lambda row: (-box_area(tuple(row[1])), row[0]))
    ranks = {cid: rank for rank, (cid, _box) in enumerate(ordered, start=1)}

    def merge(doc_id: str, src: dict[str, Any]) -> dict[str, Any]:
        if src.get('crop_rank_in_image') == ranks[doc_id]:
            return {}
        return {'crop_rank_in_image': ranks[doc_id]}

    await occ_skip_on_conflict_bulk(
        opensearch,
        doc_ids=list(ranks),
        merger=merge,
        vector_excludes=await vector_source_excludes(),
        index=cfg.items_index,
        refresh='wait_for',
        writer_id=GEOMETRY_WRITER,
    )


async def _finish(
    opensearch: AsyncOpenSearch, pe: Any | None, crop_id: str, *, dropped_regions: bool
) -> GeometryResult:
    """After the conditional write: the old crop picture goes, ranks follow the new area, the item
    is embedded (a failure leaves it ``deferred``) and the stored item is read back."""
    cfg = get_curation_config()
    evict_crop_cache(cfg.crop_cache_dir, crop_id)
    if dropped_regions:
        await prune_box_embeddings(opensearch, index=cfg.items_index, crop_ids=[crop_id])
    stored = await opensearch.get(
        index=cfg.items_index, id=crop_id, _source_excludes=await vector_source_excludes()
    )
    await rerank_image_items(opensearch, str(stored['_source'].get('image_id') or ''))
    embed = await embed_item_inline(opensearch, pe, crop_id)
    final = await opensearch.get(
        index=cfg.items_index, id=crop_id, _source_excludes=await vector_source_excludes()
    )
    return GeometryResult(final['_source'], embed)


async def edit_geometry(
    opensearch: AsyncOpenSearch,
    pe: Any | None,
    crop_id: str,
    *,
    bbox: Sequence[float],
    expected_revision: int,
    drop_region_box_ids: Sequence[str],
) -> GeometryResult:
    """Move or resize ``crop_id``'s box. Raises :class:`GeometryConflictError`,
    :class:`RegionsOutsideItemError`, :class:`~src.domain.item_geometry.ItemBoxError` or a 404
    :class:`~src.core.exceptions.RequestRejectedError`."""
    merger = geometry_edit_merger(
        bbox,
        expected_revision=expected_revision,
        drop_region_box_ids=drop_region_box_ids,
        resets=await item_vector_resets(),
    )
    try:
        update = await occ_update_one(
            opensearch,
            doc_id=crop_id,
            merger=merger,
            vector_excludes=await vector_source_excludes(),
            refresh='wait_for',
            writer_id=GEOMETRY_WRITER,
        )
    except NotFoundError as exc:
        raise _not_found(crop_id) from exc
    except GeometryConflictError as exc:
        raise await _conflict(crop_id, exc.current) from exc
    except RegionsOutsideItemError as exc:
        raise RequestRejectedError(
            422,
            error_body(
                'regions_outside_item',
                'region boxes would fall outside the item box; list them in drop_region_box_ids',
                box_ids=exc.box_ids,
            ),
        ) from exc
    return await _finish(
        opensearch, pe, crop_id, dropped_regions=get_region_fields().boxes in update
    )


async def undo_geometry(
    opensearch: AsyncOpenSearch, pe: Any | None, crop_id: str, *, expected_revision: int
) -> GeometryResult:
    """Put back the geometry before the newest un-undone edit, if the item is still at
    ``expected_revision``. Raises a 404, 409 ``geometry_conflict``, ``nothing_to_undo`` or
    ``regions_changed`` :class:`~src.core.exceptions.RequestRejectedError`."""
    try:
        update = await occ_update_one(
            opensearch,
            doc_id=crop_id,
            merger=geometry_undo_merger(
                await item_vector_resets(), expected_revision=expected_revision
            ),
            vector_excludes=await vector_source_excludes(),
            refresh='wait_for',
            writer_id=GEOMETRY_UNDO_WRITER,
        )
    except NotFoundError as exc:
        raise _not_found(crop_id) from exc
    except GeometryConflictError as exc:
        raise await _conflict(crop_id, exc.current) from exc
    except RegionsChangedError as exc:
        raise RequestRejectedError(
            409,
            error_body(
                'regions_changed',
                'the region boxes were edited after this move dropped some; undo it by hand',
            ),
        ) from exc
    except NoGeometryToUndoError as exc:
        raise RequestRejectedError(
            409, error_body('nothing_to_undo', f'nothing to undo for crop geometry {crop_id}')
        ) from exc
    return await _finish(
        opensearch, pe, crop_id, dropped_regions=get_region_fields().boxes in update
    )
