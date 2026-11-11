"""What an item's box edit needs, shared by the drawn-box create and the move.

``bbox_norm`` is written by a person only here and in
:mod:`~src.services.curation.image_item_write`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.clients.occ import occ_skip_on_conflict_bulk
from src.config import get_curation_config
from src.core.exceptions import RequestRejectedError
from src.domain.item_geometry import ItemBoxError, validate_item_box
from src.schemas.curation.config_common import error_body
from src.services.curation.edit_history import GEOMETRY_WRITER
from src.services.curation.reprocess_targets import items_by_terms
from src.services.detection.geometry import box_area
from src.services.embedding_spaces.readers import vector_source_excludes


if TYPE_CHECKING:
    from collections.abc import Sequence

    from opensearchpy import AsyncOpenSearch


def checked_box(bbox: Sequence[float]) -> tuple[float, float, float, float]:
    """``bbox`` validated, or the 422 the API answers (``box_out_of_image``, ``degenerate_box``,
    ``box_too_small`` with ``min_side``)."""
    try:
        return validate_item_box(bbox)
    except ItemBoxError as exc:
        raise RequestRejectedError(
            422,
            error_body(exc.code, exc.message, **exc.fields),  # type: ignore[arg-type]
        ) from exc


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
