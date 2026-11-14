"""Find the items of the bound project that hold too many nested documents.

An item written before the nested-document limit (``src/domain/nested_limit.py``, OpenSearch k-NN
issue 3623) can hold 150 or more nested documents: such an item is read back with another
document's box vectors, and every write on it is refused except the one repair write (a box-list
write that brings it under the limit; it also empties the box vectors, which the box embed stage
and ``backfill_region_embeddings.py`` then re-create).

Read-only. The count is the one the writers apply: box vectors are not fetched, so each box-vector
list of the project counts one entry per box (the most it can hold).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from src.config.nested_limit import NESTED_DOCS_LIMIT
from src.config.region_fields import get_region_fields
from src.domain.nested_limit import box_vector_paths, nested_doc_count
from src.services.curation.repair_scan import scan_plans


if TYPE_CHECKING:
    from collections.abc import Collection


@dataclass(frozen=True)
class OverLimitItem:
    crop_id: str
    nested_docs: int
    boxes: int
    remove_at_least: int
    """Boxes to remove so the item is under the limit: a human decides which."""


def _remove_at_least(src: dict[str, Any], boxes: int, lists: int) -> int:
    fixed = nested_doc_count({k: v for k, v in src.items() if k != get_region_fields().boxes})
    room = NESTED_DOCS_LIMIT - 1 - fixed
    return max(0, boxes - max(room, 0) // (1 + lists))


def _plan(crop_id: str, src: dict[str, Any], paths: Collection[str]) -> OverLimitItem | None:
    F = get_region_fields()
    nested = nested_doc_count(src, unread_box_vector_paths=paths, F=F)
    if nested < NESTED_DOCS_LIMIT:
        return None
    boxes = len(src.get(F.boxes) or [])
    return OverLimitItem(crop_id, nested, boxes, _remove_at_least(src, boxes, len(paths)))


async def find_over_limit_items(
    client: Any, *, index: str, vector_excludes: Collection[str], page_size: int = 500
) -> list[OverLimitItem]:
    """Every item of ``index`` at or over the limit, in ``crop_id`` order."""
    F = get_region_fields()
    paths = box_vector_paths(vector_excludes, F)
    query = {'nested': {'path': F.boxes, 'query': {'exists': {'field': f'{F.boxes}.box_id'}}}}
    return await scan_plans(
        client,
        index=index,
        query=query,
        plan_one=lambda crop_id, src: _plan(crop_id, src, paths),
        page_size=page_size,
    )


__all__ = ['OverLimitItem', 'find_over_limit_items']
