"""The ``target=box`` half of curation semantic search.

Per-box vectors live in the nested ``region_box_embeddings`` field (one entry per accepted or
false-positive box, see :mod:`src.services.curation.region_box_embeddings`). A nested ``knn``
ranks those entries; ``score_mode: max`` collapses an item's boxes to its best one, so the page
is still one card per item. The parent filter sits in an outer ``bool.filter``: parent fields
inside the nested query would match nothing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.domain.region_boxes import read_boxes
from src.services.curation.region_box_embeddings import is_current


if TYPE_CHECKING:
    from src.config.region_fields import RegionFields


# The nested kNN's ``k`` counts boxes, not items, and the parent filter is applied to its
# candidates afterwards, so a page needs more candidates than it has cards.
BOX_K_FACTOR = 5

INNER_HITS_NAME = 'matched_box'


def box_k(page: int, page_size: int, *, max_k: int) -> int:
    return min(max_k, BOX_K_FACTOR * page * page_size)


def build_box_knn_query(
    vector: list[float],
    *,
    k: int,
    filter_clause: list[dict[str, Any]],
    fields: RegionFields,
    inner_size: int = 1,
) -> dict[str, Any]:
    """Nested kNN over the box vectors, restricted to items passing ``filter_clause``.

    ``inner_hits`` carries no source: the matched entry is identified by its offset in the
    item's own ``region_box_embeddings`` list (the convention ``region_rows`` uses)."""
    path = fields.box_embeddings
    return {
        'bool': {
            'filter': filter_clause,
            'must': [
                {
                    'nested': {
                        'path': path,
                        'score_mode': 'max',
                        'query': {'knn': {f'{path}.embedding': {'vector': vector, 'k': k}}},
                        'inner_hits': {
                            'name': INNER_HITS_NAME,
                            'size': inner_size,
                            '_source': False,
                        },
                    }
                }
            ],
        }
    }


def no_box_vectors_clause(fields: RegionFields) -> dict[str, Any]:
    """Items with no box vector at all: the box search cannot see them."""
    path = fields.box_embeddings
    return {
        'bool': {
            'must_not': [
                {'nested': {'path': path, 'query': {'exists': {'field': f'{path}.embedding'}}}}
            ]
        }
    }


def box_source_excludes(excludes: list[str], fields: RegionFields) -> list[str]:
    """``excludes`` with the box entries readable but for their vector: hydration needs each
    entry's ``box_id`` and ``bbox_norm`` (to name the match and judge it stale), never the
    vector."""
    path = fields.box_embeddings
    return [*[e for e in excludes if e != path], f'{path}.embedding']


def matched_boxes(hit: dict[str, Any], fields: RegionFields) -> list[tuple[str, bool]]:
    """``[(box_id, stale)]`` for the boxes that matched, best first.

    ``stale`` is true when the vector was computed from geometry the box no longer has (or the
    box is gone). Absent ``inner_hits`` yields an empty list, not an error."""
    source = hit.get('_source') or {}
    entries = source.get(fields.box_embeddings) or []
    live = {b.box_id: b for b in read_boxes(source, fields)}
    block = ((hit.get('inner_hits') or {}).get(INNER_HITS_NAME) or {}).get('hits') or {}
    out: list[tuple[str, bool]] = []
    for inner in block.get('hits') or []:
        offset = (inner.get('_nested') or {}).get('offset')
        if not isinstance(offset, int) or not 0 <= offset < len(entries):
            continue
        entry = entries[offset]
        box_id = entry.get('box_id')
        if not box_id:
            continue
        box = live.get(box_id)
        out.append((box_id, box is None or not is_current(entry, box)))
    return out


def attach_match(
    item: dict[str, Any],
    hit: dict[str, Any],
    fields: RegionFields,
    *,
    skip: tuple[str, str] | None = None,
) -> bool:
    """Set ``matched_box_id`` / ``matched_box_stale`` on the wire ``item``.

    ``skip`` is ``(crop_id, box_id)`` of the query's own box: that box is never the match, so an
    item whose only match is the query box is dropped (returns False)."""
    candidates = matched_boxes(hit, fields)
    if skip is not None and hit.get('_id') == skip[0]:
        candidates = [c for c in candidates if c[0] != skip[1]]
        if not candidates:
            return False
    box_id, stale = candidates[0] if candidates else (None, None)
    item['matched_box_id'] = box_id
    item['matched_box_stale'] = stale
    return True
