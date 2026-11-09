"""The ``target=box`` half of curation semantic search.

Per-box vectors live in the nested ``region_box_embeddings`` field (one entry per accepted or
false-positive box, see :mod:`src.services.curation.region_box_embeddings`). A nested ``knn``
ranks those entries; ``score_mode: max`` collapses an item's boxes to its best one, so the page
is still one card per item. The parent filter sits in an outer ``bool.filter``: parent fields
inside the nested query would match nothing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.domain.embedding_spaces import require_query_space
from src.domain.region_boxes import read_boxes
from src.services.curation.region_box_embeddings import is_current


if TYPE_CHECKING:
    from src.config.region_fields import RegionFields
    from src.domain.embedding_spaces import ResolvedSpace


# The nested kNN's ``k`` counts boxes, not items, and the parent filter is applied to its
# candidates afterwards, so a page needs more candidates than it has cards.
BOX_K_FACTOR = 5

INNER_HITS_NAME = 'matched_box'


def _nested(space: ResolvedSpace) -> str:
    """The nested path of a box space; refuses a space that is not one."""
    if space.nested_path is None:  # unreachable: a box space always has its nested path
        msg = f'space {space.space_key} has no nested path; it is not a box space'
        raise ValueError(msg)
    return space.nested_path


def box_k(page: int, page_size: int, *, max_k: int) -> int:
    return min(max_k, BOX_K_FACTOR * page * page_size)


def build_box_knn_query(
    vector: list[float],
    *,
    vector_space_key: str,
    space: ResolvedSpace,
    k: int,
    filter_clause: list[dict[str, Any]],
    inner_size: int = 1,
) -> dict[str, Any]:
    """Nested kNN over the box vectors of ``space``, restricted to items passing ``filter_clause``.

    ``vector_space_key`` is the space the query vector was made in; a vector of another space is
    refused. ``inner_hits`` carries no source: the matched entry is identified by its offset in
    the item's own list under the space's nested path (the convention ``region_rows`` uses)."""
    require_query_space(space, vector_space_key)
    path = _nested(space)
    return {
        'bool': {
            'filter': filter_clause,
            'must': [
                {
                    'nested': {
                        'path': path,
                        'score_mode': 'max',
                        'query': {'knn': {space.vector_field: {'vector': vector, 'k': k}}},
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


def no_box_vectors_clause(space: ResolvedSpace) -> dict[str, Any]:
    """Items with no box vector at all in ``space``: the box search cannot see them."""
    return {
        'bool': {
            'must_not': [
                {
                    'nested': {
                        'path': _nested(space),
                        'query': {'exists': {'field': space.vector_field}},
                    }
                }
            ]
        }
    }


def box_space_source_includes(space: ResolvedSpace) -> list[str]:
    """The ``_source`` includes that read the box vectors of ``space`` back (the leaf paths: a
    search that names the nested field itself gets the number ``1`` instead of each vector on an
    index with derived source)."""
    path = _nested(space)
    return [f'{path}.box_id', f'{path}.bbox_norm', space.vector_field]


def box_source_excludes(excludes: list[str], space: ResolvedSpace) -> list[str]:
    """``excludes`` with the box entries of ``space`` readable but for their vector: hydration
    needs each entry's ``box_id`` and ``bbox_norm`` (to name the match and judge it stale),
    never the vector."""
    path = _nested(space)
    return [*[e for e in excludes if e != path], space.vector_field]


def stored_box_entries(source: dict[str, Any], space: ResolvedSpace) -> list[dict[str, Any]]:
    """The stored box vector entries of ``space`` on an item source."""
    entries = source.get(_nested(space)) or []
    return [e for e in entries if isinstance(e, dict)]


def matched_boxes(
    hit: dict[str, Any], fields: RegionFields, space: ResolvedSpace
) -> list[tuple[str, bool]]:
    """``[(box_id, stale)]`` for the boxes that matched in ``space``, best first.

    ``stale`` is true when the vector was computed from geometry the box no longer has (or the
    box is gone). Absent ``inner_hits`` yields an empty list, not an error."""
    source = hit.get('_source') or {}
    entries = source.get(_nested(space)) or []
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
    space: ResolvedSpace,
    *,
    skip: tuple[str, str] | None = None,
) -> bool:
    """Set ``matched_box_id`` / ``matched_box_stale`` on the wire ``item``.

    ``skip`` is ``(crop_id, box_id)`` of the query's own box: that box is never the match, so an
    item whose only match is the query box is dropped (returns False)."""
    candidates = matched_boxes(hit, fields, space)
    if skip is not None and hit.get('_id') == skip[0]:
        candidates = [c for c in candidates if c[0] != skip[1]]
        if not candidates:
            return False
    box_id, stale = candidates[0] if candidates else (None, None)
    item['matched_box_id'] = box_id
    item['matched_box_stale'] = stale
    return True
