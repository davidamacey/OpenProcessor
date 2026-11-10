"""Plate-text search on a box: exact, fuzzy and fragment match on the folded text.

The query is folded with the active profile's confusable map (``src/domain/region_text.py``), so
it is compared with the stored ``text_fold`` key in the same alphabet. Exact is a ``term`` and
fuzzy a ``fuzzy`` query on the ``text_fold`` keyword; fragment is a ``wildcard`` query on its
``text_fold.sub`` wildcard sub-field (the type indexes substrings for ``*q*``).

OpenSearch: https://docs.opensearch.org/latest/query-dsl/term/fuzzy/ (``fuzziness``,
``transpositions``, ``prefix_length``, ``max_expansions``),
https://docs.opensearch.org/latest/mappings/supported-field-types/wildcard/ (the wildcard field
supports ``wildcard`` queries and a normalizer, not fuzzy).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from src.config import get_region_fields
from src.config.curation_accessors import items_index
from src.domain.region_boxes import read_boxes
from src.domain.region_text import fold


if TYPE_CHECKING:
    from collections.abc import Mapping

TextMatch = Literal['exact', 'fuzzy', 'fragment']

FUZZY_FUZZINESS = 1
FUZZY_MAX_EXPANSIONS = 50


def _escape_wildcard(value: str) -> str:
    return value.replace('\\', '\\\\').replace('*', '\\*').replace('?', '\\?')


async def same_text_key(opensearch: Any, ref: str) -> str:
    """The folded text of the box ``ref`` (``<item_id>:<box_id>``) names, read server-side.

    Raises ``ValueError`` for a malformed ref and ``LookupError`` when the item or box is
    missing or the box has no text.
    """
    item_id, _, box_id = ref.partition(':')
    if not item_id or not box_id:
        msg = 'same_text_as must be <item_id>:<box_id>'
        raise ValueError(msg)
    boxes_field = get_region_fields().boxes
    resp = await opensearch.mget(
        index=items_index(), body={'ids': [item_id]}, _source_includes=[boxes_field]
    )
    [doc] = resp.get('docs') or [{}]
    source = doc.get('_source') or {} if doc.get('found') else {}
    for box in read_boxes(source):
        if box.box_id == box_id and box.text_fold:
            return box.text_fold
    msg = f'no text on box {ref!r}'
    raise LookupError(msg)


def region_text_match_clause(
    boxes_field: str, text: str, match: TextMatch, confusables: Mapping[str, str]
) -> dict[str, Any]:
    """The clause for one box (to be wrapped by ``box_query``); ``boxes_field`` is the nested
    list's path. Raises ``ValueError`` when ``text`` has no letter or digit left to match."""
    key = fold(text, confusables)
    if not key:
        msg = 'text must contain a letter or digit'
        raise ValueError(msg)
    field = f'{boxes_field}.text_fold'
    if match == 'exact':
        return {'term': {field: key}}
    if match == 'fuzzy':
        return {
            'fuzzy': {
                field: {
                    'value': key,
                    'fuzziness': FUZZY_FUZZINESS,
                    'transpositions': True,
                    'prefix_length': 0,
                    'max_expansions': FUZZY_MAX_EXPANSIONS,
                }
            }
        }
    return {'wildcard': {f'{field}.sub': {'value': f'*{_escape_wildcard(key)}*'}}}
