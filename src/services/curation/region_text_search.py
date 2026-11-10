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

from src.domain.region_text import fold


if TYPE_CHECKING:
    from collections.abc import Mapping

TextMatch = Literal['exact', 'fuzzy', 'fragment']

FUZZY_FUZZINESS = 1
FUZZY_MAX_EXPANSIONS = 50


def _escape_wildcard(value: str) -> str:
    return value.replace('\\', '\\\\').replace('*', '\\*').replace('?', '\\?')


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
