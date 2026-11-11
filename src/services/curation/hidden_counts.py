"""How many items the list pages hide, so a total lower than the dashboard's explains itself.

The frozen test holdout and human-ignored items stay out of every browse list unless asked for
(:func:`~src.services.curation.item_filter.visibility_clauses`). The counts follow the request:
they are the items that match its filter and are removed by those clauses. One size-0 ``filters``
aggregation counts them (https://docs.opensearch.org/latest/aggregations/bucket/filters/).
"""

from __future__ import annotations

from typing import Any

from src.config.curation_accessors import items_index


async def hidden_counts(
    opensearch: Any,
    filter_clauses: list[dict[str, Any]],
    *,
    include_test: bool,
    include_excluded: bool,
) -> dict[str, int]:
    """``{test_holdout, excluded}`` among the items matching ``filter_clauses``; a kind the page
    includes counts 0. A failed count raises: zero would read as nothing hidden."""
    named: dict[str, Any] = {}
    if not include_test:
        named['test_holdout'] = {'term': {'test_holdout': True}}
    if not include_excluded:
        named['excluded'] = {'term': {'class_excluded': True}}
    counts = {'test_holdout': 0, 'excluded': 0}
    if not named:
        return counts
    resp = await opensearch.search(
        index=items_index(),
        body={
            'size': 0,
            'query': {'bool': {'filter': filter_clauses}} if filter_clauses else {'match_all': {}},
            'aggs': {'hidden': {'filters': {'filters': named}}},
        },
    )
    buckets = ((resp.get('aggregations') or {}).get('hidden') or {}).get('buckets') or {}
    for key in named:
        counts[key] = int((buckets.get(key) or {}).get('doc_count', 0))
    return counts
