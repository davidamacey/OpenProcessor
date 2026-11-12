"""The review sorts a queue offers follow the data at once, on every worker.

``sort_options`` and the fallback of ``build_sort`` used to read a per-process 60 s cache of the
field coverage, so two API workers disagreed for up to a minute after items gained a field. They
now count against the index on each call; ``GET /methods`` keeps its cache.
"""

from __future__ import annotations

from typing import Any

import pytest

from src.services.curation import review_sorts
from src.services.curation.field_coverage import field_coverage


class _Index:
    """An items index whose ``cluster_score`` coverage can change between two calls."""

    def __init__(self) -> None:
        self.carrying = 0
        self.searches = 0

    async def search(self, *, index: str, body: dict[str, Any]) -> dict[str, Any]:
        self.searches += 1
        aggs = {f: {'doc_count': self.carrying} for f in body.get('aggs', {})}
        return {'hits': {'total': {'value': 10}}, 'aggregations': aggs}


def _ids(options: list[Any]) -> set[str]:
    return {o['id'] for o in options}


@pytest.mark.asyncio
async def test_a_sort_appears_as_soon_as_items_carry_its_field() -> None:
    index = _Index()
    assert 'atypicality' not in _ids(await review_sorts.sort_options('to_label', index))
    index.carrying = 4  # a clustering run wrote cluster_score; another worker sees it at once
    assert 'atypicality' in _ids(await review_sorts.sort_options('to_label', index))
    index.carrying = 0
    assert 'atypicality' not in _ids(await review_sorts.sort_options('to_label', index))


@pytest.mark.asyncio
async def test_the_sort_fallback_follows_the_data_at_once() -> None:
    index = _Index()
    _, applied, reason = await review_sorts.build_sort(None, tab='disagreements', opensearch=index)
    assert applied != 'disagreement_entropy_asc'
    assert reason
    index.carrying = 4
    _, applied, reason = await review_sorts.build_sort(None, tab='disagreements', opensearch=index)
    assert (applied, reason) == ('disagreement_entropy_asc', None)


@pytest.mark.asyncio
async def test_the_methods_lookup_still_serves_its_cache() -> None:
    index = _Index()
    index.carrying = 4
    assert (await field_coverage(index, frozenset({'cluster_score'})))['cluster_score'] == 4
    index.carrying = 0
    assert (await field_coverage(index, frozenset({'cluster_score'})))['cluster_score'] == 4
    assert index.searches == 1
