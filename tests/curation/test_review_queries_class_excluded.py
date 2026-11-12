"""Regression: ``class_excluded`` items must never reappear in any
``/review/{tab}`` queue.

Structural check (every known tab's ``must_not`` carries the clause,
regardless of whether that tab overrides the shared base) plus one
match-semantics check (``outliers``) proving an excluded item that would
otherwise clearly qualify for the queue is filtered out end-to-end.
"""

from __future__ import annotations

import pytest

from curation.query_fakes import matches
from src.services.curation.review_queries import build_tab_query
from src.services.curation.review_tab_clauses import SIMPLE_TABS


_CLASS_EXCLUDED_CLAUSE = {'term': {'class_excluded': True}}


def _guarded(must: list, must_not: list) -> bool:
    """The clause is in ``must_not``, or every source of a union in ``must`` carries it."""
    if _CLASS_EXCLUDED_CLAUSE in must_not:
        return True
    sources = [s for clause in must for s in clause.get('bool', {}).get('should', [])]
    return bool(sources) and all(
        _guarded(s['bool'].get('must', []), s['bool'].get('must_not', [])) for s in sources
    )


@pytest.mark.parametrize('tab', [*SIMPLE_TABS, 'regions'])
def test_every_tab_excludes_class_excluded_items(tab: str) -> None:
    must, must_not, _reason = build_tab_query(
        tab, vector_field='pe_embedding', include_test=False, text=None, max_rank=None
    )
    assert _guarded(must, must_not), (
        f'tab {tab!r} must_not is missing the class_excluded guard: {must_not}'
    )


def test_excluded_outlier_item_disappears_from_the_outliers_tab() -> None:
    must, must_not, _reason = build_tab_query(
        'outliers', vector_field='pe_embedding', include_test=False, text=None, max_rank=None
    )
    query = {'bool': {'must': must, 'must_not': must_not}}

    # A textbook outlier: far from its cluster centroid.
    outlier_doc = {'cluster_score': 0.1, 'class_validated': False}
    assert matches(outlier_doc, query)

    excluded_outlier_doc = {**outlier_doc, 'class_excluded': True}
    assert not matches(excluded_outlier_doc, query)
