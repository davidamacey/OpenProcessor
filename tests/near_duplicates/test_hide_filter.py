"""``hide_near_duplicates`` keeps a representative and an ungrouped document and hides the rest.

Both consumers (the review list and the semantic search) are run against the same documents; a
tiny matcher reads the clause they build, so the test follows the clause, not its spelling.
"""

from __future__ import annotations

from typing import Any

import pytest
from _embedding_registry_fake import DEFAULT_WRITE_FIELDS

from src.services.curation.semantic_search import _build_filter


DOCS = {
    'representative': {'near_dup_is_representative': True},
    'member': {'near_dup_is_representative': False},
    'ungrouped': {},
}


def matches(clause: dict[str, Any], doc: dict[str, Any]) -> bool:
    if 'term' in clause:
        ((field, value),) = clause['term'].items()
        return doc.get(field) == value
    if 'exists' in clause:
        return doc.get(clause['exists']['field']) is not None
    if 'bool' in clause:
        body = clause['bool']
        must = body.get('must', []) + body.get('filter', [])
        must_not = body.get('must_not', [])
        should = body.get('should', [])
        if any(matches(c, doc) for c in ([must_not] if isinstance(must_not, dict) else must_not)):
            return False
        if not all(matches(c, doc) for c in must):
            return False
        return not should or any(matches(c, doc) for c in should)
    raise NotImplementedError(clause)


def kept(clauses: list[dict[str, Any]]) -> set[str]:
    return {name for name, doc in DOCS.items() if all(matches(c, doc) for c in clauses)}


def _search_clauses(hide: bool) -> list[dict[str, Any]]:
    return _build_filter(
        embedding=DEFAULT_WRITE_FIELDS,
        tab=None,
        cluster_uid=None,
        date_from=None,
        date_to=None,
        max_rank=None,
        min_blur_ratio=None,
        hide_near_duplicates=hide,
        include_test=True,
    )


async def _review_clauses(hide: bool) -> list[dict[str, Any]]:
    # imported here: review_request imports through review_filter_specs, which imports it back,
    # and only resolves when another curation module has been imported first
    from src.services.curation.review_request import ReviewFilters, build_review_request

    request = await build_review_request(
        'all', ReviewFilters(hide_near_duplicates=hide, include_test=True), None, None
    )
    bool_query = request.query['bool']
    return [c for c in bool_query.get('must', []) if 'near_dup_is_representative' in str(c)]


@pytest.mark.asyncio
async def test_the_review_list_hides_a_non_representative_member() -> None:
    assert kept(await _review_clauses(True)) == {'representative', 'ungrouped'}


def test_the_semantic_search_hides_a_non_representative_member() -> None:
    clauses = [c for c in _search_clauses(True) if 'near_dup_is_representative' in str(c)]
    assert kept(clauses) == {'representative', 'ungrouped'}


@pytest.mark.asyncio
async def test_without_the_filter_nothing_is_hidden() -> None:
    assert await _review_clauses(False) == []
    assert not [c for c in _search_clauses(False) if 'near_dup_is_representative' in str(c)]
