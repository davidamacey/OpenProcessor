"""The regions tab serves and honours the plate-text match mode."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

from src.services.curation import review_queries, review_tabs
from src.services.curation.review_filter_specs import FILTER_SPECS


if TYPE_CHECKING:
    from src.services.curation.region_text_search import TextMatch


@pytest.fixture(autouse=True)
def _text_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(review_tabs, '_region_text_enabled', lambda: True)


def test_region_text_match_spec_is_served() -> None:
    spec = FILTER_SPECS['region_text_match']
    assert spec['kind'] == 'enum'
    assert spec['allows_unset'] is True
    assert {o['value'] for o in spec['options']} == {'', 'exact', 'fuzzy', 'fragment'}
    assert all(o['label'] for o in spec['options'])
    assert 'region_text_match' in review_tabs.tab_filters('regions')


def test_region_text_match_is_dropped_with_text(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(review_tabs, '_region_text_enabled', lambda: False)
    assert 'region_text_match' not in review_tabs.tab_filters('regions')


def _regions_query(match: TextMatch | None) -> str:
    must, _must_not, _reason = review_queries.build_tab_query(
        'regions', include_test=False, text='AB 12', max_rank=None, region_text_match=match
    )
    return json.dumps(must)


@pytest.mark.parametrize(
    ('match', 'marker'),
    [('exact', '"term"'), ('fuzzy', '"fuzzy"'), ('fragment', '"wildcard"')],
)
def test_match_mode_reaches_the_regions_query(match: TextMatch, marker: str) -> None:
    assert marker in _regions_query(match)


def test_unset_keeps_the_substring_clause() -> None:
    q: Any = _regions_query(None)
    assert 'text_fold' not in q
