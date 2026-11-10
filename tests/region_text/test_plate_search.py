"""PT-F3: exact, fuzzy and fragment plate-text clauses on the folded text key."""

from __future__ import annotations

import pytest

from src.routers.curation.regions import _box_filter_parts
from src.services.curation.region_text_search import TextMatch, region_text_match_clause


BOXES = 'region_boxes'
CONFUSABLES = {'O': '0', 'I': '1'}


def test_exact_is_a_term_on_the_folded_query() -> None:
    clause = region_text_match_clause(BOXES, 'abo-1i2 xyz 987', 'exact', CONFUSABLES)

    assert clause == {'term': {'region_boxes.text_fold': 'AB0112XYZ987'}}


def test_fuzzy_allows_one_edit_with_transpositions() -> None:
    clause = region_text_match_clause(BOXES, 'ABC123XYZ987', 'fuzzy', {})

    assert clause == {
        'fuzzy': {
            'region_boxes.text_fold': {
                'value': 'ABC123XYZ987',
                'fuzziness': 1,
                'transpositions': True,
                'prefix_length': 0,
                'max_expansions': 50,
            }
        }
    }


def test_fragment_is_a_wildcard_on_the_sub_field() -> None:
    clause = region_text_match_clause(BOXES, 'c12 3x', 'fragment', {})

    assert clause == {'wildcard': {'region_boxes.text_fold.sub': {'value': '*C123X*'}}}


def test_wildcard_metacharacters_in_the_query_never_widen_the_match() -> None:
    clause = region_text_match_clause(BOXES, 'AB*?CD\\EF', 'fragment', {})

    assert clause['wildcard']['region_boxes.text_fold.sub']['value'] == '*ABCDEF*'


def test_a_query_without_letters_or_digits_is_refused() -> None:
    with pytest.raises(ValueError, match='letter or digit'):
        region_text_match_clause(BOXES, ' -*? ', 'exact', {})


def _parts(text_match: TextMatch | None) -> list[dict]:
    return _box_filter_parts(
        status=None,
        box_state=None,
        cluster_uid=None,
        min_score=None,
        max_score=None,
        detector=None,
        text='abo 123',
        text_match=text_match,
        confusables=CONFUSABLES,
    )


def test_the_regions_text_filter_without_a_match_mode_is_unchanged() -> None:
    [*_, text_clause] = _parts(None)

    assert text_clause == {
        'wildcard': {'region_boxes.text': {'value': '*abo 123*', 'case_insensitive': True}}
    }


def test_the_regions_text_filter_with_a_match_mode_uses_the_same_builder() -> None:
    [*_, text_clause] = _parts('exact')

    assert text_clause == {'term': {'region_boxes.text_fold': 'AB0123'}}
