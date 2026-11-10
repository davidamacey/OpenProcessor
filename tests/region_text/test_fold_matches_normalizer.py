"""PT-F2: the domain match keys agree with the ``op_text_compact`` index normalizer.

``EXPECTED`` is what OpenSearch's ``_analyze`` returns for ``op_text_compact`` (a
``pattern_replace`` char filter on non-letters and non-digits, then ``asciifolding``, then
``uppercase``): the analysis chain is documented at
https://docs.opensearch.org/latest/analyzers/normalizers/. All strings are synthetic and 12
characters long once compacted, longer than any plate format.
"""

from __future__ import annotations

import pytest

from src.clients.curation_opensearch.text_analysis import ITEMS_ANALYSIS, NORMALIZER_NAME
from src.domain.region_text import NON_ALNUM_PATTERN, confusables_issue, fold, normalize


EXPECTED = [
    ('abc 123 xyz 987', 'ABC123XYZ987'),
    ('ABC-123-XYZ-987', 'ABC123XYZ987'),
    ('abc.123/xyz_987', 'ABC123XYZ987'),
    ('Abç 123 xÿz 987', 'ABC123XYZ987'),
    ('  QWE 456 RTY 012  ', 'QWE456RTY012'),
    ('é1è2ê3ë4à5â6', 'E1E2E3E4A5A6'),
]


@pytest.mark.parametrize(('raw', 'analyzed'), EXPECTED)
def test_normalize_matches_the_index_normalizer(raw: str, analyzed: str) -> None:
    assert normalize(raw) == analyzed


def test_the_index_normalizer_is_the_documented_chain() -> None:
    normalizer = ITEMS_ANALYSIS['normalizer'][NORMALIZER_NAME]
    assert normalizer['filter'] == ['asciifolding', 'uppercase']
    [char_filter] = normalizer['char_filter']
    assert ITEMS_ANALYSIS['char_filter'][char_filter]['pattern'] == NON_ALNUM_PATTERN


def test_fold_applies_the_confusables_after_the_compact_form() -> None:
    assert fold('abo 1l2 xyz 987', {'O': '0', 'L': '1'}) == 'AB0112XYZ987'


def test_fold_is_idempotent() -> None:
    confusables = {'O': '0', 'I': '1', 'B': '8'}
    once = fold('OIB OIB OIB OIB', confusables)
    assert fold(once, confusables) == once == '018018018018'


@pytest.mark.parametrize(
    ('confusables', 'charset', 'bad'),
    [
        ({'OO': '0'}, '', True),
        ({'O': '0', '0': 'Q'}, '', True),
        ({'O': 'a'}, '[A-Z0-9]', True),
        ({'O': '0', 'I': '1'}, '[A-Z0-9]', False),
    ],
)
def test_confusables_validation(confusables: dict[str, str], charset: str, bad: bool) -> None:
    assert (confusables_issue(confusables, charset) is not None) is bad


@pytest.mark.parametrize(('raw', 'analyzed'), EXPECTED)
def test_the_key_is_stable_under_respacing_and_recasing(raw: str, analyzed: str) -> None:
    assert normalize(normalize(raw)) == analyzed
    assert normalize(raw.lower()) == normalize(raw.upper()) == analyzed
    assert normalize(' '.join(raw)) == analyzed
    assert analyzed.isalnum()
    assert analyzed == analyzed.upper()
    assert len(analyzed) == 12


def test_a_confusable_variant_folds_to_the_same_key_as_the_original() -> None:
    confusables = {'O': '0', 'I': '1'}

    assert fold('QWO-4I6 RTY 012', confusables) == fold('QW0-416 RTY 012', confusables)
    assert fold('QWO-4I6 RTY 012', {}) != fold('QW0-416 RTY 012', {})
