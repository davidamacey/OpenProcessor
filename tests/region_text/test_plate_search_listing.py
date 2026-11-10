"""PT-F3 on the crops listing: ``region_text``, ``region_text_match`` and ``same_text_as``,
and a behavioural check that the clauses find what the writer stored."""

from __future__ import annotations

import fnmatch
from typing import Any

import pytest
from curation.query_fakes import QueryFakeOpenSearch
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.config import IndexRole, get_region_fields, index_name
from src.config.curation_accessors import base_curation_config
from src.domain.region_text import match_fields
from src.services.curation.item_filter import ItemFilter, item_filter_clauses
from src.services.curation.region_text_search import region_text_match_clause, same_text_key
from tests.curation.test_item_filter_clause_golden import DEFAULT_WRITE_FIELDS


F = get_region_fields()
ITEMS = index_name(base_curation_config(), IndexRole.ITEMS)
CONFUSABLES = {'O': '0', 'I': '1'}
READING = 'abo-1i2 xyz 987'  # synthetic, 12 characters once compacted
STORED_FOLD = str(match_fields(READING, CONFUSABLES)['text_fold'])


def _osa_distance(a: str, b: str) -> int:
    d = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) + 1):
        d[i][0] = i
    for j in range(len(b) + 1):
        d[0][j] = j
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            cost = a[i - 1] != b[j - 1]
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                d[i][j] = min(d[i][j], d[i - 2][j - 2] + 1)
    return d[-1][-1]


def _matches(clause: dict[str, Any], stored: str) -> bool:
    """The documented semantics of the three queries, applied to one stored key."""
    [(kind, body)] = clause.items()
    [(_, spec)] = body.items()
    if kind == 'term':
        return stored == spec
    if kind == 'fuzzy':
        return _osa_distance(stored, spec['value']) <= spec['fuzziness']
    return fnmatch.fnmatchcase(stored, spec['value'])


@pytest.mark.parametrize(
    ('query', 'match', 'hit'),
    [
        ('AB0 1I2-xyz987', 'exact', True),
        ('ab0112xyz98', 'exact', False),
        ('ab0112xyz978', 'fuzzy', True),  # one transposition
        ('ab0112xyz9', 'fuzzy', False),  # three edits
        ('112 xy', 'fragment', True),
        ('xyz 988', 'fragment', False),
    ],
)
def test_the_clauses_find_what_the_writer_stored(query: str, match: str, hit: bool) -> None:
    clause = region_text_match_clause(F.boxes, query, match, CONFUSABLES)  # type: ignore[arg-type]

    assert _matches(clause, STORED_FOLD) is hit


def test_the_item_filter_wraps_the_clause_in_the_box_query() -> None:
    clauses = item_filter_clauses(
        ItemFilter(region_text='abo 1i2 xyz 987'), embedding=DEFAULT_WRITE_FIELDS
    )

    assert clauses == [
        {'nested': {'path': F.boxes, 'query': {'term': {f'{F.boxes}.text_fold': 'ABO1I2XYZ987'}}}}
    ]


def test_an_item_filter_text_without_letters_or_digits_is_refused() -> None:
    with pytest.raises(ValueError, match='letter or digit'):
        item_filter_clauses(ItemFilter(region_text=' -- '), embedding=DEFAULT_WRITE_FIELDS)


def _doc(**box: Any) -> dict[str, Any]:
    return {
        F.boxes: [{'box_id': 'b1', 'bbox_norm': [0.1, 0.1, 0.2, 0.2], 'state': 'accepted', **box}]
    }


@pytest.mark.asyncio
async def test_same_text_key_reads_the_stored_fold() -> None:
    fake = QueryFakeOpenSearch({ITEMS: {'i1': {'crop_id': 'i1', **_doc(text_fold='QWE456RTY012')}}})

    assert await same_text_key(fake, 'i1:b1') == 'QWE456RTY012'


@pytest.mark.asyncio
@pytest.mark.parametrize('ref', ['i1:b9', 'nope:b1'])
async def test_same_text_key_missing_box_or_item_is_a_lookup_error(ref: str) -> None:
    fake = QueryFakeOpenSearch({ITEMS: {'i1': {'crop_id': 'i1', **_doc(text_fold='QWE456RTY012')}}})

    with pytest.raises(LookupError):
        await same_text_key(fake, ref)


@pytest.mark.asyncio
async def test_same_text_key_refuses_a_malformed_ref() -> None:
    with pytest.raises(ValueError, match='<item_id>:<box_id>'):
        await same_text_key(QueryFakeOpenSearch({ITEMS: {}}), 'no-colon')


def test_same_text_as_with_region_text_is_400() -> None:
    from _curation_app import mount_curation_routers

    from src.routers.curation import _raw_opensearch_dep, router as curation_router

    app = FastAPI()
    mount_curation_routers(app, curation_router)
    app.dependency_overrides[_raw_opensearch_dep] = lambda: QueryFakeOpenSearch({ITEMS: {}})

    resp = TestClient(app).get('/curation/projects/default/crops?same_text_as=a:b&region_text=x')

    assert resp.status_code == 400
