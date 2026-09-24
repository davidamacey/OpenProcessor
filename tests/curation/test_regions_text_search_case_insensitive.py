"""F-9 regression: ``GET /regions?text=`` must match case-insensitively and
escape wildcard metacharacters in user input, same as the review 'regions'
tab's already-correct helper (``review_queries.region_text_clause``).

The pre-fix query was ``{'wildcard': {F.text: f'*{text.upper()}*'}}`` —
correct only when the stored value happens to be uppercase, and unsafe
against a literal ``*``/``?`` in the search string.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from curation.query_fakes import QueryFakeOpenSearch
from src.config import IndexRole, get_curation_config, get_region_fields, index_name


F = get_region_fields()
CFG = get_curation_config()
ITEMS = index_name(CFG, IndexRole.ITEMS)

# GET /regions requires an active region profile (no-profile gating contract).
pytestmark = pytest.mark.usefixtures('reference_region_profile')


def _client(fake: Any) -> TestClient:
    from src.routers.curation import _raw_opensearch_dep, router as curation_router

    app = FastAPI()
    app.include_router(curation_router)
    app.dependency_overrides[_raw_opensearch_dep] = lambda: fake
    return TestClient(app)


def _fake(**docs: dict[str, Any]) -> QueryFakeOpenSearch:
    return QueryFakeOpenSearch({ITEMS: docs})


def _region_doc(text: str) -> dict[str, Any]:
    return {F.bbox_norm: [0.1, 0.1, 0.2, 0.2], F.text: text}


def test_region_text_search_matches_regardless_of_stored_or_query_case() -> None:
    fake = _fake(
        crop_lower={'crop_id': 'crop_lower', **_region_doc('abc-1234')},
        crop_upper={'crop_id': 'crop_upper', **_region_doc('XYZ-9999')},
        crop_mixed={'crop_id': 'crop_mixed', **_region_doc('AbC-1234')},
    )
    client = _client(fake)

    for query in ('abc', 'ABC', 'AbC'):
        resp = client.get('/curation/regions', params={'text': query})
        assert resp.status_code == 200, resp.text
        ids = {item['crop_id'] for item in resp.json()['items']}
        assert ids == {'crop_lower', 'crop_mixed'}, (query, ids)


def test_region_text_search_escapes_wildcard_metacharacters() -> None:
    fake = _fake(
        literal_star={'crop_id': 'literal_star', **_region_doc('AB*CD')},
        other={'crop_id': 'other', **_region_doc('ABXCD')},
    )
    client = _client(fake)

    resp = client.get('/curation/regions', params={'text': 'AB*CD'})
    assert resp.status_code == 200, resp.text
    ids = {item['crop_id'] for item in resp.json()['items']}
    # A literal '*' in the query must only match the doc that literally
    # contains '*', not act as a wildcard matching 'other' too.
    assert ids == {'literal_star'}, ids
