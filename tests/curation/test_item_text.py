"""Searchable per-item text: stored lines/tokens, the wire key, ``?item_text=``."""

from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.clients.curation_opensearch import _items_body
from src.routers.curation import _common
from src.services.curation.item_text import (
    ITEM_TEXT_LINES_FIELD,
    ITEM_TEXT_TOKENS_FIELD,
    item_text_query,
    item_text_update,
    normalize_tokens,
)
from src.services.curation.wire import serialize_item
from src.services.detection.region_text import OcrLine


LINES = [
    OcrLine('ABC-1234', (0.30, 0.60, 0.55, 0.70), 0.93),
    OcrLine('Smith Motors', (0.28, 0.72, 0.58, 0.76), 0.81),
    OcrLine('F150', (0.10, 0.20, 0.20, 0.26), 0.66),
    OcrLine('~~', (0.5, 0.5, 0.6, 0.6), 0.9),
    OcrLine('blurry', (0.7, 0.1, 0.8, 0.2), 0.3),
]


def _matches(tokens: list[str], clause: dict[str, Any]) -> bool:
    """Evaluate the prefix-filter clause the router builds against stored
    tokens, the way OpenSearch's ``prefix`` query does on a keyword."""
    filters = clause['bool']['filter']
    return all(
        any(t.startswith(f['prefix'][ITEM_TEXT_TOKENS_FIELD]['value']) for t in tokens)
        for f in filters
    )


class TestStoredShape:
    def test_lines_and_tokens(self) -> None:
        upd = item_text_update(LINES, min_confidence=0.5)
        texts = [ln['text'] for ln in upd[ITEM_TEXT_LINES_FIELD]]
        assert texts == ['ABC-1234', 'Smith Motors', 'F150']
        first = upd[ITEM_TEXT_LINES_FIELD][0]
        assert first['box_norm'] == [0.3, 0.6, 0.55, 0.7]
        assert first['confidence'] == 0.93
        assert first['rel_height'] == pytest.approx(0.1)
        assert upd[ITEM_TEXT_TOKENS_FIELD] == [
            '1234',
            'ABC',
            'ABC1234',
            'F150',
            'MOTORS',
            'SMITH',
            'SMITHMOTORS',
        ]

    def test_no_lines_writes_empty(self) -> None:
        assert item_text_update([], min_confidence=0.5) == {
            ITEM_TEXT_LINES_FIELD: [],
            ITEM_TEXT_TOKENS_FIELD: [],
        }

    def test_unicode_words(self) -> None:
        assert normalize_tokens('Müller_straße 5') == ['MÜLLER', 'STRASSE', '5']


class TestQuery:
    tokens = item_text_update(LINES, min_confidence=0.5)[ITEM_TEXT_TOKENS_FIELD]

    @pytest.mark.parametrize(
        ('q', 'hit'),
        [
            ('abc', True),
            ('ABC-1234', True),
            ('abc1234', True),
            ('123', True),
            ('smith mot', True),
            ('f15', True),
            ('xyz', False),
            ('smith xyz', False),
            ('blurry', False),
        ],
    )
    def test_case_insensitive_token_prefix(self, q: str, hit: bool) -> None:
        clause = item_text_query(q)
        assert clause is not None
        assert _matches(self.tokens, clause) is hit

    def test_query_without_words_is_none(self) -> None:
        assert item_text_query(' -- ') is None


class TestWireAndMapping:
    def test_item_text_lines_always_on_the_wire(self) -> None:
        assert serialize_item({}, 'x', api_prefix='')['item_text_lines'] == []
        stored = item_text_update(LINES[:1], min_confidence=0.5)
        item = serialize_item({**stored}, 'x', api_prefix='')
        assert item['item_text_lines'] == [
            {
                'text': 'ABC-1234',
                'box_norm': [0.3, 0.6, 0.55, 0.7],
                'confidence': 0.93,
                'rel_height': 0.1,
            }
        ]
        assert ITEM_TEXT_TOKENS_FIELD not in item

    def test_malformed_stored_value_is_empty(self) -> None:
        assert serialize_item({'item_text_lines': 'x'}, 'x', api_prefix='')['item_text_lines'] == []

    def test_explicit_mapping(self) -> None:
        props = _items_body()['mappings']['properties']
        lines = props[ITEM_TEXT_LINES_FIELD]
        assert lines['type'] == 'object'
        assert lines['properties']['text']['type'] == 'keyword'
        assert lines['properties']['box_norm']['type'] == 'float'
        assert props[ITEM_TEXT_TOKENS_FIELD]['type'] == 'keyword'


class _RecordingOS:
    def __init__(self) -> None:
        self.bodies: list[dict[str, Any]] = []

    async def search(self, *, index: str, body: dict[str, Any]) -> dict[str, Any]:  # noqa: ARG002
        json.dumps(body)
        self.bodies.append(body)
        return {'hits': {'total': {'value': 0}, 'hits': []}}


@pytest.fixture
def fake_os() -> _RecordingOS:
    return _RecordingOS()


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch, fake_os: _RecordingOS) -> Any:
    monkeypatch.setattr(_common, '_INDEXES_BOOTSTRAPPED', True)
    from src.routers.curation import _raw_opensearch_dep, router as curation_router

    app = FastAPI()
    app.include_router(curation_router)
    app.dependency_overrides[_raw_opensearch_dep] = lambda: fake_os
    with TestClient(app) as c:
        yield c


P = _common.config.api_prefix


class TestCropsFilter:
    def test_item_text_filter_is_applied(self, client: TestClient, fake_os: _RecordingOS) -> None:
        r = client.get(f'{P}/crops', params={'item_text': 'Smith mot'})
        assert r.status_code == 200, r.text
        filt = fake_os.bodies[-1]['query']['bool']['filter']
        assert item_text_query('Smith mot') in filt

    def test_no_filter_without_param(self, client: TestClient, fake_os: _RecordingOS) -> None:
        client.get(f'{P}/crops')
        assert ITEM_TEXT_TOKENS_FIELD not in json.dumps(fake_os.bodies[-1])

    def test_wordless_query_is_400(self, client: TestClient) -> None:
        r = client.get(f'{P}/crops', params={'item_text': '!!'})
        assert r.status_code == 400
