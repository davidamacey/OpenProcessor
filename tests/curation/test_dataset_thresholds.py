"""Per-class dataset thresholds are backend constants, served as data.

The frontend's adequacy tiers (ok/low/critical at 500/100), the export
test-holdout minimum (5) and the augmentation target clamp (500..3000)
now come from ``src/services/curation/dataset_thresholds.py`` — the same
constants the training preflight enforces — on ``/train/preflight``,
``/stats/classes``, ``/classes`` and ``/test_holdout/stats``.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.services.curation import dataset_thresholds as T  # noqa: N812


EXPECTED = {
    'block_below': T.HARD_MIN_CROPS_PER_CLASS,
    'warn_below': T.WARN_MIN_CROPS_PER_CLASS,
    'min_test_per_class': T.MIN_TEST_CROPS_PER_CLASS,
    'aug_target_min': T.AUG_TARGET_MIN,
    'aug_target_max': T.AUG_TARGET_MAX,
}


@pytest.mark.parametrize(
    ('n', 'tier'),
    [(0, 'block'), (19, 'block'), (20, 'warn'), (499, 'warn'), (500, 'ok'), (9000, 'ok')],
)
def test_adequacy(n: int, tier: str) -> None:
    assert T.adequacy(n) == tier


@pytest.mark.parametrize(('n', 'target'), [(0, 500), (800, 800), (5000, 3000)])
def test_aug_target(n: int, target: int) -> None:
    assert T.aug_target(n) == target


def test_preflight_uses_the_same_constants() -> None:
    from src.routers import curation_train
    from src.services.curation import holdout

    assert curation_train.HARD_MIN_CROPS_PER_CLASS == T.HARD_MIN_CROPS_PER_CLASS
    assert curation_train.WARN_MIN_CROPS_PER_CLASS == T.WARN_MIN_CROPS_PER_CLASS
    assert curation_train.MIN_TEST_CROPS_PER_CLASS == T.MIN_TEST_CROPS_PER_CLASS
    assert holdout.MIN_TEST_PER_CLASS == T.MIN_TEST_CROPS_PER_CLASS
    assert T.dataset_thresholds() == EXPECTED


class _Entry:
    def __init__(self, cid: int, name: str) -> None:
        self.class_id = cid
        self.class_name = name
        self.group = ''
        self.deprecated = False
        self.sample_count = 0
        self.validated_count = 0
        self.hotkey_letter = None


class _Reg:
    classes = [_Entry(1, 'a'), _Entry(2, 'b'), _Entry(3, 'c')]

    def load(self) -> _Reg:
        return self


def _client(monkeypatch: pytest.MonkeyPatch, search_resp: dict[str, Any]) -> TestClient:
    from src.routers.curation import _raw_opensearch_dep, router as curation_router

    fake = AsyncMock()
    fake.search = AsyncMock(return_value=search_resp)
    fake.count = AsyncMock(return_value={'count': 0})
    monkeypatch.setattr('src.routers.curation.get_class_registry', lambda: _Reg())
    app = FastAPI()
    app.include_router(curation_router)
    app.dependency_overrides[_raw_opensearch_dep] = lambda: fake
    return TestClient(app)


_BY_CLASS = {
    'aggregations': {
        'by_class': {
            'buckets': [
                {'key': 1, 'doc_count': 900, 'validated': {'doc_count': 600}},
                {'key': 2, 'doc_count': 300, 'validated': {'doc_count': 150}},
                {'key': 3, 'doc_count': 30, 'validated': {'doc_count': 10}},
            ]
        },
        'by_cluster': {'buckets': []},
    }
}


def test_stats_classes_serves_adequacy_and_aug_target(monkeypatch: pytest.MonkeyPatch) -> None:
    r = _client(monkeypatch, _BY_CLASS).get('/curation/stats/classes')
    assert r.status_code == 200, r.text
    body = r.json()
    assert body['thresholds'] == EXPECTED
    rows = {c['class_id']: c for c in body['classes']}
    assert (rows[1]['adequacy'], rows[1]['aug_target'], rows[1]['aug_gap']) == ('ok', 600, 0)
    assert (rows[2]['adequacy'], rows[2]['aug_target'], rows[2]['aug_gap']) == ('warn', 500, 350)
    assert rows[3]['adequacy'] == 'block'


def test_classes_list_serves_adequacy(monkeypatch: pytest.MonkeyPatch) -> None:
    r = _client(monkeypatch, _BY_CLASS).get('/curation/classes')
    assert r.status_code == 200, r.text
    body = r.json()
    assert body['thresholds'] == EXPECTED
    assert {c['class_id']: c['adequacy'] for c in body['classes']} == {
        1: 'ok',
        2: 'warn',
        3: 'block',
    }


def test_holdout_stats_flags_deficient_classes(monkeypatch: pytest.MonkeyPatch) -> None:
    resp = {
        'hits': {'total': {'value': 12}},
        'aggregations': {
            'by_class': {'buckets': [{'key': 1, 'doc_count': 9}, {'key': 2, 'doc_count': 3}]}
        },
    }
    r = _client(monkeypatch, resp).get('/curation/test_holdout/stats')
    assert r.status_code == 200, r.text
    body = r.json()
    assert body['min_test_per_class'] == T.MIN_TEST_CROPS_PER_CLASS
    assert {b['key']: b['deficient'] for b in body['by_class']} == {1: False, 2: True}
