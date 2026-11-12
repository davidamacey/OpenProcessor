"""The exclude reasons are a served vocabulary (`GET /exclude_reasons`), and the exclude
write accepts exactly those: an unknown reason is refused, `not_an_object` is a person's soft
delete of a box that was never an object."""

from __future__ import annotations

from typing import Any, get_args

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from curation.query_fakes import QueryFakeOpenSearch
from src.config.curation_accessors import base_curation_config
from src.domain.item_geometry import NOT_AN_OBJECT
from src.schemas.curation.exclusion import ExcludeReason


ITEMS = base_curation_config().items_index
BASE = '/curation/projects/default'


def _client(fake: Any) -> TestClient:
    from _curation_app import mount_curation_routers

    from src.routers.curation import _raw_opensearch_dep, router as curation_router

    app = FastAPI()
    mount_curation_routers(app, curation_router)
    app.dependency_overrides[_raw_opensearch_dep] = lambda: fake
    return TestClient(app)


def test_reasons_are_served() -> None:
    resp = _client(QueryFakeOpenSearch()).get(f'{BASE}/exclude_reasons')
    assert resp.status_code == 200
    reasons = resp.json()['reasons']
    assert [r['id'] for r in reasons] == list(get_args(ExcludeReason))
    assert all(r['label'].strip() for r in reasons)
    assert NOT_AN_OBJECT in {r['id'] for r in reasons}


def test_unknown_reason_is_refused() -> None:
    fake = QueryFakeOpenSearch({ITEMS: {'a': {'crop_id': 'a'}}})
    resp = _client(fake).post(
        f'{BASE}/crops/batch_exclude', json={'crop_ids': ['a'], 'reason': 'because'}
    )
    assert resp.status_code == 422
    assert 'class_excluded' not in fake.docs(ITEMS)['a']


@pytest.mark.parametrize('reason', get_args(ExcludeReason))
def test_every_served_reason_is_accepted_and_recorded(reason: str) -> None:
    fake = QueryFakeOpenSearch({ITEMS: {'a': {'crop_id': 'a'}}})
    resp = _client(fake).post(
        f'{BASE}/crops/batch_exclude', json={'crop_ids': ['a'], 'reason': reason}
    )
    assert resp.status_code == 200, resp.text
    doc = fake.docs(ITEMS)['a']
    assert (doc['class_excluded'], doc['excluded_reason'], doc['excluded_by']) == (
        True,
        reason,
        'human',
    )
