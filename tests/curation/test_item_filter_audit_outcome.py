"""``audit_outcome``: the audited crops whose human verdict was agree, detector_wrong, vlm_wrong
or both_wrong, on the shared item filter and as a query parameter of ``GET /crops``."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from _embedding_registry_fake import DEFAULT_WRITE_FIELDS
from pydantic import ValidationError

from curation._cluster_cards_helpers import curation_client
from curation.query_fakes import QueryFakeOpenSearch
from src.config.curation_accessors import base_curation_config
from src.services.curation.item_filter import ItemFilter, item_filter_clauses


if TYPE_CHECKING:
    from fastapi.testclient import TestClient

ITEMS = base_curation_config().items_index
URL = '/curation/projects/default'


def _doc(crop_id: str, outcome: str | None, **kw: Any) -> dict[str, Any]:
    return {
        'crop_id': crop_id,
        'class_name': 'car',
        'class_validated': outcome is not None,
        'audit_sample': outcome is not None,
        'audit_outcome': outcome,
        **kw,
    }


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    docs = [
        _doc('agree', 'agree'),
        _doc('det', 'detector_wrong'),
        _doc('vlm', 'vlm_wrong'),
        _doc('both', 'both_wrong'),
        _doc('waiting', None, audit_sample=True),  # drawn, no verdict yet
        _doc('undone', 'agree', class_validated=False),  # the verdict was undone
        _doc('plain', None),
    ]
    return curation_client(
        QueryFakeOpenSearch({ITEMS: {d['crop_id']: d for d in docs}}), monkeypatch
    )


def _ids(client: TestClient, *outcomes: str) -> set[str]:
    resp = client.get(f'{URL}/crops', params={'audit_outcome': list(outcomes), 'page_size': 50})
    assert resp.status_code == 200, resp.text
    return {i['crop_id'] for i in resp.json()['crops']}


def test_the_clause_is_the_reports_own_verdict_set_narrowed_to_the_outcomes() -> None:
    (clause,) = item_filter_clauses(
        ItemFilter(audit_outcome=['agree', 'vlm_wrong']), embedding=DEFAULT_WRITE_FIELDS
    )
    filters = clause['bool']['filter']
    assert {'term': {'audit_sample': True}} in filters
    assert {'term': {'class_validated': True}} in filters
    assert {'terms': {'audit_outcome': ['agree', 'vlm_wrong']}} in filters


def test_an_unknown_outcome_is_refused() -> None:
    with pytest.raises(ValidationError):
        ItemFilter(audit_outcome=['sort_of_right'])


def test_one_outcome_lists_its_crops(client: TestClient) -> None:
    assert _ids(client, 'detector_wrong') == {'det'}


def test_several_outcomes_are_an_or(client: TestClient) -> None:
    assert _ids(client, 'agree', 'both_wrong') == {'agree', 'both'}


def test_an_undone_verdict_and_an_unlabelled_sample_are_not_listed(client: TestClient) -> None:
    every = _ids(client, 'agree', 'detector_wrong', 'vlm_wrong', 'both_wrong')
    assert every == {'agree', 'det', 'vlm', 'both'}


def test_the_query_parameter_rejects_an_unknown_outcome(client: TestClient) -> None:
    resp = client.get(f'{URL}/crops', params={'audit_outcome': 'sort_of_right'})
    assert resp.status_code == 422
