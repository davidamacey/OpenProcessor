"""Tests for the model-disagreement source of the ``disagreements`` review queue (Phase 5).

We mount the curation router on a minimal FastAPI app and stub OpenSearch
with an AsyncMock. The tests verify:

* GET /curation/review/disagreements builds the right OpenSearch body: the model source
  keeps validated items (class_validated=true, probe_pred_class exists, painless script for
  inequality) while the other two sources keep the unvalidated rule.
* Probe fields surface in the response items (probe_pred_class,
  probe_pred_entropy).
* A tab id that is not a queue answers 404 not_a_queue.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.fixture
def app_client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """Mount the real curation router with OpenSearch stubbed."""
    from src.routers.curation import _raw_opensearch_dep, router as curation_router

    fake_os = AsyncMock()

    # Default search response: one validated crop where the new model
    # disagrees with the human label.
    fake_os.search = AsyncMock(
        return_value={
            'hits': {
                'total': {'value': 1},
                'hits': [
                    {
                        '_id': 'crop-disagree-1',
                        '_source': {
                            'crop_id': 'crop-disagree-1',
                            'image_path': '/data/images/x.jpg',
                            'bbox_norm': [0.1, 0.1, 0.5, 0.5],
                            'class_id': 12,
                            'class_name': 'pickup',
                            'class_source': 'human',
                            'confidence': 1.0,
                            'label_source': 'human',
                            'class_validated': True,
                            'probe_pred_class': 'sedan',
                            'probe_pred_entropy': 0.42,
                            'updated_at': '2026-05-09T01:00:00Z',
                        },
                    }
                ],
            }
        }
    )

    # Make ensure_indexes a no-op so we never touch opensearch.indices.
    monkeypatch.setattr(
        'src.services.curation.index_bootstrap.ensure_indexes',
        AsyncMock(return_value=None),
    )

    app = FastAPI()
    from _curation_app import mount_curation_routers

    mount_curation_routers(app, curation_router)
    app.dependency_overrides[_raw_opensearch_dep] = lambda: fake_os

    client = TestClient(app)
    client.fake_os = fake_os  # type: ignore[attr-defined]
    return client


URL = '/curation/projects/default/review'


def _sources(client: TestClient, path: str = 'disagreements', query: str = '') -> list[Any]:
    """The sub-queries of the queue's union, one per source."""
    r = client.get(f'{URL}/{path}{query}')
    assert r.status_code == 200, r.text
    body = client.fake_os.search.call_args.kwargs['body']  # type: ignore[attr-defined]
    return body['query']['bool']['must'][0]['bool']['should']


def _model_source(client: TestClient, query: str = '') -> dict[str, Any]:
    return next(
        src['bool'] for src in _sources(client, query=query) if 'probe_pred_class' in str(src)
    )


def test_model_source_includes_class_validated_and_probe_exists(app_client: TestClient) -> None:
    """The model source filters on class_validated=True (the class-side flag) rather than the
    legacy conflated label_validated."""
    must_strs = [str(m) for m in _model_source(app_client)['must']]
    assert any("'class_validated': True" in s for s in must_strs)
    assert any("'probe_pred_class'" in s and 'exists' in s for s in must_strs)
    # Painless script enforces inequality.
    assert any('probe_pred_class' in s and 'painless' in s for s in must_strs)


def test_model_source_script_uses_keyword_fields(app_client: TestClient) -> None:
    """Regression guard: ``probe_pred_class`` and ``class_name`` are mapped ``keyword`` *directly*
    on the live index, so there is no ``.keyword`` sub-field to fall back on (a `text` mapping was
    assumed once and caused ``illegal_argument_exception: Fielddata is disabled on text fields``).
    The mocked-OpenSearch tests only assert on the outgoing body; the script itself ran live.
    """
    script_clause = next(m for m in _model_source(app_client)['must'] if 'script' in m)
    source = script_clause['script']['script']['source']
    # A non-existent ``.keyword`` sub-field must NOT appear (that was the bug).
    assert "doc['probe_pred_class.keyword']" not in source
    assert "doc['class_name.keyword']" not in source
    assert "doc['probe_pred_class']" in source
    assert "doc['class_name']" in source


def test_only_the_model_source_keeps_validated_items(app_client: TestClient) -> None:
    """The default must_not excludes class_validated=true; the model source inverts it and the
    other sources keep it."""
    for src in _sources(app_client):
        excludes_validated = {'term': {'class_validated': True}} in src['bool']['must_not']
        assert excludes_validated is ('probe_pred_class' not in str(src))


def test_every_source_excludes_test_holdout_by_default(app_client: TestClient) -> None:
    for src in _sources(app_client):
        assert {'term': {'test_holdout': True}} in src['bool']['must_not']


def test_include_test_drops_the_holdout_filter(app_client: TestClient) -> None:
    for src in _sources(app_client, query='?include_test=true'):
        assert {'term': {'test_holdout': True}} not in src['bool']['must_not']


def test_response_surfaces_probe_fields(app_client: TestClient) -> None:
    r = app_client.get(f'{URL}/disagreements')
    assert r.status_code == 200
    out = r.json()
    assert out['total'] == 1
    item = out['items'][0]
    assert item['class_name'] == 'pickup'
    assert item['probe_pred_class'] == 'sedan'
    assert item['probe_pred_entropy'] == 0.42
    assert item['reason'] == 'new model disagrees with the validated label'


def test_a_tab_that_is_not_a_queue_answers_404(app_client: TestClient) -> None:
    r = app_client.get(f'{URL}/totally-not-a-tab')
    assert r.status_code == 404
    assert r.json()['detail']['error'] == 'not_a_queue'
    assert 'disagreements' in r.json()['detail']['message']


def test_other_queues_still_exclude_validated(app_client: TestClient) -> None:
    """Smoke check: the standard queues keep the class_validated exclusion."""
    for queue in ('to_label', 'check_machine'):
        r = app_client.get(f'{URL}/{queue}')
        assert r.status_code == 200
        body = app_client.fake_os.search.call_args.kwargs['body']  # type: ignore[attr-defined]
        assert {'term': {'class_validated': True}} in body['query']['bool']['must_not']
