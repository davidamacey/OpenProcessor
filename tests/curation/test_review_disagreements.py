"""Tests for the model_disagreements review tab (Phase 5).

We mount the curation router on a minimal FastAPI app and stub OpenSearch
with an AsyncMock. The tests verify:

* GET /curation/review/model_disagreements builds the right OpenSearch body
  (label_validated=true, probe_pred_class exists, painless script for
  inequality, override of must_not).
* Probe fields surface in the response items (probe_pred_class,
  probe_pred_entropy).
* Unknown tab returns 400 mentioning the new option.
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

    # Make _ensure_indexes a no-op so we never touch opensearch.indices.
    monkeypatch.setattr(
        'src.routers.curation._ensure_indexes',
        AsyncMock(return_value=None),
    )

    app = FastAPI()
    app.include_router(curation_router)
    app.dependency_overrides[_raw_opensearch_dep] = lambda: fake_os

    client = TestClient(app)
    client.fake_os = fake_os  # type: ignore[attr-defined]
    return client


def test_disagreements_query_includes_class_validated_and_probe_exists(
    app_client: TestClient,
) -> None:
    """Plan §1.3, A-PR2: model_disagreements filters on class_validated=True
    (the class-side flag) rather than the legacy conflated label_validated.
    """
    r = app_client.get('/curation/review/model_disagreements')
    assert r.status_code == 200, r.text

    body = app_client.fake_os.search.call_args.kwargs['body']  # type: ignore[attr-defined]
    must = body['query']['bool']['must']
    must_strs = [str(m) for m in must]
    assert any("'class_validated': True" in s for s in must_strs)
    assert any("'probe_pred_class'" in s and 'exists' in s for s in must_strs)
    # Painless script enforces inequality.
    assert any('probe_pred_class' in s and 'painless' in s for s in must_strs)


def test_disagreements_script_uses_keyword_subfields(app_client: TestClient) -> None:
    """Regression guard (Phase 11 audit-remediation): ``probe_pred_class``
    and ``class_name`` are mapped ``keyword`` *directly* on the live
    index — there is no ``.keyword`` sub-field to fall back on (a `text`
    mapping was assumed at one point; that assumption is what caused the
    original ``illegal_argument_exception: Fielddata is disabled on text
    fields`` failure). The fix was to use the bare field name against
    ``doc[...]`` directly, NOT to append ``.keyword`` (there's nothing
    there). This was invisible to the mocked-OpenSearch tests above (they
    only assert on the outgoing query body, never execute the script)
    because ``probe_pred_class`` had zero real coverage until Phase 11's
    probe backfill — the ``exists`` clause matched nothing, so the script
    never actually ran. Confirmed live against triton-opensearch pre-fix
    (400 script_exception) and post-fix (real rows) before landing this
    guard.
    """
    r = app_client.get('/curation/review/model_disagreements')
    assert r.status_code == 200, r.text

    body = app_client.fake_os.search.call_args.kwargs['body']  # type: ignore[attr-defined]
    must = body['query']['bool']['must']
    script_clause = next(m for m in must if 'script' in m)
    source = script_clause['script']['script']['source']
    # A non-existent ``.keyword`` sub-field must NOT appear (that was the bug).
    assert "doc['probe_pred_class.keyword']" not in source
    assert "doc['class_name.keyword']" not in source
    assert "doc['probe_pred_class']" in source
    assert "doc['class_name']" in source


def test_disagreements_overrides_default_must_not(app_client: TestClient) -> None:
    """The default must_not excludes class_validated=true; this tab inverts it.

    Plan §1.3, A-PR2: review tabs filter on class_validated (class side)
    or plate_validated (plate tab); model_disagreements wants validated
    class rows so it drops the must_not.
    """
    r = app_client.get('/curation/review/model_disagreements')
    assert r.status_code == 200
    body = app_client.fake_os.search.call_args.kwargs['body']  # type: ignore[attr-defined]
    must_not = body['query']['bool'].get('must_not', [])
    # No filter excluding class_validated=true (we want validated rows).
    assert not any(m == {'term': {'class_validated': True}} for m in must_not)


def test_disagreements_excludes_test_holdout_by_default(app_client: TestClient) -> None:
    r = app_client.get('/curation/review/model_disagreements')
    assert r.status_code == 200
    body = app_client.fake_os.search.call_args.kwargs['body']  # type: ignore[attr-defined]
    must_not = body['query']['bool'].get('must_not', [])
    assert {'term': {'test_holdout': True}} in must_not


def test_disagreements_include_test_param_drops_filter(app_client: TestClient) -> None:
    r = app_client.get('/curation/review/model_disagreements?include_test=true')
    assert r.status_code == 200
    body = app_client.fake_os.search.call_args.kwargs['body']  # type: ignore[attr-defined]
    must_not = body['query']['bool'].get('must_not', [])
    assert {'term': {'test_holdout': True}} not in must_not


def test_disagreements_response_surfaces_probe_fields(app_client: TestClient) -> None:
    r = app_client.get('/curation/review/model_disagreements')
    assert r.status_code == 200
    out = r.json()
    assert out['total'] == 1
    item = out['items'][0]
    assert item['class_name'] == 'pickup'
    assert item['probe_pred_class'] == 'sedan'
    assert item['probe_pred_entropy'] == 0.42
    assert item['reason'] == 'new model disagrees with the validated label'


def test_unknown_tab_lists_disagreements_in_400(app_client: TestClient) -> None:
    r = app_client.get('/curation/review/totally-not-a-tab')
    assert r.status_code == 400
    assert 'model_disagreements' in r.json()['detail']


def test_existing_tabs_still_exclude_validated(app_client: TestClient) -> None:
    """Smoke check: refactoring didn't break the standard tabs' must_not list.

    Plan §1.3, A-PR2: review tabs filter on class_validated (the class-
    side flag — the common case for the labeler /clusters view).
    """
    r = app_client.get('/curation/review/all')
    assert r.status_code == 200
    body = app_client.fake_os.search.call_args.kwargs['body']  # type: ignore[attr-defined]
    must_not = body['query']['bool']['must_not']
    assert {'term': {'class_validated': True}} in must_not


def _disagreement_query_fixture() -> dict[str, Any]:
    """Helper: build a representative search response a test could swap in."""
    return {
        'hits': {
            'total': {'value': 0},
            'hits': [],
        }
    }
