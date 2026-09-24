"""Write-path tests for the human region-labelling endpoints (plan Wave
5 W5.c): ``PUT /crops/{id}/region``, ``PATCH /crops/{id}/region_meta``,
``POST /regions/batch_status``.

Before this file, ``src/routers/curation/regions.py`` (12.62% coverage)
had never had a single test exercise a write path — the entire human
region-labelling surface was ported with zero tests. Drives the real
FastAPI routes (not the bare functions) against a fake OpenSearch so
request validation, dependency wiring, and the handlers all run for
real.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.config import get_region_fields
from src.services.curation.wire import ITEM_WIRE_KEYS


F = get_region_fields()


class _FakeRegionOS:
    """AsyncOpenSearch double covering exactly what regions.py's write
    routes call: get/update (OCC single-doc), plus indices.refresh for
    the batch-status endpoint."""

    class _Indices:
        def __init__(self, outer: _FakeRegionOS) -> None:
            self._outer = outer

        async def refresh(self, *, index: str) -> None:  # noqa: ARG002
            self._outer.refresh_calls += 1

    def __init__(self, docs: dict[str, dict[str, Any]] | None = None) -> None:
        self._docs = docs or {}
        self._seq: dict[str, int] = dict.fromkeys(self._docs, 0)
        self.update_calls: list[dict[str, Any]] = []
        self.refresh_calls = 0
        self.indices = self._Indices(self)

    async def get(self, *, index: str, id: str) -> dict[str, Any]:  # noqa: A002, ARG002
        if id not in self._docs:
            raise KeyError(id)
        return {
            '_id': id,
            '_source': dict(self._docs[id]),
            '_seq_no': self._seq[id],
            '_primary_term': 1,
            'found': True,
        }

    async def update(
        self,
        *,
        index: str,  # noqa: ARG002
        id: str,  # noqa: A002
        body: dict[str, Any],
        if_seq_no: int,
        if_primary_term: int,  # noqa: ARG002
        refresh: bool | str = False,  # noqa: ARG002
    ) -> dict[str, Any]:
        if self._seq.get(id) != if_seq_no:
            msg = f'version conflict on {id}'
            raise Exception(msg)
        doc = body['doc']
        self.update_calls.append({'id': id, 'doc': dict(doc)})
        self._docs[id].update(doc)
        self._seq[id] = self._seq.get(id, 0) + 1
        return {'result': 'updated'}


@pytest.fixture
def fake_os() -> _FakeRegionOS:
    return _FakeRegionOS(
        {
            # A box on each: confirming ('detected') needs one.
            'crop-1': {
                'crop_id': 'crop-1',
                F.status: 'pending_detection',
                F.bbox_norm: [0.1, 0.1, 0.2, 0.2],
            },
            'crop-2': {
                'crop_id': 'crop-2',
                F.status: 'pending_detection',
                F.bbox_norm: [0.1, 0.1, 0.2, 0.2],
            },
        }
    )


@pytest.fixture
def app_client(fake_os: _FakeRegionOS) -> Any:
    from src.routers.curation import _raw_opensearch_dep, router as curation_router

    app = FastAPI()
    app.include_router(curation_router)
    app.dependency_overrides[_raw_opensearch_dep] = lambda: fake_os

    with TestClient(app) as client:
        yield client


# ---------------------------------------------------------------------------
# PUT crops-id-region
# ---------------------------------------------------------------------------


def test_set_crop_region_stamps_verifier_fields(
    app_client: TestClient, fake_os: _FakeRegionOS
) -> None:
    resp = app_client.put(
        '/curation/crops/crop-1/region',
        json={'region_bbox_norm': [0.1, 0.2, 0.3, 0.4], 'region_label_source': 'human'},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body['region_status'] == 'detected'

    written = fake_os._docs['crop-1']
    assert written[F.bbox_norm] == [0.1, 0.2, 0.3, 0.4]
    assert written[F.verified] is True
    assert written[F.validated] is True
    # Verifier fields stamped (human PUT is its own verifier).
    assert written[F.verifier]
    assert written[F.verifier_version]
    assert written[F.verified_at]
    assert written[F.detector]
    assert written[F.detected_at]


def test_set_crop_region_null_bbox_marks_no_region_visible(
    app_client: TestClient, fake_os: _FakeRegionOS
) -> None:
    resp = app_client.put('/curation/crops/crop-1/region', json={'region_bbox_norm': None})
    assert resp.status_code == 200, resp.text
    assert resp.json()['region_status'] == 'no_region_visible'
    written = fake_os._docs['crop-1']
    assert written[F.bbox_norm] is None
    assert written[F.score] is None
    assert written[F.validated] is True


def test_set_crop_region_rejects_out_of_range_bbox(app_client: TestClient) -> None:
    resp = app_client.put(
        '/curation/crops/crop-1/region',
        json={'region_bbox_norm': [1.5, 0.2, 0.3, 0.4]},
    )
    assert resp.status_code == 400


def test_set_crop_region_rejects_degenerate_bbox(app_client: TestClient) -> None:
    resp = app_client.put(
        '/curation/crops/crop-1/region',
        json={'region_bbox_norm': [0.5, 0.5, 0.5, 0.5]},
    )
    assert resp.status_code == 400


def test_set_crop_region_missing_crop_returns_404(app_client: TestClient) -> None:
    resp = app_client.put(
        '/curation/crops/does-not-exist/region',
        json={'region_bbox_norm': [0.1, 0.2, 0.3, 0.4]},
    )
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# PATCH crops-id-region_meta
# ---------------------------------------------------------------------------


def test_patch_region_meta_rejects_status_outside_human_settable_set(
    app_client: TestClient,
) -> None:
    resp = app_client.patch(
        '/curation/crops/crop-1/region_meta',
        json={'region_status': 'pending_detection'},  # pipeline-only status
    )
    assert resp.status_code == 400
    assert 'region_status must be one of' in resp.json()['detail']


def test_patch_region_meta_accepts_human_settable_status(
    app_client: TestClient, fake_os: _FakeRegionOS
) -> None:
    resp = app_client.patch(
        '/curation/crops/crop-1/region_meta',
        json={'region_status': 'verify_rejected', 'region_label_source': 'human'},
    )
    assert resp.status_code == 200, resp.text
    written = fake_os._docs['crop-1']
    assert written[F.status] == 'verify_rejected'
    # Human-sourced patch is terminal.
    assert written[F.validated] is True


def test_patch_region_meta_false_positive_routes_to_fp_bucket(
    app_client: TestClient, fake_os: _FakeRegionOS
) -> None:
    resp = app_client.patch(
        '/curation/crops/crop-1/region_meta',
        json={'region_status': 'false_positive', 'region_label_source': 'human'},
    )
    assert resp.status_code == 200, resp.text
    written = fake_os._docs['crop-1']
    assert written[F.cluster_id] is not None  # FALSE_POSITIVE_REGION_CLUSTER_ID
    assert written[F.cluster_subid] is None


def test_patch_region_meta_text_only_does_not_touch_cluster_fields(
    app_client: TestClient, fake_os: _FakeRegionOS
) -> None:
    resp = app_client.patch(
        '/curation/crops/crop-1/region_meta',
        json={'region_text': 'ABC123'},
    )
    assert resp.status_code == 200, resp.text
    written = fake_os._docs['crop-1']
    assert written[F.text] == 'ABC123'
    assert written[F.text_source] == 'human'
    assert F.cluster_id not in written


def test_patch_region_meta_requires_at_least_one_field(app_client: TestClient) -> None:
    resp = app_client.patch('/curation/crops/crop-1/region_meta', json={})
    assert resp.status_code == 400


def test_patch_region_meta_response_reports_wire_names(
    app_client: TestClient,
) -> None:
    """``updated_fields`` echoes the fixed ``region_*`` wire names, never
    storage keys (docs/design/curation_api_contract.md)."""
    resp = app_client.patch(
        '/curation/crops/crop-1/region_meta',
        json={'region_text': 'ABC123', 'region_status': 'detected', 'region_label_source': 'human'},
    )
    assert resp.status_code == 200, resp.text
    updated_fields = resp.json()['updated_fields']
    assert updated_fields == ['region_status', 'region_text']


def test_get_crop_returns_shared_wire_item(app_client: TestClient, fake_os: _FakeRegionOS) -> None:
    """``GET /crops/{id}`` returns the shared wire item, never the raw
    OpenSearch ``_source``."""
    app_client.patch(
        '/curation/crops/crop-1/region_meta',
        json={'region_text': 'ABC123', 'region_status': 'detected', 'region_label_source': 'human'},
    )
    resp = app_client.get('/curation/crops/crop-1')
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body['region_text'] == 'ABC123'
    assert body['region_status'] == 'detected'
    assert set(body) == ITEM_WIRE_KEYS


# ---------------------------------------------------------------------------
# POST regions-batch_status
# ---------------------------------------------------------------------------


def test_batch_set_region_status_rejects_non_human_settable_status(
    app_client: TestClient,
) -> None:
    resp = app_client.post(
        '/curation/regions/batch_status',
        json={'crop_ids': ['crop-1'], 'region_status': 'detection_failed'},
    )
    assert resp.status_code == 400


def test_batch_set_region_status_updates_every_crop_and_refreshes(
    app_client: TestClient, fake_os: _FakeRegionOS
) -> None:
    resp = app_client.post(
        '/curation/regions/batch_status',
        json={
            'crop_ids': ['crop-1', 'crop-2'],
            'region_status': 'detected',
            'region_verified': True,
            'region_label_source': 'human',
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body['updated'] == 2
    assert body['conflicts'] == []
    assert fake_os._docs['crop-1'][F.status] == 'detected'
    assert fake_os._docs['crop-2'][F.status] == 'detected'
    assert fake_os._docs['crop-1'][F.verified] is True
    assert fake_os.refresh_calls == 1


def test_batch_set_region_status_false_positive_marks_fp_bucket_for_every_crop(
    app_client: TestClient, fake_os: _FakeRegionOS
) -> None:
    resp = app_client.post(
        '/curation/regions/batch_status',
        json={'crop_ids': ['crop-1', 'crop-2'], 'region_status': 'false_positive'},
    )
    assert resp.status_code == 200, resp.text
    assert fake_os._docs['crop-1'][F.cluster_id] is not None
    assert fake_os._docs['crop-2'][F.cluster_id] is not None


def test_batch_set_region_status_empty_crop_ids_is_a_noop(app_client: TestClient) -> None:
    resp = app_client.post(
        '/curation/regions/batch_status',
        json={'crop_ids': [], 'region_status': 'detected'},
    )
    assert resp.status_code == 200
    assert resp.json() == {'updated': 0, 'conflicts': [], 'invalid': [], 'items': []}


def test_batch_set_region_status_missing_crop_reports_conflict_not_500(
    app_client: TestClient,
) -> None:
    resp = app_client.post(
        '/curation/regions/batch_status',
        json={'crop_ids': ['crop-1', 'does-not-exist'], 'region_status': 'detected'},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body['updated'] == 1
    assert len(body['conflicts']) == 1
    assert body['conflicts'][0]['crop_id'] == 'does-not-exist'
