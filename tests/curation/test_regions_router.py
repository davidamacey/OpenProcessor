"""Write-path tests for the human region-labelling endpoints:
``PUT /crops/{id}/regions``, ``PATCH /crops/{id}/region_meta``,
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
from src.services.curation.cluster_ids import FALSE_POSITIVE_REGION_CLUSTER_ID
from src.services.curation.wire import ITEM_WIRE_KEYS


F = get_region_fields()

# No-profile gating contract: every route in this file requires an
# active region profile (409 otherwise) -- this file tests the region
# write paths themselves, so a profile is always active.
pytestmark = pytest.mark.usefixtures('reference_region_profile')


class _FakeRegionOS:
    """AsyncOpenSearch double covering exactly what regions.py's write
    routes call: get/update (OCC single-doc), mget/bulk (batched
    OCC writes), plus indices.refresh for the batch-status endpoint."""

    class _Indices:
        def __init__(self, outer: _FakeRegionOS) -> None:
            self._outer = outer

        async def refresh(self, *, index: str) -> None:  # noqa: ARG002
            self._outer.refresh_calls += 1

    def __init__(self, docs: dict[str, dict[str, Any]] | None = None) -> None:
        self._docs = docs or {}
        self._seq: dict[str, int] = dict.fromkeys(self._docs, 0)
        self.update_calls: list[dict[str, Any]] = []
        self.bulk_calls: list[list[dict[str, Any]]] = []
        self.mget_calls: list[list[str]] = []
        self.refresh_calls = 0
        self.indices = self._Indices(self)

    async def get(self, *, index: str, id: str, **_kw: Any) -> dict[str, Any]:  # noqa: A002, ARG002
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

    async def mget(self, *, body: dict[str, Any]) -> dict[str, Any]:
        ids = [d['_id'] for d in body['docs']]
        self.mget_calls.append(ids)
        docs = []
        for doc_id in ids:
            if doc_id in self._docs:
                docs.append(
                    {
                        '_id': doc_id,
                        '_index': next(
                            (d.get('_index') for d in body['docs'] if d['_id'] == doc_id), None
                        ),
                        '_source': dict(self._docs[doc_id]),
                        '_seq_no': self._seq[doc_id],
                        '_primary_term': 1,
                        'found': True,
                    }
                )
            else:
                docs.append({'_id': doc_id, 'found': False})
        return {'docs': docs}

    async def bulk(
        self,
        *,
        body: list[dict[str, Any]],
        refresh: bool | str = False,  # noqa: ARG002
    ) -> dict[str, Any]:
        self.bulk_calls.append(body)
        items = []
        for action, doc in zip(body[0::2], body[1::2], strict=True):
            meta = action['update']
            doc_id = meta['_id']
            if self._seq.get(doc_id) != meta['if_seq_no']:
                items.append(
                    {
                        'update': {
                            '_id': doc_id,
                            'status': 409,
                            'error': {'type': 'version_conflict_engine_exception'},
                        }
                    }
                )
                continue
            self._docs.setdefault(doc_id, {}).update(doc['doc'])
            self._seq[doc_id] = self._seq.get(doc_id, 0) + 1
            items.append({'update': {'_id': doc_id, 'status': 200}})
        errors = any(i['update']['status'] not in (200, 201) for i in items)
        return {'errors': errors, 'items': items}


@pytest.fixture
def fake_os() -> _FakeRegionOS:
    def _pending_verification_item(crop_id: str) -> dict[str, Any]:
        # A proposed box on each: PATCH region_meta / batch_status's
        # whole-set transition (boxes_with_status) needs one to promote to
        # 'detected', reject to 'verify_rejected', or mark 'false_positive'.
        return {
            'crop_id': crop_id,
            F.status: 'pending_verification',
            F.boxes: [{'box_id': 'b1', 'bbox_norm': [0.1, 0.1, 0.2, 0.2], 'state': 'proposed'}],
        }

    return _FakeRegionOS(
        {
            'crop-1': _pending_verification_item('crop-1'),
            'crop-2': _pending_verification_item('crop-2'),
        }
    )


@pytest.fixture
def app_client(fake_os: _FakeRegionOS) -> Any:
    from src.routers.curation import _raw_opensearch_dep, router as curation_router

    app = FastAPI()
    from _curation_app import mount_curation_routers

    mount_curation_routers(app, curation_router)
    app.dependency_overrides[_raw_opensearch_dep] = lambda: fake_os

    with TestClient(app) as client:
        yield client


# ---------------------------------------------------------------------------
# PUT crops-id-regions
# ---------------------------------------------------------------------------


def test_put_regions_new_box_stamps_the_human_verifier_and_provenance(
    app_client: TestClient, fake_os: _FakeRegionOS
) -> None:
    resp = app_client.put(
        '/curation/projects/default/crops/crop-1/regions',
        json={
            'boxes': [{'box_id': None, 'bbox_norm': [0.1, 0.2, 0.3, 0.4]}],
            'region_label_source': 'human',
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body['item']['region_status'] == 'detected'

    written = fake_os._docs['crop-1']
    (box,) = written[F.boxes]
    assert box['bbox_norm'] == [0.1, 0.2, 0.3, 0.4]
    assert box['state'] == 'accepted'
    assert box['score'] == 1.0
    assert box['detector']
    assert box['detector_version']
    assert box['source'] == 'human'
    assert box['detected_at']
    assert written[F.verified] is True
    assert written[F.validated] is True
    # Verifier fields stamped (a human write is its own verifier).
    assert written[F.verifier]
    assert written[F.verifier_version]
    assert written[F.verified_at]


def test_put_regions_leaving_a_proposed_box_is_a_partial_review(
    app_client: TestClient, fake_os: _FakeRegionOS
) -> None:
    """W8.7: a human write that leaves a ``proposed`` box is a partial
    review -- not validated, set_complete untouched -- so the item stays
    in the review queue."""
    resp = app_client.put(
        '/curation/projects/default/crops/crop-1/regions',
        json={
            'boxes': [
                {'box_id': 'b1'},
                {'box_id': None, 'bbox_norm': [0.5, 0.5, 0.6, 0.6]},
            ]
        },
    )
    assert resp.status_code == 200, resp.text
    written = fake_os._docs['crop-1']
    assert [b['state'] for b in written[F.boxes]] == ['proposed', 'accepted']
    assert F.validated not in written
    assert F.set_complete not in written


@pytest.mark.parametrize(
    'bad_box',
    [
        [1.5, 0.2, 0.3, 0.4],  # out of range
        [-0.1, 0.2, 0.3, 0.4],  # out of range (negative)
        [0.5, 0.5, 0.5, 0.5],  # degenerate (zero area)
        [0.6, 0.2, 0.3, 0.4],  # degenerate (x2 <= x1)
    ],
)
def test_put_regions_rejects_an_invalid_bbox(
    app_client: TestClient, fake_os: _FakeRegionOS, bad_box: list[float]
) -> None:
    before = dict(fake_os._docs['crop-1'])
    resp = app_client.put(
        '/curation/projects/default/crops/crop-1/regions',
        json={'boxes': [{'box_id': None, 'bbox_norm': bad_box}]},
    )
    assert resp.status_code == 422, resp.text
    assert fake_os._docs['crop-1'] == before


def test_put_regions_rejects_an_invalid_bbox_on_a_stored_box_move(
    app_client: TestClient, fake_os: _FakeRegionOS
) -> None:
    resp = app_client.put(
        '/curation/projects/default/crops/crop-1/regions',
        json={'boxes': [{'box_id': 'b1', 'bbox_norm': [0.5, 0.5, 0.5, 0.5]}]},
    )
    assert resp.status_code == 422, resp.text
    assert fake_os._docs['crop-1'][F.boxes][0]['bbox_norm'] == [0.1, 0.1, 0.2, 0.2]


def test_put_regions_new_box_without_a_bbox_is_422(app_client: TestClient) -> None:
    resp = app_client.put(
        '/curation/projects/default/crops/crop-1/regions',
        json={'boxes': [{'box_id': None, 'state': 'accepted'}]},
    )
    assert resp.status_code == 422, resp.text


def test_put_regions_duplicate_box_id_is_422(app_client: TestClient) -> None:
    resp = app_client.put(
        '/curation/projects/default/crops/crop-1/regions',
        json={'boxes': [{'box_id': 'b1'}, {'box_id': 'b1'}]},
    )
    assert resp.status_code == 422, resp.text


def test_put_regions_missing_crop_returns_404(app_client: TestClient) -> None:
    resp = app_client.put(
        '/curation/projects/default/crops/does-not-exist/regions',
        json={'boxes': [{'box_id': None, 'bbox_norm': [0.1, 0.2, 0.3, 0.4]}]},
    )
    assert resp.status_code == 404


def test_legacy_single_box_routes_are_gone(app_client: TestClient) -> None:
    put = app_client.put(
        '/curation/projects/default/crops/crop-1/region',
        json={'region_bbox_norm': [0.1, 0.2, 0.3, 0.4]},
    )
    batch = app_client.put(
        '/curation/projects/default/crops/batch_region',
        json={'crop_ids': ['crop-1'], 'region_bbox_norm': None},
    )
    assert put.status_code in (404, 405)
    assert batch.status_code in (404, 405)


# ---------------------------------------------------------------------------
# PATCH crops-id-region_meta
# ---------------------------------------------------------------------------


def test_patch_region_meta_rejects_status_outside_human_settable_set(
    app_client: TestClient,
) -> None:
    resp = app_client.patch(
        '/curation/projects/default/crops/crop-1/region_meta',
        json={'region_status': 'pending_detection'},  # pipeline-only status
    )
    assert resp.status_code == 400
    assert 'region_status must be one of' in resp.json()['detail']


def test_patch_region_meta_accepts_human_settable_status(
    app_client: TestClient, fake_os: _FakeRegionOS
) -> None:
    resp = app_client.patch(
        '/curation/projects/default/crops/crop-1/region_meta',
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
        '/curation/projects/default/crops/crop-1/region_meta',
        json={'region_status': 'false_positive', 'region_label_source': 'human'},
    )
    assert resp.status_code == 200, resp.text
    (box,) = fake_os._docs['crop-1'][F.boxes]
    assert box['state'] == 'false_positive'
    assert box['cluster_id'] == FALSE_POSITIVE_REGION_CLUSTER_ID
    assert box['cluster_subid'] is None


def test_patch_region_box_text_only_does_not_touch_cluster_fields(
    app_client: TestClient, fake_os: _FakeRegionOS
) -> None:
    """``region_text`` moved off ``region_meta`` onto the per-box PATCH
    route (W8, D decision, 2026-09-26)."""
    fake_os._docs['crop-1'][F.boxes] = [
        {'box_id': 'b1', 'bbox_norm': [0.1, 0.1, 0.2, 0.2], 'state': 'accepted'}
    ]
    resp = app_client.patch(
        '/curation/projects/default/crops/crop-1/regions/b1',
        json={'text': 'ABC123'},
    )
    assert resp.status_code == 200, resp.text
    written = fake_os._docs['crop-1']
    box = written[F.boxes][0]
    assert box['text'] == 'ABC123'
    assert box['text_source'] == 'human'
    assert box.get('cluster_id') is None


def test_patch_region_meta_requires_at_least_one_field(app_client: TestClient) -> None:
    resp = app_client.patch('/curation/projects/default/crops/crop-1/region_meta', json={})
    assert resp.status_code == 400


def test_patch_region_meta_response_reports_wire_names(
    app_client: TestClient,
) -> None:
    """``updated_fields`` echoes the fixed ``region_*`` wire names, never
    storage keys (docs/design/curation_api_contract.md). ``region_text``
    is gone from this route (W8, D decision) -- ``region_status`` alone."""
    resp = app_client.patch(
        '/curation/projects/default/crops/crop-1/region_meta',
        json={'region_status': 'detected', 'region_label_source': 'human'},
    )
    assert resp.status_code == 200, resp.text
    updated_fields = resp.json()['updated_fields']
    assert updated_fields == ['region_status']


def test_get_crop_returns_shared_wire_item(app_client: TestClient, fake_os: _FakeRegionOS) -> None:
    """``GET /crops/{id}`` returns the shared wire item, never the raw
    OpenSearch ``_source``."""
    app_client.patch(
        '/curation/projects/default/crops/crop-1/region_meta',
        json={'region_status': 'detected', 'region_label_source': 'human'},
    )
    resp = app_client.get('/curation/projects/default/crops/crop-1')
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body['region_status'] == 'detected'
    assert set(body) == ITEM_WIRE_KEYS


# ---------------------------------------------------------------------------
# POST regions-batch_status
# ---------------------------------------------------------------------------


def test_batch_set_region_status_rejects_non_human_settable_status(
    app_client: TestClient,
) -> None:
    resp = app_client.post(
        '/curation/projects/default/regions/batch_status',
        json={'crop_ids': ['crop-1'], 'region_status': 'detection_failed'},
    )
    assert resp.status_code == 400


def test_batch_set_region_status_updates_every_crop_and_refreshes(
    app_client: TestClient, fake_os: _FakeRegionOS
) -> None:
    resp = app_client.post(
        '/curation/projects/default/regions/batch_status',
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
    assert [r['region_box_id'] for r in body['items']] == [None, None]  # item-level rows
    assert body['conflicts'] == []
    assert fake_os._docs['crop-1'][F.status] == 'detected'
    assert fake_os._docs['crop-2'][F.status] == 'detected'
    assert fake_os._docs['crop-1'][F.verified] is True
    # The batch write refreshes via the final bulk call's
    # refresh='wait_for', not a separate forced indices.refresh().
    assert fake_os.refresh_calls == 0
    assert len(fake_os.bulk_calls) == 1


def test_batch_set_region_status_false_positive_marks_fp_bucket_for_every_crop(
    app_client: TestClient, fake_os: _FakeRegionOS
) -> None:
    resp = app_client.post(
        '/curation/projects/default/regions/batch_status',
        json={'crop_ids': ['crop-1', 'crop-2'], 'region_status': 'false_positive'},
    )
    assert resp.status_code == 200, resp.text
    for crop_id in ('crop-1', 'crop-2'):
        (box,) = fake_os._docs[crop_id][F.boxes]
        assert box['cluster_id'] == FALSE_POSITIVE_REGION_CLUSTER_ID


def test_batch_set_region_status_empty_crop_ids_is_a_noop(app_client: TestClient) -> None:
    resp = app_client.post(
        '/curation/projects/default/regions/batch_status',
        json={'crop_ids': [], 'region_status': 'detected'},
    )
    assert resp.status_code == 200
    assert resp.json() == {
        'updated': 0,
        'conflicts': [],
        'invalid': [],
        'items': [],
        'vector_refresh': {'embedded': 0, 'pending': 0},
    }


def test_batch_set_region_status_missing_crop_reports_conflict_not_500(
    app_client: TestClient,
) -> None:
    resp = app_client.post(
        '/curation/projects/default/regions/batch_status',
        json={'crop_ids': ['crop-1', 'does-not-exist'], 'region_status': 'detected'},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body['updated'] == 1
    assert len(body['conflicts']) == 1
    assert body['conflicts'][0]['crop_id'] == 'does-not-exist'
