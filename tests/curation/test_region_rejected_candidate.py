"""A verifier-rejected candidate box stays reviewable and reversible (DQ-B2).

The worker keeps the box the verifier rejected (with its detector, score
and source) in the ``candidate_*`` fields plus the rejection reason --
never in ``bbox_norm``, which every reader treats as an accepted region.
``GET /regions?status=verify_rejected`` lists those items, and a human
confirm promotes the candidate into the region box with the detector's
provenance, undoable through the region edit history.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from curation.query_fakes import QueryFakeOpenSearch
from scripts.curation.worker.verify import _region_write_doc, candidate_reject_doc
from src.config import get_curation_config, get_region_fields
from src.services.curation.edit_history import EditKind, restore_edit_state
from src.services.curation.wire import serialize_item
from src.services.detection.cascade_detect import RegionCandidate
from src.services.labeling.vlm_labeler import VlmCombinedReply

from .test_region_cascade_integrity import _drive_worker, _FakeOpenSearch, _item, _profile


if TYPE_CHECKING:
    from pathlib import Path


F = get_region_fields()
INDEX = get_curation_config().items_index
CANDIDATE = [0.3, 0.6, 0.4, 0.65]


def _rejected(crop_id: str, *, with_candidate: bool = True) -> dict[str, Any]:
    doc: dict[str, Any] = {
        'crop_id': crop_id,
        'bbox_norm': [0.2, 0.4, 0.6, 0.8],
        F.status: 'verify_rejected',
        F.rejection_reason: 'region_visible_elsewhere',
        F.bbox_correct: False,
        F.detector_chain: ['det_model:hit', 'det_model:combined_verify_reject:x'],
        F.detected_at: '2026-09-24T03:08:19+00:00',
    }
    if with_candidate:
        doc.update(
            {
                F.candidate_bbox_norm: list(CANDIDATE),
                F.candidate_score: 0.81,
                F.candidate_detector: 'det_model',
                F.candidate_detector_version: '3',
                F.candidate_source: 'detector',
            }
        )
    return doc


@pytest.fixture
def fake_os() -> QueryFakeOpenSearch:
    return QueryFakeOpenSearch(
        {
            INDEX: {
                'rej': _rejected('rej'),
                'legacy': _rejected('legacy', with_candidate=False),
                'det': {
                    'crop_id': 'det',
                    'bbox_norm': [0.0, 0.0, 0.5, 0.5],
                    F.bbox_norm: [0.1, 0.1, 0.2, 0.2],
                    F.status: 'detected',
                    F.detected_at: '2026-09-24T04:00:00+00:00',
                },
            }
        }
    )


@pytest.fixture
def client(fake_os: QueryFakeOpenSearch) -> Any:
    from src.routers.curation import _raw_opensearch_dep, router as curation_router

    app = FastAPI()
    app.include_router(curation_router)
    app.dependency_overrides[_raw_opensearch_dep] = lambda: fake_os
    with TestClient(app) as c:
        yield c


def _doc(fake_os: QueryFakeOpenSearch, crop_id: str) -> dict[str, Any]:
    return fake_os.docs(INDEX)[crop_id]


class TestWorkerKeepsTheCandidate:
    def test_reject_doc_keeps_box_detector_and_reason_out_of_bbox_norm(self) -> None:
        doc = candidate_reject_doc(
            candidate_in_source=(0.3, 0.6, 0.4, 0.65),
            candidate_score=0.81,
            detector='det_model',
            detector_version='3',
            candidate_source='detector',
            reason='region_visible_elsewhere',
            chain=['det_model:hit'],
            bbox_correct=False,
        )
        assert doc[F.status] == 'verify_rejected'
        assert doc[F.bbox_norm] is None
        assert doc[F.score] is None
        assert doc[F.candidate_bbox_norm] == CANDIDATE
        assert doc[F.candidate_score] == 0.81
        assert doc[F.candidate_detector] == 'det_model'
        assert doc[F.candidate_detector_version] == '3'
        assert doc[F.candidate_source] == 'detector'
        assert doc[F.rejection_reason] == 'region_visible_elsewhere'
        assert doc[F.bbox_correct] is False

    def test_an_accepted_write_clears_a_stale_candidate(self) -> None:
        doc = _region_write_doc(
            plate_in_source=(0.1, 0.1, 0.2, 0.2),
            score=0.9,
            detector='det_model',
            detector_version='3',
            chain=[],
        )
        assert doc[F.candidate_bbox_norm] is None
        assert doc[F.candidate_detector] is None
        assert doc[F.rejection_reason] is None

    @pytest.mark.asyncio
    @pytest.mark.usefixtures('reference_region_profile')
    async def test_streaming_worker_stores_the_rejected_candidate(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = _FakeOpenSearch({'c1': _item()}, search_delay=0.0, lag_searches=0)
        await _drive_worker(
            tmp_path,
            monkeypatch,
            fake_os=fake,
            primary=RegionCandidate(bbox_norm=(0.3, 0.6, 0.6, 0.75), score=0.77, source='det'),
            segmenter=None,
            reply=VlmCombinedReply(img_id='c1', region_visible=True, region_bbox_correct=False),
        )
        doc = fake.live['c1']
        det = _profile().detector_model
        assert doc[F.status] == 'verify_rejected'
        assert doc.get(F.bbox_norm) is None
        assert doc[F.candidate_bbox_norm] == pytest.approx([0.34, 0.58, 0.58, 0.7])
        assert doc[F.candidate_score] == pytest.approx(0.77)
        assert doc[F.candidate_detector] == det
        assert doc[F.candidate_source] == 'detector'
        assert doc[F.rejection_reason] == 'region_visible_elsewhere'
        assert doc[F.bbox_correct] is False


class TestRegionsStatusFilter:
    def test_status_lists_rejected_items_without_a_box(self, client: TestClient) -> None:
        resp = client.get('/curation/regions', params={'status': 'verify_rejected'})
        assert resp.status_code == 200, resp.text
        items = {i['crop_id']: i for i in resp.json()['items']}
        assert set(items) == {'rej', 'legacy'}
        assert items['rej']['region_candidate_bbox_norm'] == CANDIDATE
        assert items['rej']['region_candidate_bbox_in_parent'] == pytest.approx(
            [0.25, 0.5, 0.5, 0.625]
        )
        assert items['rej']['region_bbox_norm'] is None

    def test_status_detected_lists_only_detected(self, client: TestClient) -> None:
        resp = client.get('/curation/regions', params={'status': 'detected'})
        assert [i['crop_id'] for i in resp.json()['items']] == ['det']

    def test_default_lists_only_boxed_items(self, client: TestClient) -> None:
        resp = client.get('/curation/regions')
        assert [i['crop_id'] for i in resp.json()['items']] == ['det']

    def test_unknown_status_is_400(self, client: TestClient) -> None:
        resp = client.get('/curation/regions', params={'status': 'bogus'})
        assert resp.status_code == 400


class TestHumanReversal:
    def test_confirm_promotes_the_candidate_with_its_provenance(
        self, client: TestClient, fake_os: QueryFakeOpenSearch
    ) -> None:
        resp = client.patch(
            '/curation/crops/rej/region_meta',
            json={'region_status': 'detected', 'region_label_source': 'human'},
        )
        assert resp.status_code == 200, resp.text
        doc = _doc(fake_os, 'rej')
        assert doc[F.status] == 'detected'
        assert doc[F.bbox_norm] == CANDIDATE
        assert doc[F.bbox_frame] == 'source'
        assert doc[F.score] == 0.81
        assert doc[F.detector] == 'det_model'
        assert doc[F.detector_version] == '3'
        assert doc[F.source] == 'detector'
        assert doc[F.verified] is True
        assert doc[F.validated] is True
        assert doc.get(F.candidate_bbox_norm) is None
        assert doc.get(F.rejection_reason) is None
        assert resp.json()['item']['region_bbox_norm'] == CANDIDATE

    def test_confirm_then_undo_restores_the_rejection(
        self, client: TestClient, fake_os: QueryFakeOpenSearch
    ) -> None:
        before = dict(_doc(fake_os, 'rej'))
        client.patch(
            '/curation/crops/rej/region_meta',
            json={'region_status': 'detected', 'region_label_source': 'human'},
        )
        resp = client.post('/curation/crops/rej/region/undo')
        assert resp.status_code == 200, resp.text
        doc = _doc(fake_os, 'rej')
        for key in (
            F.status,
            F.rejection_reason,
            F.candidate_bbox_norm,
            F.candidate_score,
            F.candidate_detector,
            F.candidate_detector_version,
            F.candidate_source,
        ):
            assert doc.get(key) == before.get(key), key
        assert doc.get(F.bbox_norm) is None
        assert doc.get(F.detector) is None

    def test_false_positive_keeps_the_candidate_as_the_box(
        self, client: TestClient, fake_os: QueryFakeOpenSearch
    ) -> None:
        resp = client.patch(
            '/curation/crops/rej/region_meta',
            json={'region_status': 'false_positive', 'region_label_source': 'human'},
        )
        assert resp.status_code == 200, resp.text
        doc = _doc(fake_os, 'rej')
        assert doc[F.status] == 'false_positive'
        assert doc[F.bbox_norm] == CANDIDATE
        assert doc[F.detector] == 'det_model'

    def test_put_of_the_candidate_box_is_a_confirmation(
        self, client: TestClient, fake_os: QueryFakeOpenSearch
    ) -> None:
        resp = client.put('/curation/crops/rej/region', json={'region_bbox_norm': CANDIDATE})
        assert resp.status_code == 200, resp.text
        doc = _doc(fake_os, 'rej')
        assert doc[F.status] == 'detected'
        assert doc[F.detector] == 'det_model'
        assert doc[F.score] == 0.81
        assert doc[F.verifier] == 'human'
        assert doc.get(F.candidate_bbox_norm) is None

    def test_put_of_a_new_box_replaces_the_candidate(
        self, client: TestClient, fake_os: QueryFakeOpenSearch
    ) -> None:
        resp = client.put(
            '/curation/crops/rej/region', json={'region_bbox_norm': [0.31, 0.61, 0.42, 0.66]}
        )
        assert resp.status_code == 200, resp.text
        doc = _doc(fake_os, 'rej')
        assert doc[F.detector] == 'human'
        assert doc.get(F.candidate_bbox_norm) is None
        assert doc.get(F.rejection_reason) is None

    def test_confirm_without_a_box_or_candidate_is_still_refused(self, client: TestClient) -> None:
        resp = client.patch(
            '/curation/crops/legacy/region_meta',
            json={'region_status': 'detected', 'region_label_source': 'human'},
        )
        assert resp.status_code == 422


def test_undo_of_an_older_snapshot_leaves_fields_it_never_recorded() -> None:
    entry = {'kind': 'region', 'state': {F.status: 'detected', F.bbox_norm: [0.1, 0.1, 0.2, 0.2]}}
    restored = restore_edit_state(entry, EditKind.REGION)
    assert restored == {F.status: 'detected', F.bbox_norm: [0.1, 0.1, 0.2, 0.2]}
    assert F.source not in restored


def test_wire_item_carries_the_candidate_fields() -> None:
    item = serialize_item(_rejected('x'), 'x', api_prefix='')
    assert item['region_candidate_detector'] == 'det_model'
    assert item['region_candidate_score'] == 0.81
    assert item['region_candidate_source'] == 'detector'
