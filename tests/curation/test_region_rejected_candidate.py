"""A verifier-rejected box stays reviewable and reversible (DQ-B2).

The worker keeps the box the verifier rejected (with its detector, score
and source) as a ``rejected`` entry of ``region_boxes`` plus the
rejection reason -- never as an accepted region.
``GET /regions?status=verify_rejected`` lists those items, and a human
reversal (the whole-set confirm, or the per-box accept) turns the box
into an accepted one with the detector's provenance kept, undoable
through the region edit history.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from curation.query_fakes import QueryFakeOpenSearch
from src.config import get_region_fields
from src.config.curation import base_curation_config
from src.config.region_rejection import REJECT_REASON_HUMAN, REJECT_REASON_SANITY_PREFIX
from src.services.curation.cluster_ids import FALSE_POSITIVE_REGION_CLUSTER_ID
from src.services.curation.edit_history import EditKind, restore_edit_state
from src.services.curation.region_boxes import RegionBox, boxes_write_fields
from src.services.detection.cascade_detect.candidate import RegionCandidate
from src.services.labeling.region_overlay import VlmBoxVerdict
from src.services.labeling.vlm_models import VlmCombinedReply

from .test_region_cascade_integrity import _drive_worker, _FakeOpenSearch, _item, _profile


# No-profile gating contract: this file exercises region routes, which
# require an active region profile (409 otherwise).
pytestmark = pytest.mark.usefixtures('reference_region_profile')


if TYPE_CHECKING:
    from pathlib import Path


F = get_region_fields()
INDEX = base_curation_config().items_index
CANDIDATE = [0.3, 0.6, 0.4, 0.65]


def _rejected_box(**over: Any) -> RegionBox:
    kwargs: dict[str, Any] = {
        'box_id': 'b1',
        'bbox_norm': tuple(CANDIDATE),
        'state': 'rejected',
        'score': 0.81,
        'detector': 'det_model',
        'detector_version': '3',
        'source': 'detector',
        'rejection_reason': 'region_visible_elsewhere',
        'bbox_correct': False,
    }
    kwargs.update(over)
    return RegionBox(**kwargs)


def _item_with(crop_id: str, boxes: list[RegionBox], status: str, **extra: Any) -> dict[str, Any]:
    """A production-shaped item: the box list and its summary fields come
    from ``boxes_write_fields``, exactly as every writer produces them."""
    return {
        'crop_id': crop_id,
        'bbox_norm': [0.2, 0.4, 0.6, 0.8],
        F.status: status,
        F.detector_chain: ['det_model:hit', 'det_model:combined_verify_reject:x'],
        F.detected_at: '2026-09-24T03:08:19+00:00',
        **boxes_write_fields(boxes, current_src={}),
        **extra,
    }


@pytest.fixture
def fake_os() -> QueryFakeOpenSearch:
    return QueryFakeOpenSearch(
        {
            INDEX: {
                'rej': _item_with('rej', [_rejected_box()], 'verify_rejected'),
                'legacy': _item_with('legacy', [], 'verify_rejected'),
                'det': _item_with(
                    'det',
                    [RegionBox(box_id='b1', bbox_norm=(0.1, 0.1, 0.2, 0.2), state='accepted')],
                    'detected',
                    **{F.detected_at: '2026-09-24T04:00:00+00:00'},
                ),
            }
        }
    )


@pytest.fixture
def client(fake_os: QueryFakeOpenSearch) -> Any:
    from src.routers.curation import _raw_opensearch_dep, router as curation_router

    app = FastAPI()
    from _curation_app import mount_curation_routers

    mount_curation_routers(app, curation_router)
    app.dependency_overrides[_raw_opensearch_dep] = lambda: fake_os
    with TestClient(app) as c:
        yield c


def _doc(fake_os: QueryFakeOpenSearch, crop_id: str) -> dict[str, Any]:
    return fake_os.docs(INDEX)[crop_id]


def _box(fake_os: QueryFakeOpenSearch, crop_id: str, box_id: str = 'b1') -> dict[str, Any]:
    return next(b for b in _doc(fake_os, crop_id)[F.boxes] if b['box_id'] == box_id)


class TestWorkerKeepsTheCandidate:
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
            reply=VlmCombinedReply(
                img_id='c1',
                region_visible=True,
                region_boxes=[VlmBoxVerdict(box=1, bbox_correct=False, confidence=None)],
            ),
        )
        doc = fake.live['c1']
        det = _profile().detector_model
        assert doc[F.status] == 'verify_rejected'
        box = doc[F.boxes][0]
        assert box['state'] == 'rejected'
        assert box['bbox_norm'] == pytest.approx([0.34, 0.58, 0.58, 0.7])
        assert box['score'] == pytest.approx(0.77)
        assert box['detector'] == det
        assert box['source'] == 'detector'
        assert box['rejection_reason'] == 'region_visible_elsewhere'
        assert box['bbox_correct'] is False
        # R-M4: a fully-rejected combined-VLM write must never mark the
        # item `verified` -- the VLM DID answer, but it rejected the only
        # candidate, so no region was ever confirmed.
        assert doc[F.verified] is False


class TestRegionsStatusFilter:
    def test_status_lists_rejected_items_with_their_rejected_box(self, client: TestClient) -> None:
        resp = client.get(
            '/curation/projects/default/regions', params={'status': 'verify_rejected'}
        )
        assert resp.status_code == 200, resp.text
        items = {i['crop_id']: i for i in resp.json()['items']}
        assert set(items) == {'rej', 'legacy'}
        (box,) = items['rej']['region_boxes']
        assert box['state'] == 'rejected'
        assert box['bbox_norm'] == CANDIDATE
        assert box['bbox_in_parent'] == pytest.approx([0.25, 0.5, 0.5, 0.625])
        assert items['legacy']['region_boxes'] == []

    def test_status_detected_lists_only_detected(self, client: TestClient) -> None:
        resp = client.get('/curation/projects/default/regions', params={'status': 'detected'})
        assert [i['crop_id'] for i in resp.json()['items']] == ['det']

    def test_default_lists_only_boxed_items(self, client: TestClient) -> None:
        resp = client.get('/curation/projects/default/regions')
        assert [i['crop_id'] for i in resp.json()['items']] == ['det']

    def test_unknown_status_is_400(self, client: TestClient) -> None:
        resp = client.get('/curation/projects/default/regions', params={'status': 'bogus'})
        assert resp.status_code == 400


def _assert_reopened_with_provenance(fake_os: QueryFakeOpenSearch, crop_id: str = 'rej') -> None:
    """The reversed rejection: the box is accepted, keeps the detector's
    provenance (a human accepting a box doesn't change who found it), and
    carries no rejection reason."""
    doc = _doc(fake_os, crop_id)
    box = _box(fake_os, crop_id)
    assert box['state'] == 'accepted'
    assert box['bbox_norm'] == CANDIDATE
    assert box['score'] == 0.81
    assert box['detector'] == 'det_model'
    assert box['detector_version'] == '3'
    assert box['source'] == 'detector'
    assert box['rejection_reason'] is None
    assert doc[F.status] == 'detected'
    assert doc[F.count] == 1
    assert doc[F.rejected_count] == 0
    assert doc.get(F.rejection_reason) is None


class TestHumanReversal:
    def test_whole_set_confirm_reopens_the_box_with_its_provenance(
        self, client: TestClient, fake_os: QueryFakeOpenSearch
    ) -> None:
        resp = client.patch(
            '/curation/projects/default/crops/rej/region_meta',
            json={'region_status': 'detected', 'region_label_source': 'human'},
        )
        assert resp.status_code == 200, resp.text
        _assert_reopened_with_provenance(fake_os)
        doc = _doc(fake_os, 'rej')
        assert doc[F.verified] is True
        assert doc[F.validated] is True
        (wire_box,) = resp.json()['item']['region_boxes']
        assert wire_box['state'] == 'accepted'

    def test_per_box_accept_reverses_the_rejection_with_its_provenance(
        self, client: TestClient, fake_os: QueryFakeOpenSearch
    ) -> None:
        resp = client.patch(
            '/curation/projects/default/crops/rej/regions/b1', json={'state': 'accepted'}
        )
        assert resp.status_code == 200, resp.text
        _assert_reopened_with_provenance(fake_os)
        doc = _doc(fake_os, 'rej')
        assert doc[F.verified] is True
        assert doc[F.validated] is True
        assert doc[F.verifier] == 'human'

    def test_confirm_then_undo_restores_the_rejection(
        self, client: TestClient, fake_os: QueryFakeOpenSearch
    ) -> None:
        before = dict(_doc(fake_os, 'rej'))
        client.patch(
            '/curation/projects/default/crops/rej/region_meta',
            json={'region_status': 'detected', 'region_label_source': 'human'},
        )
        resp = client.post('/curation/projects/default/crops/rej/region/undo')
        assert resp.status_code == 200, resp.text
        doc = _doc(fake_os, 'rej')
        for key in (F.status, F.rejection_reason, F.boxes, F.count, F.rejected_count):
            assert doc.get(key) == before.get(key), key

    def test_false_positive_keeps_the_box_and_parks_it_in_the_fp_cluster(
        self, client: TestClient, fake_os: QueryFakeOpenSearch
    ) -> None:
        resp = client.patch(
            '/curation/projects/default/crops/rej/region_meta',
            json={'region_status': 'false_positive', 'region_label_source': 'human'},
        )
        assert resp.status_code == 200, resp.text
        doc = _doc(fake_os, 'rej')
        box = _box(fake_os, 'rej')
        assert doc[F.status] == 'false_positive'
        assert box['state'] == 'false_positive'
        assert box['bbox_norm'] == CANDIDATE
        assert box['detector'] == 'det_model'
        assert box['cluster_id'] == FALSE_POSITIVE_REGION_CLUSTER_ID
        # The verdict that made it a rejected box does not outlive the
        # transition (m5): a false-positive box carries no stale reason.
        assert box['rejection_reason'] is None

    def test_put_accepting_the_stored_box_is_a_confirmation(
        self, client: TestClient, fake_os: QueryFakeOpenSearch
    ) -> None:
        resp = client.put(
            '/curation/projects/default/crops/rej/regions',
            json={'boxes': [{'box_id': 'b1', 'bbox_norm': CANDIDATE, 'state': 'accepted'}]},
        )
        assert resp.status_code == 200, resp.text
        _assert_reopened_with_provenance(fake_os)
        assert _doc(fake_os, 'rej')[F.verifier] == 'human'

    def test_put_of_a_new_box_replaces_the_rejected_one(
        self, client: TestClient, fake_os: QueryFakeOpenSearch
    ) -> None:
        resp = client.put(
            '/curation/projects/default/crops/rej/regions',
            json={'boxes': [{'box_id': None, 'bbox_norm': [0.31, 0.61, 0.42, 0.66]}]},
        )
        assert resp.status_code == 200, resp.text
        doc = _doc(fake_os, 'rej')
        (box,) = doc[F.boxes]
        assert box['box_id'] == 'b2'
        assert box['detector'] == 'human'
        assert box['state'] == 'accepted'
        assert doc[F.status] == 'detected'
        assert doc[F.rejected_count] == 0
        assert doc.get(F.rejection_reason) is None

    def test_confirm_without_a_box_is_still_refused(self, client: TestClient) -> None:
        resp = client.patch(
            '/curation/projects/default/crops/legacy/region_meta',
            json={'region_status': 'detected', 'region_label_source': 'human'},
        )
        assert resp.status_code == 422

    def test_confirm_never_reopens_a_human_rejected_or_sanity_rejected_box(
        self, client: TestClient, fake_os: QueryFakeOpenSearch
    ) -> None:
        """W8-cleanup M3: a whole-set CONFIRM must only reopen a box the
        VERIFIER rejected -- never a human's own per-box rejection, and
        never a sanity-gate reject. Two boxes here: one rejected by a
        human, one by the sanity gate. Neither is reopenable, so CONFIRM
        must 422 (pre-W8 refused a confirm with no accepted box too)."""
        fake_os.docs(INDEX)['mixedrej'] = _item_with(
            'mixedrej',
            [
                _rejected_box(box_id='b1', rejection_reason=REJECT_REASON_HUMAN),
                _rejected_box(
                    box_id='b2',
                    bbox_norm=(0.3, 0.3, 0.31, 0.31),
                    rejection_reason=f'{REJECT_REASON_SANITY_PREFIX}degenerate_zero_size',
                ),
            ],
            'detection_failed',
        )
        resp = client.patch(
            '/curation/projects/default/crops/mixedrej/region_meta',
            json={'region_status': 'detected', 'region_label_source': 'human'},
        )
        assert resp.status_code == 422, resp.text
        doc = _doc(fake_os, 'mixedrej')
        assert [b['state'] for b in doc[F.boxes]] == ['rejected', 'rejected']

    def test_confirm_only_reopens_the_verifier_rejected_box_in_a_mixed_set(
        self, client: TestClient, fake_os: QueryFakeOpenSearch
    ) -> None:
        """A whole-set CONFIRM over a box a human rejected plus a box the
        VERIFIER rejected must reopen only the verifier one."""
        fake_os.docs(INDEX)['mixedrej2'] = _item_with(
            'mixedrej2',
            [
                _rejected_box(box_id='b1', rejection_reason=REJECT_REASON_HUMAN),
                _rejected_box(box_id='b2', score=0.9),
            ],
            'verify_rejected',
        )
        resp = client.patch(
            '/curation/projects/default/crops/mixedrej2/region_meta',
            json={'region_status': 'detected', 'region_label_source': 'human'},
        )
        assert resp.status_code == 200, resp.text
        by_id = {b['box_id']: b for b in _doc(fake_os, 'mixedrej2')[F.boxes]}
        assert by_id['b1']['state'] == 'rejected'
        assert by_id['b2']['state'] == 'accepted'


def test_undo_of_an_older_snapshot_leaves_fields_it_never_recorded() -> None:
    entry = {'kind': 'region', 'state': {F.status: 'detected'}}
    restored = restore_edit_state(entry, EditKind.REGION, current={F.revision: 4})
    assert restored[F.status] == 'detected'
    assert F.boxes not in restored
    # A restore is a write: the revision moves forward from the stored one.
    assert restored[F.revision] == 5
