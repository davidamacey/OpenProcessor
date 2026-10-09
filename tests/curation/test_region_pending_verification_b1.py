"""W8 B1 regression: a human's proposed box must survive re-verification.

Reproduces the review's exact scenario (docs/design/openprocessor_internal/
w8_pipeline_review_2026-09-27.md, finding B1): a human proposes a box via
the real ``PUT /crops/{crop_id}/regions`` route (W8a), which -- with no
``region_status`` in the request -- derives ``pending_verification``. The
streaming worker's Path 1 must read that stored ``proposed`` box (never a
single-box scalar, which this item never had), re-verify it through the VLM,
and merge the result back into the item's ``region_boxes`` list WITHOUT
discarding the untouched rejected sibling the human's PUT also carried
forward.

Before the B1 fix: Path 1 fired only on the legacy single-box scalar,
which this item never populated, so the
worker ran the segmenter cold and wrote a brand-new single-box list --
silently destroying both the human's proposed box and its sibling.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from _curation_app import mount_curation_routers
from fastapi import FastAPI
from fastapi.testclient import TestClient

from curation.test_regions_router import _FakeRegionOS
from src.config import get_region_fields
from src.routers.curation import _raw_opensearch_dep, router as curation_router
from src.services.labeling.region_overlay import VlmBoxVerdict
from src.services.labeling.vlm_models import VlmCombinedReply

from .test_region_cascade_integrity import _drive_worker, _FakeOpenSearch


if TYPE_CHECKING:
    from pathlib import Path


pytestmark = pytest.mark.usefixtures('reference_region_profile')

F = get_region_fields()

ITEM_BBOX = [0.1, 0.1, 0.9, 0.9]
# In SOURCE frame (the PUT route's default `frame='source'`), inside the
# item bbox above.
PROPOSED_BBOX = [0.3, 0.6, 0.6, 0.75]
SIBLING_BBOX = [0.15, 0.15, 0.2, 0.2]


def _seed_via_real_put_route() -> dict[str, Any]:
    """Seed a `pending_verification` item exactly the way a human does:
    a real `PUT /crops/{crop_id}/regions` call, no `region_status` in the
    body (so the route itself derives `pending_verification` from the
    resulting box list -- see `regions_boxes_edit._regions_put_build`).
    """
    fake_os = _FakeRegionOS(
        {
            'c1': {
                'crop_id': 'c1',
                'image_path': '/nonexistent/source.jpg',
                'bbox_norm': ITEM_BBOX,
                F.status: 'pending_detection',
                'class_name': 'sedan',
                'class_source': 'classifier_model',
                'confidence': 0.95,
                'created_at': '2026-09-24T00:00:00+00:00',
                F.boxes: [
                    {
                        'box_id': 'b2',
                        'bbox_norm': SIBLING_BBOX,
                        'state': 'rejected',
                        'score': 0.4,
                        'rejection_reason': 'verifier_no_verdict',
                    }
                ],
                F.box_seq: 2,
                F.revision: 1,
            }
        }
    )
    app = FastAPI()
    mount_curation_routers(app, curation_router)
    app.dependency_overrides[_raw_opensearch_dep] = lambda: fake_os
    with TestClient(app) as client:
        resp = client.put(
            '/curation/projects/default/crops/c1/regions',
            json={
                'boxes': [
                    {'box_id': 'b2'},  # untouched sibling
                    {'box_id': None, 'bbox_norm': PROPOSED_BBOX, 'state': 'proposed'},
                ]
            },
        )
    assert resp.status_code == 200, resp.text
    doc = fake_os._docs['c1']
    # Sanity-check the seed itself before handing it to the worker: the
    # human's PUT (no region_status) must have derived pending_verification
    # from the box list (a proposed box present, nothing accepted).
    assert doc[F.status] == 'pending_verification'
    box_ids = {b['box_id'] for b in doc[F.boxes]}
    assert box_ids == {'b2', 'b3'}
    proposed = next(b for b in doc[F.boxes] if b['box_id'] == 'b3')
    assert proposed['state'] == 'proposed'
    assert proposed['bbox_norm'] == PROPOSED_BBOX
    return doc


class TestPendingVerificationPreservesHumanBox:
    @pytest.mark.asyncio
    async def test_proposed_box_survives_reverification_with_sibling_intact(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seeded = _seed_via_real_put_route()
        fake_os = _FakeOpenSearch({'c1': seeded}, search_delay=0.0, lag_searches=0)

        reply = VlmCombinedReply(
            img_id='c1',
            region_visible=True,
            region_boxes=[
                VlmBoxVerdict(box=1, bbox_correct=True, confidence='high', text_reply='DNV20')
            ],
        )
        await _drive_worker(
            tmp_path,
            monkeypatch,
            fake_os=fake_os,
            primary=None,
            segmenter=None,
            reply=reply,
        )

        doc = fake_os.live['c1']
        boxes_by_id = {b['box_id']: b for b in doc[F.boxes]}

        # B1: the human's proposed box (b3) is still present -- resolved
        # by the VLM (accepted here), never silently dropped.
        assert 'b3' in boxes_by_id, f'human box b3 missing from {boxes_by_id}'
        assert boxes_by_id['b3']['state'] == 'accepted'
        assert boxes_by_id['b3']['bbox_norm'] == PROPOSED_BBOX

        # The untouched rejected sibling (b2) is still present too --
        # never overwritten just because b3 went through verification.
        assert 'b2' in boxes_by_id, f'sibling box b2 missing from {boxes_by_id}'
        assert boxes_by_id['b2']['state'] == 'rejected'
        assert boxes_by_id['b2']['bbox_norm'] == SIBLING_BBOX

        # An accepted box exists -> item status is detected.
        assert doc[F.status] == 'detected'

    @pytest.mark.asyncio
    async def test_proposed_box_rejected_by_vlm_still_kept_reviewable(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A VLM rejection of the re-verified box must still keep it (and
        its sibling) in the list -- never wipe the item back to empty."""
        seeded = _seed_via_real_put_route()
        fake_os = _FakeOpenSearch({'c1': seeded}, search_delay=0.0, lag_searches=0)

        reply = VlmCombinedReply(
            img_id='c1',
            region_visible=True,
            region_boxes=[VlmBoxVerdict(box=1, bbox_correct=False, confidence='high')],
        )
        await _drive_worker(
            tmp_path,
            monkeypatch,
            fake_os=fake_os,
            primary=None,
            segmenter=None,
            reply=reply,
        )

        doc = fake_os.live['c1']
        boxes_by_id = {b['box_id']: b for b in doc[F.boxes]}
        assert 'b3' in boxes_by_id
        assert boxes_by_id['b3']['state'] == 'rejected'
        assert 'b2' in boxes_by_id
        assert doc[F.status] == 'verify_rejected'
