"""W8c M2 regression: requeue to ``pending_verification``, then drive the
REAL streaming worker end-to-end, and confirm Path 1 actually re-verifies
the requeued box instead of silently falling through to a fresh detection
pass.

Before the fix: a ``verify_rejected`` item's box is always ``rejected``
(a ``REQUEUEABLE_STATUSES`` item can never already carry a ``proposed``
box -- ``derive_status`` would report ``pending_verification`` instead),
so ``apply_requeue`` flipped the item's status without ever giving Path 1
a ``proposed`` box to find. The worker's Path 1 guard
(a stored ``proposed`` box) then
failed, and the item silently ran a fresh detection pass instead -- the
documented operator action ("Re-verify boxes the previous verify prompt
rejected", ``requeue_regions.py``) did something else entirely.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from curation.query_fakes import QueryFakeOpenSearch
from src.config import CurationConfig, RegionStatus, get_region_fields
from src.services.curation.region_requeue import RequeueSelection, apply_requeue
from src.services.labeling.region_overlay import VlmBoxVerdict
from src.services.labeling.vlm_labeler import VlmCombinedReply

from .test_region_cascade_integrity import _drive_worker, _FakeOpenSearch


if TYPE_CHECKING:
    from pathlib import Path


pytestmark = pytest.mark.usefixtures('reference_region_profile')

F = get_region_fields()
ITEMS = 'test_items'
CFG = CurationConfig(items_index=ITEMS)

REJECTED_BBOX = [0.3, 0.6, 0.6, 0.75]


class TestRequeueToPendingVerificationThenWorker:
    @pytest.mark.asyncio
    async def test_requeued_rejected_box_is_reverified_not_fresh_detected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        doc = {
            'crop_id': 'c1',
            'image_path': '/nonexistent/source.jpg',
            'bbox_norm': [0.1, 0.1, 0.9, 0.9],
            F.status: RegionStatus.VERIFY_REJECTED.value,
            'class_name': 'sedan',
            'class_source': 'classifier_model',
            'confidence': 0.95,
            'created_at': '2026-09-24T00:00:00+00:00',
            F.boxes: [
                {
                    'box_id': 'b1',
                    'bbox_norm': REJECTED_BBOX,
                    'state': 'rejected',
                    'score': 0.4,
                    'detector': 'old_det',
                    'detector_version': '1',
                    'source': 'detector',
                    'rejection_reason': 'region_visible_elsewhere',
                }
            ],
            F.rejected_count: 1,
        }
        requeue_fake = QueryFakeOpenSearch({ITEMS: {'c1': doc}})
        sel = RequeueSelection(
            RegionStatus.VERIFY_REJECTED, target=RegionStatus.PENDING_VERIFICATION
        )
        totals = await apply_requeue(requeue_fake, sel, config=CFG)
        assert totals['updated'] == 1

        requeued_doc = requeue_fake.docs(ITEMS)['c1']
        assert requeued_doc[F.status] == RegionStatus.PENDING_VERIFICATION.value
        assert requeued_doc[F.boxes][0]['state'] == 'proposed'
        assert requeued_doc[F.boxes][0]['rejection_reason'] is None

        # Hand the requeued doc to the streaming worker's own fake and run
        # it for real -- this is the part `apply_requeue`'s own unit tests
        # can't see: whether the worker actually re-verifies.
        fake_os = _FakeOpenSearch({'c1': requeued_doc}, search_delay=0.0, lag_searches=0)
        reply = VlmCombinedReply(
            img_id='c1',
            region_visible=True,
            region_boxes=[
                VlmBoxVerdict(box=1, bbox_correct=True, confidence='high', text_reply='ABC123')
            ],
        )
        mocks = await _drive_worker(
            tmp_path,
            monkeypatch,
            fake_os=fake_os,
            primary=None,
            segmenter=None,
            reply=reply,
        )

        # Path 1 ran (the item `continue`d out of the loop before ever
        # reaching Path 2's primary-detector call): no fresh detection.
        mocks['primary'].detect_batch_multi.assert_not_awaited()
        doc_after = fake_os.live['c1']
        assert doc_after[F.status] == RegionStatus.DETECTED.value
        boxes_by_id = {b['box_id']: b for b in doc_after[F.boxes]}
        # The SAME box (by id and geometry) was re-verified, not replaced
        # by a brand-new candidate.
        assert boxes_by_id['b1']['state'] == 'accepted'
        assert boxes_by_id['b1']['bbox_norm'] == REJECTED_BBOX
