"""Machine auto-confirm is not human validation (DQ-M1).

``region_validated`` means a human confirmed the region; only human
writers set it. When the worker's auto-confirm policy accepts a box
(detector + verifier agree strongly enough) it records that as
``region_auto_confirmed`` and leaves ``region_validated`` false, so the
box stays in the human region-review queue while still being an accepted
(``detected``) region for export.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from src.config import get_region_fields
from src.config.curation import base_curation_config
from src.services.curation.review_queries import build_tab_query
from src.services.curation.wire import serialize_item
from src.services.detection.cascade_detect.candidate import RegionCandidate
from src.services.labeling.region_overlay import VlmBoxVerdict
from src.services.labeling.vlm_models import VlmCombinedReply

from .test_region_cascade_integrity import _drive_worker, _FakeOpenSearch, _item
from .test_region_text_worker import _drive


if TYPE_CHECKING:
    from pathlib import Path


F = get_region_fields()
INDEX = base_curation_config().items_index


class TestWorkerWrites:
    @pytest.mark.asyncio
    @pytest.mark.usefixtures('reference_region_profile')
    async def test_streaming_worker_combined_accept_writes_an_accepted_box(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """W8: the combined-call accept path writes the box-list shape
        (``region_boxes[i].state == 'accepted'``) AND restores the
        item-level ``region_verified``/``region_auto_confirmed``/
        ``region_validated`` fields (W8 M2 fix, pipeline-wiring review
        2026-09-27) -- these stopped being written when the box-list
        rewrite landed, going blind every reader that still filters on
        them (regions.py's ``verified`` filter, the training-candidate
        cohorts, region_requeue.py). ``region_validated`` stays human-only
        (unchanged, DQ-M1); ``region_auto_confirmed`` is the box-aware
        rule: >=1 accepted box, VLM confidence 'high' here, so it fires.
        """
        fake = _FakeOpenSearch({'c1': _item()}, search_delay=0.0, lag_searches=0)
        await _drive_worker(
            tmp_path,
            monkeypatch,
            fake_os=fake,
            primary=RegionCandidate(bbox_norm=(0.3, 0.6, 0.6, 0.75), score=0.9, source='det'),
            segmenter=None,
            reply=VlmCombinedReply(
                img_id='c1',
                region_visible=True,
                region_boxes=[
                    VlmBoxVerdict(box=1, bbox_correct=True, confidence='high', text_reply='DNV20')
                ],
            ),
        )
        doc = fake.live['c1']
        assert doc[F.status] == 'detected'
        assert doc[F.boxes][0]['state'] == 'accepted'
        # M2: item-level verification fields restored.
        assert doc[F.verified] is True
        assert doc[F.validated] is False
        assert doc[F.auto_confirmed] is True
        assert doc[F.verifier] is not None
        assert doc[F.verifier_version] is not None
        assert doc[F.verified_at] is not None

    @pytest.mark.asyncio
    @pytest.mark.usefixtures('reference_region_profile')
    async def test_streaming_worker_low_confidence_accept_is_not_auto_confirmed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """M2 box-aware rule: an accepted box the VLM only rated 'medium'
        confidence, off a detector score below the auto-confirm floor,
        is verified (the VLM did answer) but NOT auto-confirmed."""
        fake = _FakeOpenSearch({'c1': _item()}, search_delay=0.0, lag_searches=0)
        await _drive_worker(
            tmp_path,
            monkeypatch,
            fake_os=fake,
            primary=RegionCandidate(bbox_norm=(0.3, 0.6, 0.6, 0.75), score=0.5, source='det'),
            segmenter=None,
            reply=VlmCombinedReply(
                img_id='c1',
                region_visible=True,
                region_boxes=[VlmBoxVerdict(box=1, bbox_correct=True, confidence='medium')],
            ),
        )
        doc = fake.live['c1']
        assert doc[F.status] == 'detected'
        assert doc[F.verified] is True
        assert doc[F.auto_confirmed] is False
        assert doc[F.validated] is False

    @pytest.mark.asyncio
    @pytest.mark.usefixtures('reference_region_profile')
    async def test_streaming_worker_no_vlm_path_is_unverified_and_unconfirmed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """M2: the no-VLM-configured accept path (``accept_without_vlm``)
        never calls the VLM -- verified/auto_confirmed must both stay
        False, the same as the pre-W8 skip-verify write."""
        fake = _FakeOpenSearch({'c1': _item()}, search_delay=0.0, lag_searches=0)
        await _drive(
            tmp_path,
            monkeypatch,
            fake_os=fake,
            primary=RegionCandidate(bbox_norm=(0.3, 0.6, 0.6, 0.75), score=0.9, source='det'),
            segmenter=None,
            vlm_url='',
        )
        doc = fake.live['c1']
        assert doc[F.status] == 'detected'
        assert doc[F.boxes][0]['state'] == 'accepted'
        assert doc[F.verified] is False
        assert doc[F.auto_confirmed] is False
        assert doc[F.validated] is False


def test_auto_confirmed_region_is_on_the_wire_and_not_label_validated() -> None:
    item = serialize_item(
        {'crop_id': 'x', F.status: 'detected', F.auto_confirmed: True, F.validated: False},
        'x',
        api_prefix='',
    )
    assert item['region_auto_confirmed'] is True
    assert item['region_validated'] is False
    assert item['label_validated'] is False


def test_region_review_tab_keeps_auto_confirmed_regions() -> None:
    """The region tab excludes only human-validated regions, so an
    auto-confirmed (unvalidated) region is in the human queue."""
    _must, must_not, _reason = build_tab_query(
        'regions', include_test=True, text=None, max_rank=None
    )
    assert {'term': {F.validated: True}} in must_not
    assert not any(F.auto_confirmed in str(clause) for clause in must_not)
