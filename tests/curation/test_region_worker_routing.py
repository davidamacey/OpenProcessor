"""Routing rules of the cascade's decision table, driven through the real streaming pipeline, and the secondary-shape helper."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

import scripts.curation.worker.state as worker_state
from curation.region_cascade_helpers import _accept, _drive_worker, _FakeOpenSearch, _item, _profile
from scripts.curation.worker.client import SegmenterRequestFailed
from src.config import get_region_fields
from src.services.detection.cascade_detect.candidate import RegionCandidate
from src.services.detection.cascade_detect.sanity import crop_norm_to_source_norm


if TYPE_CHECKING:
    from pathlib import Path
from curation.region_worker_helpers import (
    _drive_text_hint_rescue,
    _item_with,
    _make_task,
    _proposed_box_fields,
    _reject,
)


# The cascade needs an active region profile; the default is none.
pytestmark = pytest.mark.usefixtures('reference_region_profile')


# =============================================================================
# Routing rules
# =============================================================================


class TestRouting:
    @pytest.mark.asyncio
    async def test_pending_verify_accepted_writes_detected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A stored proposed box verified by the VLM -> status='detected',
        the existing bbox is kept (round-tripped through crop/source
        frames)."""
        F = get_region_fields()
        detector_region_in_source = (0.20, 0.30, 0.30, 0.34)
        fake_os = _FakeOpenSearch(
            {
                'c1': _item_with(
                    status='pending_verification',
                    **_proposed_box_fields(detector_region_in_source, 0.91),
                )
            },
            search_delay=0.0,
            lag_searches=0,
        )
        await _drive_worker(
            tmp_path,
            monkeypatch,
            fake_os=fake_os,
            primary=None,
            segmenter=None,
            reply=_accept(),
        )
        doc = fake_os.live['c1']
        assert doc[F.status] == 'detected'
        box = doc[F.boxes][0]
        assert box['bbox_norm'] == pytest.approx(list(detector_region_in_source))
        assert box['score'] == pytest.approx(0.91)
        assert box['state'] == 'accepted'

    @pytest.mark.asyncio
    async def test_pending_verify_rejected_is_terminal_not_a_sam3_fallback(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A combined-verify reject on the existing candidate is a
        terminal ``verify_rejected`` write for this pass -- there is no
        intra-pass fallback to the secondary segmenter (the old
        two-call ``verify_region`` cascade did that; the single combined
        call doesn't)."""
        F = get_region_fields()
        sam_cand = RegionCandidate(bbox_norm=(0.4, 0.5, 0.6, 0.55), score=0.77, source='sam3')
        fake_os = _FakeOpenSearch(
            {
                'c1': _item_with(
                    status='pending_verification',
                    **_proposed_box_fields((0.2, 0.2, 0.3, 0.22), 0.7),
                )
            },
            search_delay=0.0,
            lag_searches=0,
        )
        mocks = await _drive_worker(
            tmp_path,
            monkeypatch,
            fake_os=fake_os,
            primary=None,
            segmenter=sam_cand,
            reply=_reject(),
        )
        doc = fake_os.live['c1']
        assert doc[F.status] == 'verify_rejected'
        box = doc[F.boxes][0]
        assert box['state'] == 'rejected'
        assert box['bbox_correct'] is False
        mocks['seg'].segment_multi.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_pending_secondary_shape_skips_detector_calls_sam3(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Secondary-shape pending -> the primary detector is NOT called,
        the secondary segmenter runs (via the visibility pre-filter)."""
        F = get_region_fields()
        sam_cand = RegionCandidate(bbox_norm=(0.4, 0.45, 0.5, 0.50), score=0.81, source='sam3')
        fake_os = _FakeOpenSearch(
            {'c1': _item_with(status='pending_detection', class_name='some-class')},
            search_delay=0.0,
            lag_searches=0,
        )
        mocks = await _drive_worker(
            tmp_path,
            monkeypatch,
            fake_os=fake_os,
            primary=None,
            segmenter=sam_cand,
            reply=_accept(),
            visible=True,
            class_group=lambda _name: 'group_a',
            profile_overrides={'secondary_shape_groups': frozenset({'group_a'})},
        )
        mocks['primary'].detect_batch_multi.assert_not_awaited()
        assert fake_os.live['c1'][F.status] == 'detected'

    @pytest.mark.asyncio
    async def test_pending_car_runs_detector_first_and_succeeds(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Non-secondary-shape pending -> primary hit + combined accept ->
        done, the secondary segmenter is never called."""
        F = get_region_fields()
        detector_cand = RegionCandidate(
            bbox_norm=(0.3, 0.4, 0.5, 0.45), score=0.82, source='license_plate_detector'
        )
        fake_os = _FakeOpenSearch(
            {'c1': _item_with(status='pending_detection')}, search_delay=0.0, lag_searches=0
        )
        mocks = await _drive_worker(
            tmp_path,
            monkeypatch,
            fake_os=fake_os,
            primary=detector_cand,
            segmenter=None,
            reply=_accept(),
        )
        mocks['seg'].segment_multi.assert_not_awaited()
        doc = fake_os.live['c1']
        assert doc[F.status] == 'detected'
        expected = crop_norm_to_source_norm(detector_cand.bbox_norm, tuple(_item()['bbox_norm']))
        assert doc[F.boxes][0]['bbox_norm'] == pytest.approx(list(expected))

    @pytest.mark.asyncio
    async def test_pending_car_detector_rejected_is_terminal_not_a_sam3_fallback(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Same note as the pending_verify case: a primary-
        detector hit that the combined VLM call rejects is terminal, not a
        trigger to try the secondary segmenter within the same pass."""
        F = get_region_fields()
        detector_cand = RegionCandidate(
            bbox_norm=(0.3, 0.4, 0.5, 0.45), score=0.82, source='license_plate_detector'
        )
        sam_cand = RegionCandidate(bbox_norm=(0.6, 0.6, 0.7, 0.65), score=0.79, source='sam3')
        fake_os = _FakeOpenSearch(
            {'c1': _item_with(status='pending_detection')}, search_delay=0.0, lag_searches=0
        )
        mocks = await _drive_worker(
            tmp_path,
            monkeypatch,
            fake_os=fake_os,
            primary=detector_cand,
            segmenter=sam_cand,
            reply=_reject(),
        )
        doc = fake_os.live['c1']
        assert doc[F.status] == 'verify_rejected'
        mocks['seg'].segment_multi.assert_not_awaited()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        'status', ['detected', 'no_region_visible', 'verify_rejected', 'no_region_box']
    )
    async def test_terminal_status_is_never_touched(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, status: str
    ) -> None:
        """Terminal statuses must never be picked up by the pending query
        nor re-written: the producer's query filter excludes them, and
        stage A's own terminal-status check is a redundant safety net for
        a status that changed out from under an in-flight fetch."""
        fake_os = _FakeOpenSearch({'c1': _item(status=status)}, search_delay=0.0, lag_searches=0)
        mocks = await _drive_worker(
            tmp_path,
            monkeypatch,
            fake_os=fake_os,
            primary=None,
            segmenter=None,
            reply=_accept(),
            until_writes=0,
        )
        assert fake_os.writes == []
        mocks['vlm'].label_combined_batch.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_all_detectors_miss_marks_no_region_box(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        F = get_region_fields()
        fake_os = _FakeOpenSearch(
            {'c1': _item_with(status='pending_detection')}, search_delay=0.0, lag_searches=0
        )
        await _drive_worker(
            tmp_path,
            monkeypatch,
            fake_os=fake_os,
            primary=None,
            segmenter=None,
            reply=_accept(),
            visible=True,
        )
        doc = fake_os.live['c1']
        # Phase A2: every write carries a detector_chain. Status remains
        # 'no_region_box' (queue for human review) and the chain captures
        # which detectors were tried — used by the detector-blind-spot
        # training-set selector.
        assert doc[F.status] == 'no_region_box'
        chain = doc.get(F.detector_chain) or []
        assert any(f'{_profile().detector_model}:miss' in s for s in chain)
        # The reference profile has text_hint_enabled=True, so a plain
        # segmenter miss is never recorded on its own -- the segmenter's
        # global attempt folds into the text-hint branch, which records
        # its own miss tag instead (pre-existing runner.py behavior, not
        # touched by this pass).
        assert any(f'{_profile().ocr_rec_model}:text_hint:miss' in s for s in chain)

    @pytest.mark.asyncio
    async def test_text_hint_sam3_rescue_succeeds(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Primary + secondary globally miss but the OCR pipeline locates
        plate-shaped text; the secondary segmenter re-prompted on a tight
        sub-crop produces the final box.

        The OCR-detection geometry is intentionally NEVER written as a
        region bbox — that path produced visibly-oversized boxes. The
        final detector stamped on the box is the secondary segmenter,
        not paddleocr_det_trt.
        """
        F = get_region_fields()
        ocr_pick = MagicMock()
        ocr_pick.bbox_norm = (0.40, 0.40, 0.55, 0.45)
        ocr_pick.text = 'ABC1234'
        ocr_pick.rec_score = 0.85
        sub_sam_cand = RegionCandidate(
            bbox_norm=(0.20, 0.30, 0.80, 0.60), score=0.74, source='sam3'
        )
        fake_os = _FakeOpenSearch(
            {'c1': _item_with(status='pending_detection')}, search_delay=0.0, lag_searches=0
        )
        await _drive_text_hint_rescue(
            tmp_path,
            monkeypatch,
            fake_os=fake_os,
            ocr_pick=ocr_pick,
            sub_sam_cand=sub_sam_cand,
            reply=_accept(),
        )
        doc = fake_os.live['c1']
        assert doc[F.status] == 'detected'
        box = doc[F.boxes][0]
        # Provenance: the secondary segmenter owns the final geometry.
        # The OCR-detection model never appears as the detector for a
        # text-hint write.
        assert box['detector'] == 'sam3'
        chain = doc.get(F.detector_chain) or []
        assert any('paddleocr_rec_trt:text_hint:hit' in s for s in chain)
        assert any('sam3:combined_verify_ok' in s for s in chain)

    @pytest.mark.asyncio
    async def test_text_hint_segmenter_failure_leaves_the_crop_pending(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Found in review: a failed sub-crop segmenter request was recorded
        as ``text_hint:miss`` and the crop finalized as ``no_region_box``.
        An unavailable segmenter must leave the crop untouched instead."""
        F = get_region_fields()
        ocr_pick = MagicMock()
        ocr_pick.bbox_norm = (0.40, 0.40, 0.55, 0.45)
        ocr_pick.text = 'ABC1234'
        ocr_pick.rec_score = 0.85
        fake_os = _FakeOpenSearch(
            {'c1': _item_with(status='pending_detection')}, search_delay=0.0, lag_searches=0
        )
        mocks = await _drive_text_hint_rescue(
            tmp_path,
            monkeypatch,
            fake_os=fake_os,
            ocr_pick=ocr_pick,
            sub_sam_cand=SegmenterRequestFailed('sub-crop request failed'),
            reply=_accept(),
        )
        assert mocks['seg'].segment.await_count >= 1
        assert fake_os.live['c1'][F.status] == 'pending_detection'
        assert not fake_os.writes


# =============================================================================
# is_secondary_shape helper
# =============================================================================


class TestIsSecondaryShape:
    """Group resolution now comes from the class registry
    (class_name -> group), not the dead ``_ItemTask.group`` field nothing
    ever wrote or mapped on the item doc."""

    def test_explicit_groups(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for grp in worker_state.region_profile().secondary_shape_groups:
            monkeypatch.setattr(worker_state, '_class_group', lambda _name, grp=grp: grp)
            assert worker_state._is_secondary_shape(_make_task(class_name='some-class'))

    def test_non_secondary_groups(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for grp in ('cars', 'sportycars', 'exotics'):
            monkeypatch.setattr(worker_state, '_class_group', lambda _name, grp=grp: grp)
            assert not worker_state._is_secondary_shape(_make_task(class_name='audi'))

    def test_no_registry_group_is_never_secondary_shape(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Class not in the registry (no group resolves) -- there is no
        # private-naming-convention fallback; a group-less item is simply
        # not routed to the secondary-shape path.
        monkeypatch.setattr(worker_state, '_class_group', lambda _name: None)
        t = _make_task(class_name='class_b')
        assert not worker_state._is_secondary_shape(t)

    def test_registry_group_takes_priority_over_class_name(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The registry's group answer is the only source of truth --
        the class name itself is never inspected."""
        monkeypatch.setattr(worker_state, '_class_group', lambda _name: 'cars')
        t = _make_task(class_name='class_b')
        assert not worker_state._is_secondary_shape(t)
