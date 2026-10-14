"""No-verdict handling, coordinate re-projection, provenance and the sanity gate of the region worker."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from curation.region_cascade_helpers import _accept, _drive_worker, _FakeOpenSearch
from src.config import get_region_fields
from src.services.detection.cascade_detect.candidate import RegionCandidate
from src.services.detection.cascade_detect.sanity import crop_norm_to_source_norm


if TYPE_CHECKING:
    from pathlib import Path
from curation.region_worker_helpers import _item_with, _proposed_box_fields


# The cascade needs an active region profile; the default is none.
pytestmark = pytest.mark.usefixtures('reference_region_profile')


# =============================================================================
# No-verdict handling: absence of a VLM answer must never read as a reject
# =============================================================================


class TestNoVerdictLeavesItemPending:
    @pytest.mark.asyncio
    async def test_pending_verify_no_verdict_never_falls_through_to_sam3(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A combined reply with no usable verdict (unparseable / no
        entry) must not be treated as a reject -- it retries (see
        ``test_region_no_verdict_cap.py`` for the retry-then-cap
        contract in full) rather than falling through to the secondary
        segmenter. This drives the pipeline all the way to the cap
        (``verify_rejected``, reason ``verifier_no_verdict``) and checks
        the one thing specific to this routing row: the secondary
        segmenter is never called, on any attempt."""
        F = get_region_fields()
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
            segmenter=RegionCandidate(bbox_norm=(0.4, 0.5, 0.6, 0.55), score=0.77),
            reply=_accept(),
            combined_side_effect=lambda crops, **_kw: dict.fromkeys(
                (c.crop_id for c in crops), None
            ),
        )
        assert fake_os.live['c1'][F.status] == 'verify_rejected'
        mocks['seg'].segment_multi.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_pending_car_detector_no_verdict_never_falls_through_to_sam3(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
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
            segmenter=RegionCandidate(bbox_norm=(0.6, 0.6, 0.7, 0.65), score=0.79),
            reply=_accept(),
            combined_side_effect=lambda crops, **_kw: dict.fromkeys(
                (c.crop_id for c in crops), None
            ),
        )
        assert fake_os.live['c1'][F.status] == 'verify_rejected'
        mocks['seg'].segment_multi.assert_not_awaited()


# =============================================================================
# Coordinate re-projection
# =============================================================================


class TestReprojection:
    @pytest.mark.asyncio
    async def test_sam3_bbox_projected_to_source_frame(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The secondary segmenter emits crop-frame coords; the worker
        must re-project them before writing."""
        F = get_region_fields()
        # Region-shaped crop-frame bbox (aspect = 4) so the Phase A3
        # sanity gate would admit it even if it ran here; below the
        # skip-verify score so it goes through the ordinary combined
        # verify path.
        sam_cand = RegionCandidate(bbox_norm=(0.5, 0.85, 0.9, 0.95), score=0.7, source='sam3')
        vehicle = (0.20, 0.10, 0.60, 0.50)
        fake_os = _FakeOpenSearch(
            {'c1': _item_with(status='pending_detection', bbox_norm=list(vehicle))},
            search_delay=0.0,
            lag_searches=0,
        )
        await _drive_worker(
            tmp_path,
            monkeypatch,
            fake_os=fake_os,
            primary=None,
            segmenter=sam_cand,
            reply=_accept(),
            visible=True,
        )
        doc = fake_os.live['c1']
        expected = list(crop_norm_to_source_norm(sam_cand.bbox_norm, vehicle))
        # Sanity: the projected box should be inside the vehicle box.
        assert expected[0] >= vehicle[0]
        assert expected[2] <= vehicle[2]
        assert doc[F.boxes][0]['bbox_norm'] == pytest.approx(expected)


# =============================================================================
# Provenance + sanity gate (Phase A2 / A3)
# =============================================================================


class TestProvenance:
    @pytest.mark.asyncio
    async def test_detector_write_carries_provenance(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A successful primary-detector write must stamp detector +
        version + a detected_at timestamp on the box."""
        F = get_region_fields()
        detector_cand = RegionCandidate(
            bbox_norm=(0.3, 0.4, 0.5, 0.45), score=0.82, source='license_plate_detector'
        )
        fake_os = _FakeOpenSearch(
            {'c1': _item_with(status='pending_detection')}, search_delay=0.0, lag_searches=0
        )
        await _drive_worker(
            tmp_path,
            monkeypatch,
            fake_os=fake_os,
            primary=detector_cand,
            segmenter=None,
            reply=_accept(),
        )
        doc = fake_os.live['c1']
        box = doc[F.boxes][0]
        assert box['detector'] == 'license_plate_detector'
        assert box['detector_version'] == '1'
        assert box['detected_at'] is not None
        assert box['state'] == 'accepted'
        # Chain captures the cascade: hit + combined_verify_ok.
        chain = doc.get(F.detector_chain) or []
        assert any('license_plate_detector:hit' in s for s in chain)
        assert any('combined_verify_ok' in s for s in chain)

    @pytest.mark.asyncio
    async def test_sam3_write_carries_sam3_detector(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        F = get_region_fields()
        sam_cand = RegionCandidate(bbox_norm=(0.30, 0.40, 0.45, 0.45), score=0.78, source='sam3')
        fake_os = _FakeOpenSearch(
            {'c1': _item_with(status='pending_detection')}, search_delay=0.0, lag_searches=0
        )
        await _drive_worker(
            tmp_path,
            monkeypatch,
            fake_os=fake_os,
            primary=None,
            segmenter=sam_cand,
            reply=_accept(),
            visible=True,
        )
        doc = fake_os.live['c1']
        box = doc[F.boxes][0]
        assert box['detector'] == 'sam3'
        chain = doc.get(F.detector_chain) or []
        # A primary-detector miss must be recorded so training-set
        # selection can find it.
        assert any('license_plate_detector:miss' in s for s in chain)
        assert any('sam3:hit' in s for s in chain)

    @pytest.mark.asyncio
    async def test_large_detector_bbox_no_longer_sanity_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A square bbox covering ~64% of the crop. A naive shape-prior
        # gate would reject this on aspect/size heuristics; the
        # geometry-only gate lets a well-formed box flow to the VLM —
        # which here confirms it — and it's written as a primary-detector
        # region. Leaning-motorcycle regions legitimately produce large,
        # square-ish axis-aligned boxes.
        F = get_region_fields()
        big_cand = RegionCandidate(
            bbox_norm=(0.1, 0.1, 0.9, 0.9), score=0.95, source='license_plate_detector'
        )
        fake_os = _FakeOpenSearch(
            {'c1': _item_with(status='pending_detection')}, search_delay=0.0, lag_searches=0
        )
        await _drive_worker(
            tmp_path,
            monkeypatch,
            fake_os=fake_os,
            primary=big_cand,
            segmenter=None,
            reply=_accept(),
        )
        doc = fake_os.live['c1']
        chain = doc.get(F.detector_chain) or []
        assert not any('sanity_reject' in s for s in chain)
        box = doc[F.boxes][0]
        assert box['detector'] == 'license_plate_detector'
        assert box['bbox_norm'] is not None
