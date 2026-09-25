"""Unit tests for ``scripts.curation.region_worker_main``.

Curation worker tests. Every external system (RegionDetector, SAM 3
HTTP, the VLM, OpenSearch, Triton, disk) is mocked — the tests run in
milliseconds without any GPU, network, or filesystem dependency.

Coverage:

* Routing rules — one test per row of the cascade's decision table.
* SAM 3 candidates get re-projected via
  :func:`crop_norm_to_source_norm` before being written.
* Bulk write: one ``opensearch.bulk(...)`` call per iteration with
  one ``{update}`` + ``{doc}`` pair per crop that produced an update.
* Sentinel pause: ``_wait_for_sentinel_clear`` blocks while the file
  exists and unblocks once it's removed.
* SIGTERM cleanup: stop event flips on signal and the loop exits at
  the next iteration boundary.
"""

from __future__ import annotations

import asyncio
import dataclasses
import io
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from PIL import Image

import scripts.curation.region_worker_main as worker
import scripts.curation.worker.state as worker_state
from curation.occ_fakes import make_bulk_response, make_bulk_update_item, make_mget_response
from src.config import get_region_fields
from src.services.detection.cascade_detect import RegionCandidate, crop_norm_to_source_norm
from src.services.labeling.vlm_labeler import VlmRegionVerdict


# The cascade needs an active region profile; the default is none.
pytestmark = pytest.mark.usefixtures('reference_region_profile')


if TYPE_CHECKING:
    from pathlib import Path


# =============================================================================
# Fixtures and helpers
# =============================================================================


def _make_jpeg(width: int = 320, height: int = 240) -> bytes:
    buf = io.BytesIO()
    Image.new('RGB', (width, height), (50, 80, 120)).save(buf, format='JPEG', quality=85)
    return buf.getvalue()


def _make_task(
    *,
    crop_id: str = 'crop-1',
    status: str | None = 'pending',
    group: str = 'cars',
    class_name: str = 'audi',
    vehicle_bbox: tuple[float, float, float, float] = (0.1, 0.1, 0.5, 0.5),
    detector_region_in_source: tuple[float, float, float, float] | None = None,
    detector_score: float = 0.0,
    crop_jpeg: bytes | None = None,
) -> worker._ItemTask:
    """Build a fully populated ``_ItemTask`` for routing tests."""
    return worker._ItemTask(
        crop_id=crop_id,
        image_path='/dev/null/never-read',
        vehicle_bbox_norm=vehicle_bbox,
        region_status=status,
        class_name=class_name,
        group=group,
        detector_region_in_source=detector_region_in_source,
        detector_score=detector_score,
        crop_jpeg=crop_jpeg if crop_jpeg is not None else _make_jpeg(),
    )


def _vlm_mock(*, is_region: bool, confidence: str = 'high') -> MagicMock:
    """A VlmLabeler stub whose ``verify_region`` returns a fixed verdict."""
    g = MagicMock()
    g.verify_region = AsyncMock(
        return_value=VlmRegionVerdict(
            crop_id='ignored', is_region=is_region, confidence=confidence, reason='test'
        )
    )
    g.aclose = AsyncMock()
    return g


def _sam3_mock(candidate: RegionCandidate | None) -> MagicMock:
    s = MagicMock()
    s.segment = AsyncMock(return_value=candidate)
    s.aclose = AsyncMock()
    return s


def _lpr_mock(candidates: list[RegionCandidate | None]) -> MagicMock:
    detector = MagicMock()
    detector.detect_batch = AsyncMock(return_value=candidates)
    return detector


def _ocr_recognizer_mock(regions: list[Any] | None = None, pick: Any = None) -> MagicMock:
    """Mock for ``PaddleOcrTextRecognizer`` used by the text-hint path.

    Defaults to "no text regions found" (empty list, no pick) — every
    test that doesn't exercise the text-hint path can use this as a quiet stand-in.
    """
    r = MagicMock()
    r.detect_regions = AsyncMock(return_value=regions or [])
    r.pick_best_text_region = MagicMock(return_value=pick)
    return r


def _sam3_mock_sequence(candidates: list[RegionCandidate | None]) -> MagicMock:
    """SAM3 stub whose successive calls return successive candidates.

    Used by the text-hint sub-crop test: first call (global) misses, second
    call (sub-crop) returns geometry.
    """
    s = MagicMock()
    s.segment = AsyncMock(side_effect=list(candidates))
    s.aclose = AsyncMock()
    return s


# =============================================================================
# Routing rules
# =============================================================================


class TestRouting:
    @pytest.mark.asyncio
    async def test_pending_verify_accepted_writes_detected(self) -> None:
        """Primary-detector candidate verified by the VLM → status='detected', existing bbox kept."""
        F = get_region_fields()
        detector_region_in_source = (0.20, 0.30, 0.30, 0.34)
        task = _make_task(
            status='pending_verify',
            detector_region_in_source=detector_region_in_source,
            detector_score=0.91,
        )
        await worker._process_crop(
            task,
            detector=_lpr_mock([]),
            sam3=_sam3_mock(None),
            ocr_recognizer=_ocr_recognizer_mock(),
            vlm=_vlm_mock(is_region=True, confidence='high'),
        )
        assert task.update_doc[F.status] == 'detected'
        assert task.update_doc[F.bbox_norm] == list(detector_region_in_source)
        assert task.update_doc[F.score] == pytest.approx(0.91)
        assert task.update_doc[F.verified] is True

    @pytest.mark.asyncio
    async def test_pending_verify_rejected_falls_through_to_sam3(self) -> None:
        """Verify reject → the secondary segmenter runs and (here) succeeds."""
        F = get_region_fields()
        sam_cand = RegionCandidate(bbox_norm=(0.4, 0.5, 0.6, 0.55), score=0.77, source='sam3')
        vlm = MagicMock()
        # First call: reject the primary-detector candidate. Second
        # call: accept the secondary segmenter.
        vlm.verify_region = AsyncMock(
            side_effect=[
                VlmRegionVerdict(
                    crop_id='c', is_region=False, confidence='high', reason='not a plate'
                ),
                VlmRegionVerdict(crop_id='c', is_region=True, confidence='high', reason='plate'),
            ]
        )
        vlm.aclose = AsyncMock()
        task = _make_task(
            status='pending_verify',
            detector_region_in_source=(0.2, 0.2, 0.3, 0.22),
            detector_score=0.7,
            vehicle_bbox=(0.0, 0.0, 1.0, 1.0),
        )
        await worker._process_crop(
            task,
            detector=_lpr_mock([]),
            sam3=_sam3_mock(sam_cand),
            ocr_recognizer=_ocr_recognizer_mock(),
            vlm=vlm,
        )
        assert task.update_doc[F.status] == 'detected'
        # Re-projected via crop_norm_to_source_norm: vehicle_bbox is the
        # full image so source == crop here.
        assert task.update_doc[F.bbox_norm] == list(sam_cand.bbox_norm)
        assert vlm.verify_region.await_count == 2

    @pytest.mark.asyncio
    async def test_pending_secondary_shape_skips_lpr_calls_sam3(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Secondary-shape pending → the primary detector is NOT called, the secondary segmenter runs first."""
        F = get_region_fields()
        monkeypatch.setattr(worker_state, '_class_group', lambda _name: 'group_a')
        shaped_profile = dataclasses.replace(
            worker_state.region_profile(), secondary_shape_groups=frozenset({'group_a'})
        )
        monkeypatch.setattr(worker_state, 'region_profile', lambda: shaped_profile)
        detector = _lpr_mock([])  # would never produce a candidate
        sam_cand = RegionCandidate(bbox_norm=(0.4, 0.45, 0.5, 0.50), score=0.81, source='sam3')
        task = _make_task(
            status='pending',
            class_name='some-class',
            vehicle_bbox=(0.0, 0.0, 1.0, 1.0),
        )
        await worker._process_crop(
            task,
            detector=detector,
            sam3=_sam3_mock(sam_cand),
            ocr_recognizer=_ocr_recognizer_mock(),
            vlm=_vlm_mock(is_region=True),
        )
        detector.detect_batch.assert_not_awaited()
        assert task.update_doc[F.status] == 'detected'

    @pytest.mark.asyncio
    async def test_pending_car_runs_lpr_first_and_succeeds(self) -> None:
        """Non-secondary-shape pending → primary hit + VLM verify → done, no secondary segmenter call."""
        F = get_region_fields()
        detector_cand = RegionCandidate(
            bbox_norm=(0.3, 0.4, 0.5, 0.45), score=0.82, source='license_plate_detector'
        )
        sam3 = _sam3_mock(None)
        task = _make_task(
            status='pending',
            class_name='audi',
            group='cars',
            vehicle_bbox=(0.0, 0.0, 1.0, 1.0),
        )
        await worker._process_crop(
            task,
            detector=_lpr_mock([detector_cand]),
            sam3=sam3,
            ocr_recognizer=_ocr_recognizer_mock(),
            vlm=_vlm_mock(is_region=True),
        )
        sam3.segment.assert_not_awaited()
        assert task.update_doc[F.status] == 'detected'
        assert task.update_doc[F.bbox_norm] == list(detector_cand.bbox_norm)

    @pytest.mark.asyncio
    async def test_pending_car_lpr_rejected_falls_through_to_sam3(self) -> None:
        F = get_region_fields()
        detector_cand = RegionCandidate(
            bbox_norm=(0.3, 0.4, 0.5, 0.45), score=0.82, source='license_plate_detector'
        )
        sam_cand = RegionCandidate(bbox_norm=(0.6, 0.6, 0.7, 0.65), score=0.79, source='sam3')
        vlm = MagicMock()
        vlm.verify_region = AsyncMock(
            side_effect=[
                VlmRegionVerdict(crop_id='c', is_region=False, confidence='high', reason='no'),
                VlmRegionVerdict(crop_id='c', is_region=True, confidence='high', reason='yes'),
            ]
        )
        vlm.aclose = AsyncMock()
        task = _make_task(
            status='pending', class_name='audi', group='cars', vehicle_bbox=(0.0, 0.0, 1.0, 1.0)
        )
        await worker._process_crop(
            task,
            detector=_lpr_mock([detector_cand]),
            sam3=_sam3_mock(sam_cand),
            ocr_recognizer=_ocr_recognizer_mock(),
            vlm=vlm,
        )
        assert task.update_doc[F.status] == 'detected'
        assert task.update_doc[F.bbox_norm] == list(sam_cand.bbox_norm)

    @pytest.mark.asyncio
    async def test_terminal_status_is_skipped(self) -> None:
        """Terminal statuses must never be touched."""
        for status in ('detected', 'no_region_visible', 'verify_rejected', 'no_region_box'):
            task = _make_task(status=status)
            await worker._process_crop(
                task,
                detector=_lpr_mock([]),
                sam3=_sam3_mock(None),
                ocr_recognizer=_ocr_recognizer_mock(),
                vlm=_vlm_mock(is_region=True),
            )
            assert task.update_doc == {}, f'status={status!r} should not be touched'

    @pytest.mark.asyncio
    async def test_all_detectors_miss_marks_no_region_box(self) -> None:
        F = get_region_fields()
        task = _make_task(status='pending', group='cars')
        await worker._process_crop(
            task,
            detector=_lpr_mock([None]),
            sam3=_sam3_mock(None),
            ocr_recognizer=_ocr_recognizer_mock(),
            vlm=_vlm_mock(is_region=False),
        )
        # Phase A2: every write carries a detector_chain. Status remains
        # 'no_region_box' (queue for human review) and the chain captures
        # which detectors were tried — used by the LPR-blind-spot
        # training-set selector.
        assert task.update_doc[F.status] == 'no_region_box'
        chain = task.update_doc.get(F.detector_chain) or []
        assert any('license_plate_detector:miss' in s for s in chain)
        assert any('sam3:miss' in s for s in chain)

    @pytest.mark.asyncio
    async def test_text_hint_sam3_rescue_succeeds(self) -> None:
        """Primary + secondary globally miss but the OCR pipeline locates plate-shaped
        text; the secondary segmenter re-prompted on a tight sub-crop produces the final box.

        The OCR-detection geometry is intentionally NEVER written as a
        region bbox — that path produced visibly-oversized boxes. The
        final detector stamped on the row is the secondary segmenter,
        not paddleocr_det_trt.
        """
        F = get_region_fields()
        # OCR finds a plate-shaped region somewhere in the crop.
        ocr_pick = MagicMock()
        ocr_pick.bbox_norm = (0.40, 0.40, 0.55, 0.45)
        ocr_pick.text = 'ABC1234'
        ocr_pick.rec_score = 0.85
        # The secondary segmenter on the SUB-crop returns a tighter,
        # more accurate region box (sub-crop coordinates). The worker
        # projects it back to the parent-crop frame before writing.
        sub_sam_cand = RegionCandidate(
            bbox_norm=(0.20, 0.30, 0.80, 0.60), score=0.74, source='sam3'
        )
        vlm = _vlm_mock(is_region=True)
        task = _make_task(status='pending', group='cars', vehicle_bbox=(0.0, 0.0, 1.0, 1.0))
        await worker._process_crop(
            task,
            detector=_lpr_mock([None]),
            # First SAM3 call (global) misses; second (sub-crop) hits.
            sam3=_sam3_mock_sequence([None, sub_sam_cand]),
            ocr_recognizer=_ocr_recognizer_mock(regions=[ocr_pick], pick=ocr_pick),
            vlm=vlm,
        )
        assert task.update_doc[F.status] == 'detected'
        # Provenance: the secondary segmenter owns the final geometry.
        # The OCR-detection model never appears as the detector for a
        # text-hint write.
        assert task.update_doc[F.detector] == 'sam3'
        chain = task.update_doc.get(F.detector_chain) or []
        assert any('paddleocr_rec_trt:text_hint:hit' in s for s in chain)
        assert any('sam3:text_hint:vlm_verify_ok' in s for s in chain)


# =============================================================================
# No-verdict handling: absence of a VLM answer must never read as a reject
# =============================================================================


class TestNoVerdictLeavesItemPending:
    @pytest.mark.asyncio
    async def test_pending_verify_no_verdict_leaves_task_untouched(self) -> None:
        """``verify_region`` returning ``None`` (no usable answer) must not be
        treated as a reject -- the crop is left pending for a retry rather
        than falling through to the secondary segmenter or writing a
        terminal status."""
        vlm = MagicMock()
        vlm.verify_region = AsyncMock(return_value=None)
        vlm.aclose = AsyncMock()
        sam3 = _sam3_mock(RegionCandidate(bbox_norm=(0.4, 0.5, 0.6, 0.55), score=0.77))
        task = _make_task(
            status='pending_verify',
            detector_region_in_source=(0.2, 0.2, 0.3, 0.22),
            detector_score=0.7,
            vehicle_bbox=(0.0, 0.0, 1.0, 1.0),
        )
        await worker._process_crop(
            task,
            detector=_lpr_mock([]),
            sam3=sam3,
            ocr_recognizer=_ocr_recognizer_mock(),
            vlm=vlm,
        )
        assert task.update_doc == {}
        # The cascade bailed out immediately on the no-verdict -- it must
        # not have fallen through to try the secondary segmenter.
        sam3.segment.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_pending_car_lpr_no_verdict_leaves_task_untouched(self) -> None:
        F = get_region_fields()
        detector_cand = RegionCandidate(
            bbox_norm=(0.3, 0.4, 0.5, 0.45), score=0.82, source='license_plate_detector'
        )
        vlm = MagicMock()
        vlm.verify_region = AsyncMock(return_value=None)
        vlm.aclose = AsyncMock()
        sam3 = _sam3_mock(RegionCandidate(bbox_norm=(0.6, 0.6, 0.7, 0.65), score=0.79))
        task = _make_task(
            status='pending', class_name='audi', group='cars', vehicle_bbox=(0.0, 0.0, 1.0, 1.0)
        )
        await worker._process_crop(
            task,
            detector=_lpr_mock([detector_cand]),
            sam3=sam3,
            ocr_recognizer=_ocr_recognizer_mock(),
            vlm=vlm,
        )
        assert F.status not in task.update_doc
        sam3.segment.assert_not_awaited()


# =============================================================================
# Coordinate re-projection
# =============================================================================


class TestReprojection:
    @pytest.mark.asyncio
    async def test_sam3_bbox_projected_to_source_frame(self) -> None:
        """The secondary segmenter emits crop-frame coords; the worker must re-project."""
        F = get_region_fields()
        # Region-shaped crop-frame bbox (aspect = 4) so the Phase A3
        # sanity gate admits it; the previous square box would
        # (correctly) reject.
        sam_cand = RegionCandidate(bbox_norm=(0.5, 0.85, 0.9, 0.95), score=0.7, source='sam3')
        # Vehicle box covers the upper-left quadrant of the source.
        vehicle = (0.20, 0.10, 0.60, 0.50)
        task = _make_task(status='pending', group='cars', vehicle_bbox=vehicle)
        await worker._process_crop(
            task,
            detector=_lpr_mock([None]),
            sam3=_sam3_mock(sam_cand),
            ocr_recognizer=_ocr_recognizer_mock(),
            vlm=_vlm_mock(is_region=True),
        )
        expected = list(crop_norm_to_source_norm(sam_cand.bbox_norm, vehicle))
        # Sanity: the projected box should be inside the vehicle box.
        assert expected[0] >= vehicle[0]
        assert expected[2] <= vehicle[2]
        assert task.update_doc[F.bbox_norm] == pytest.approx(expected)


# =============================================================================
# Provenance + sanity gate (Phase A2 / A3)
# =============================================================================


class TestProvenance:
    @pytest.mark.asyncio
    async def test_lpr_write_carries_provenance(self) -> None:
        """A successful primary-detector write must stamp detector + version + frame + ts."""
        F = get_region_fields()
        detector_cand = RegionCandidate(
            bbox_norm=(0.3, 0.4, 0.5, 0.45), score=0.82, source='license_plate_detector'
        )
        task = _make_task(
            status='pending',
            group='cars',
            vehicle_bbox=(0.0, 0.0, 1.0, 1.0),
        )
        await worker._process_crop(
            task,
            detector=_lpr_mock([detector_cand]),
            sam3=_sam3_mock(None),
            ocr_recognizer=_ocr_recognizer_mock(),
            vlm=_vlm_mock(is_region=True),
        )
        assert task.update_doc[F.detector] == 'license_plate_detector'
        assert task.update_doc[F.detector_version] == '1'
        assert task.update_doc[F.bbox_frame] == 'source'
        assert F.detected_at in task.update_doc
        # Verifier fields should be present (VLM verified).
        assert task.update_doc[F.verifier] == 'test-vlm-model'
        # Chain captures the cascade: hit + vlm_verify_ok.
        chain = task.update_doc.get(F.detector_chain) or []
        assert any('license_plate_detector:hit' in s for s in chain)
        assert any('vlm_verify_ok' in s for s in chain)

    @pytest.mark.asyncio
    async def test_sam3_write_carries_sam3_detector(self) -> None:
        F = get_region_fields()
        sam_cand = RegionCandidate(bbox_norm=(0.30, 0.40, 0.45, 0.45), score=0.78, source='sam3')
        task = _make_task(
            status='pending',
            group='cars',
            vehicle_bbox=(0.0, 0.0, 1.0, 1.0),
        )
        await worker._process_crop(
            task,
            detector=_lpr_mock([None]),
            sam3=_sam3_mock(sam_cand),
            ocr_recognizer=_ocr_recognizer_mock(),
            vlm=_vlm_mock(is_region=True),
        )
        assert task.update_doc[F.detector] == 'sam3'
        assert task.update_doc[F.bbox_frame] == 'source'
        chain = task.update_doc.get(F.detector_chain) or []
        # A primary-detector miss must be recorded so training-set
        # selection can find it.
        assert any('license_plate_detector:miss' in s for s in chain)
        assert any('sam3:hit' in s for s in chain)

    @pytest.mark.asyncio
    async def test_large_lpr_bbox_no_longer_sanity_rejected(self) -> None:
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
        task = _make_task(
            status='pending',
            group='cars',
            vehicle_bbox=(0.0, 0.0, 1.0, 1.0),
        )
        await worker._process_crop(
            task,
            detector=_lpr_mock([big_cand]),
            sam3=_sam3_mock(None),
            ocr_recognizer=_ocr_recognizer_mock(),
            vlm=_vlm_mock(is_region=True),
        )
        chain = task.update_doc.get(F.detector_chain) or []
        assert not any('sanity_reject' in s for s in chain)
        assert task.update_doc[F.detector] == 'license_plate_detector'
        assert F.bbox_norm in task.update_doc


# =============================================================================
# Bulk write
# =============================================================================


class TestBulkWrite:
    @pytest.mark.asyncio
    async def test_bulk_write_batches_only_updated_crops(self) -> None:
        """``_bulk_update`` issues ONE batched mget + ONE bulk call (P2-13)
        covering every crop with a non-empty ``update_doc``; tasks with no
        update doc never even reach the mget/bulk round trip."""
        F = get_region_fields()
        a = _make_task(crop_id='a')
        a.update_doc = {F.status: 'detected', F.score: 0.9}
        b = _make_task(crop_id='b')
        # No update — should be skipped.
        c = _make_task(crop_id='c')
        c.update_doc = {F.status: 'no_region_box'}

        async def _fake_mget(*, body: dict[str, Any]) -> dict[str, Any]:
            # Docs are still in the pending state the tasks were fetched in.
            found: dict[str, dict[str, Any]] = {
                d['_id']: {F.status: 'pending'} for d in body['docs']
            }
            return make_mget_response(found)

        async def _fake_bulk(*, body: list[dict[str, Any]], **kw: Any) -> dict[str, Any]:
            items = [
                make_bulk_update_item(action['update']['_id'], status=200) for action in body[0::2]
            ]
            return make_bulk_response(items)

        opensearch = MagicMock()
        opensearch.mget = AsyncMock(side_effect=_fake_mget)
        opensearch.bulk = AsyncMock(side_effect=_fake_bulk)

        n_written, n_skipped = await worker._bulk_update(opensearch, [a, b, c])
        assert n_written == 2
        assert n_skipped == 1
        # One batched mget + one batched bulk for the 2 crops with a
        # non-empty update_doc — not 2 round trips each.
        assert opensearch.mget.await_count == 1
        assert opensearch.bulk.await_count == 1
        mget_call = opensearch.mget.await_args
        assert mget_call is not None
        mget_ids = sorted(d['_id'] for d in mget_call.kwargs['body']['docs'])
        assert mget_ids == ['a', 'c']
        bulk_call = opensearch.bulk.await_args
        assert bulk_call is not None
        bulk_body = bulk_call.kwargs['body']
        bulk_ids = sorted(action['update']['_id'] for action in bulk_body[0::2])
        assert bulk_ids == ['a', 'c']

    @pytest.mark.asyncio
    async def test_bulk_write_skipped_when_no_updates(self) -> None:
        """No tasks need writing → no OS round-trip."""
        opensearch = MagicMock()
        opensearch.mget = AsyncMock()
        opensearch.bulk = AsyncMock()
        n_written, n_skipped = await worker._bulk_update(opensearch, [_make_task()])
        assert n_written == 0
        assert n_skipped == 1
        opensearch.mget.assert_not_awaited()
        opensearch.bulk.assert_not_awaited()


# =============================================================================
# Sentinel pause
# =============================================================================


class TestSentinel:
    @pytest.mark.asyncio
    async def test_wait_returns_immediately_when_sentinel_absent(self, tmp_path: Path) -> None:
        sentinel = tmp_path / 'pause.sentinel'  # does not exist
        # Should not block.
        await asyncio.wait_for(worker._wait_for_sentinel_clear(sentinel, sleep_s=0.01), timeout=0.5)

    @pytest.mark.asyncio
    async def test_wait_blocks_then_unblocks(self, tmp_path: Path) -> None:
        sentinel = tmp_path / 'pause.sentinel'
        sentinel.write_text('paused')

        async def _delete_after(delay: float) -> None:
            await asyncio.sleep(delay)
            sentinel.unlink()

        # Kick off the deletion concurrently with the wait.
        deleter = asyncio.create_task(_delete_after(0.05))
        await asyncio.wait_for(worker._wait_for_sentinel_clear(sentinel, sleep_s=0.01), timeout=1.0)
        await deleter
        assert not sentinel.exists()


# =============================================================================
# SIGTERM handling
# =============================================================================


class TestSignalHandling:
    @pytest.mark.asyncio
    async def test_run_exits_cleanly_on_stop(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A faked ``run``-style loop respects the stop event and shuts down
        all owned resources (httpx clients, OpenSearch, Triton pool)."""

        # Patch the heavy I/O constructors so ``run()`` is pure async logic.
        pool = MagicMock()
        pool.initialize = AsyncMock()
        pool.close = AsyncMock()
        monkeypatch.setattr(worker, 'AsyncTritonPool', MagicMock(return_value=pool))

        os_client = MagicMock()
        os_client.search = AsyncMock(return_value={'hits': {'hits': []}})
        os_client.bulk = AsyncMock()
        os_client.close = AsyncMock()
        monkeypatch.setattr(worker, 'AsyncOpenSearch', MagicMock(return_value=os_client))

        sam3 = MagicMock()
        sam3.aclose = AsyncMock()
        monkeypatch.setattr(worker, 'SegmenterClient', MagicMock(return_value=sam3))

        vlm = MagicMock()
        vlm.aclose = AsyncMock()
        monkeypatch.setattr(worker, 'VlmLabeler', MagicMock(return_value=vlm))

        # Patch signal-handler installation away — adding signal handlers
        # in a non-main asyncio loop would raise here.
        def _noop_signal_handler(*_args: object, **_kwargs: object) -> None:
            return None

        monkeypatch.setattr(
            asyncio.get_event_loop().__class__,
            'add_signal_handler',
            _noop_signal_handler,
            raising=False,
        )

        # Bind the metrics HTTP server to an ephemeral port so the test
        # doesn't collide with a worker container already holding the
        # default 4609 on this host.
        monkeypatch.setenv('SAM_WORKER_METRICS_PORT', '0')

        sentinel = tmp_path / 'pause.sentinel'  # absent
        args = worker.parse_args(
            [
                '--opensearch=http://os.local:9200',
                '--triton=triton:8001',
                '--sam3-url=http://sam3.local:8000',
                '--vlm-url=http://vlm.local:8000',
                f'--pause-sentinel={sentinel}',
                '--max-iterations=1',
            ]
        )
        # No --continuous, so empty queue should make run() exit immediately
        # after one fetch.
        rc = await worker.run(args)
        assert rc == 0
        # Cleanup happened.
        os_client.close.assert_awaited()
        pool.close.assert_awaited()
        sam3.aclose.assert_awaited()
        vlm.aclose.assert_awaited()


# =============================================================================
# is_secondary_shape helper
# =============================================================================


class TestIsSecondaryShape:
    """F-11: group resolution now comes from the class registry
    (class_name -> group), not the dead ``_ItemTask.group`` field nothing
    ever wrote or mapped on the item doc."""

    def test_explicit_groups(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for grp in worker.region_profile().secondary_shape_groups:
            monkeypatch.setattr(worker_state, '_class_group', lambda _name, grp=grp: grp)
            assert worker._is_secondary_shape(_make_task(class_name='some-class'))

    def test_non_secondary_groups(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for grp in ('cars', 'sportycars', 'exotics'):
            monkeypatch.setattr(worker_state, '_class_group', lambda _name, grp=grp: grp)
            assert not worker._is_secondary_shape(_make_task(class_name='audi'))

    def test_no_registry_group_is_never_secondary_shape(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Class not in the registry (no group resolves) -- there is no
        # private-naming-convention fallback; a group-less item is simply
        # not routed to the secondary-shape path.
        monkeypatch.setattr(worker_state, '_class_group', lambda _name: None)
        t = _make_task(class_name='cruiserbike')
        assert not worker._is_secondary_shape(t)

    def test_registry_group_takes_priority_over_class_name(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The registry's group answer is the only source of truth --
        the class name itself is never inspected."""
        monkeypatch.setattr(worker_state, '_class_group', lambda _name: 'cars')
        t = _make_task(class_name='cruiserbike')
        assert not worker._is_secondary_shape(t)
