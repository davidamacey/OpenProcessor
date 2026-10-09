"""Unit tests for ``scripts.curation.region_worker_main``.

Curation worker tests. Every external system (RegionDetector, SAM 3
HTTP, the VLM, OpenSearch, Triton, disk) is mocked — the tests run in
milliseconds without any GPU, network, or filesystem dependency.

Coverage:

* Routing rules — one test per row of the cascade's decision table,
  driven end to end through the real streaming pipeline
  (:func:`scripts.curation.worker.runner.run`) rather than the deleted
  per-crop ``cascade._process_crop`` unit-call harness. W8: every
  candidate is a ``region_boxes`` list element (N=1 is a list of one),
  and the pipeline's single combined VLM call is the only verify path
  — there is no separate ``verify_region`` two-call cascade to unit-test
  anymore, and a combined-verify reject is terminal (``verify_rejected``)
  for this pass, not an intra-pass fallback to the secondary segmenter.
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
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from _fake_project_registry import install_static_project_registry

import scripts.curation.region_worker_main as worker
import scripts.curation.worker.state as worker_state
from curation.occ_fakes import make_bulk_response, make_bulk_update_item, make_mget_response
from scripts.curation.worker import runner as runner_mod, stage_a as stage_a_mod
from scripts.curation.worker.client import SegmenterRequestFailed
from src.config import get_region_fields
from src.config.project_context import current_project
from src.services.curation.class_write_guard import class_state_token
from src.services.curation.region_boxes import RegionBox, boxes_write_fields
from src.services.detection.cascade_detect import RegionCandidate, crop_norm_to_source_norm
from src.services.labeling.region_overlay import VlmBoxVerdict
from src.services.labeling.vlm_models import VlmCombinedReply

from .test_region_cascade_integrity import (
    _accept,
    _capture_signal_handler,
    _drive_worker,
    _FakeOpenSearch,
    _item,
    _jpeg,
    _profile,
)


# The cascade needs an active region profile; the default is none.
pytestmark = pytest.mark.usefixtures('reference_region_profile')


if TYPE_CHECKING:
    from pathlib import Path


# =============================================================================
# Fixtures and helpers
# =============================================================================


def _make_jpeg(width: int = 320, height: int = 240) -> bytes:
    return _jpeg()


def _make_task(
    *,
    crop_id: str = 'crop-1',
    status: str | None = 'pending',
    group: str = 'cars',
    class_name: str = 'audi',
    item_bbox: tuple[float, float, float, float] = (0.1, 0.1, 0.5, 0.5),
    crop_jpeg: bytes | None = None,
) -> worker._ItemTask:
    """Build a fully populated ``_ItemTask`` for the (non-cascade)
    bulk-write / secondary-shape tests below."""
    return worker._ItemTask(
        project=current_project().record,
        crop_id=crop_id,
        image_path='/dev/null/never-read',
        item_bbox_norm=item_bbox,
        region_status=status,
        class_name=class_name,
        group=group,
        crop_jpeg=crop_jpeg if crop_jpeg is not None else _make_jpeg(),
    )


def _proposed_box_fields(bbox: tuple[float, float, float, float], score: float) -> dict[str, Any]:
    """The stored fields of an item carrying one ``proposed`` box (what a
    ``pending_verification`` item holds), built the way every writer
    builds them."""
    box = RegionBox(
        box_id='b1',
        bbox_norm=bbox,
        state='proposed',
        score=score,
        detector='det_model',
        source='detector',
    )
    return boxes_write_fields([box], current_src={})


def _item_with(**over: Any) -> dict[str, Any]:
    """``_item()`` with top-level overrides (``bbox_norm``, region-field
    overrides via ``get_region_fields()``, etc.)."""
    doc = _item(status=over.pop('status', 'pending_detection'))
    doc.update(over)
    return doc


def _reject(reason_text: str | None = None) -> VlmCombinedReply:
    return VlmCombinedReply(
        img_id='c1',
        region_visible=True,
        region_boxes=[VlmBoxVerdict(box=1, bbox_correct=False, confidence='high')],
    )


async def _drive_text_hint_rescue(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    fake_os: _FakeOpenSearch,
    ocr_pick: Any,
    sub_sam_cand: RegionCandidate | Exception,
    reply: VlmCombinedReply,
) -> dict[str, Any]:
    """Drive the real pipeline through the text-hint sub-crop rescue: the
    primary detector and the segmenter's global attempt both miss, the
    OCR pipeline locates plate-shaped text, and the segmenter re-prompted
    on a tight sub-crop (``SegmenterClient.segment``, deliberately kept
    single-candidate -- see ``client.py``) produces the final box.

    Not expressible through ``_drive_worker`` (fixed OCR/segmenter
    mocks): this needs the segmenter's two call shapes (``segment_multi``
    for the global miss, ``segment`` for the sub-crop) and a real OCR
    pick, so it stands up the pipeline directly, the same way
    ``_drive_worker`` does.
    """
    handlers = _capture_signal_handler(monkeypatch)
    monkeypatch.setenv('OP_REGION_WORKER_METRICS_PORT', '0')

    pool = MagicMock(initialize=AsyncMock(), close=AsyncMock())
    monkeypatch.setattr(worker, 'AsyncTritonPool', MagicMock(return_value=pool))
    monkeypatch.setattr(worker, 'make_script_opensearch', MagicMock(return_value=fake_os))
    install_static_project_registry(monkeypatch)

    primary_det = MagicMock()
    primary_det.confidence_floor = 0.0
    primary_det.detect_batch = AsyncMock(return_value=[None])
    primary_det.detect_batch_multi = AsyncMock(return_value=[[]])
    monkeypatch.setattr(runner_mod, 'RegionDetector', MagicMock(return_value=primary_det))

    ocr = MagicMock()
    ocr.detect_regions = AsyncMock(return_value=[ocr_pick])
    ocr.pick_best_text_region = MagicMock(return_value=ocr_pick)
    monkeypatch.setattr(runner_mod, 'PaddleOcrTextRecognizer', MagicMock(return_value=ocr))
    monkeypatch.setattr(stage_a_mod, '_crop_jpeg_for_task', lambda *_a: _jpeg())

    seg = MagicMock(aclose=AsyncMock())
    seg.segment_multi = AsyncMock(return_value=[])  # global attempt always misses
    # sub-crop re-prompt hits, or raises when given an exception
    seg.segment = AsyncMock(
        side_effect=sub_sam_cand if isinstance(sub_sam_cand, Exception) else None,
        return_value=None if isinstance(sub_sam_cand, Exception) else sub_sam_cand,
    )
    monkeypatch.setattr(worker, 'SegmenterClient', MagicMock(return_value=seg))

    vlm = MagicMock(aclose=AsyncMock())
    vlm.class_names = []
    vlm.label_combined_batch = AsyncMock(
        side_effect=lambda crops, **_kw: {c.crop_id: reply for c in crops}
    )
    vlm.region_visible_batch = AsyncMock(
        side_effect=lambda crops, **_kw: dict.fromkeys((c.crop_id for c in crops), True)
    )
    monkeypatch.setattr(worker, 'build_vlm_labeler', MagicMock(return_value=vlm))
    monkeypatch.setattr(
        'src.clients.curation_opensearch.ClassRegistry',
        MagicMock(side_effect=RuntimeError('no registry in test')),
    )
    monkeypatch.setattr('scripts.curation.worker.state._class_group', lambda _name: None)

    monkeypatch.setenv('OP_VLM_URL', 'http://vlm.invalid:8000')
    args = worker.parse_args(
        [
            '--opensearch=http://os.invalid:9200',
            '--triton=triton.invalid:8001',
            '--segmenter-url=http://seg.invalid:8000',
            f'--pause-sentinel={tmp_path / "absent.sentinel"}',
            '--continuous',
            '--poll-interval=0.01',
            '--batch-size=4',
            '--concurrency=2',
        ]
    )

    async def _stopper() -> None:
        for _ in range(500):
            await asyncio.sleep(0.01)
            if fake_os.writes:
                break
        await asyncio.sleep(0.3)
        handlers[0]()

    stopper = asyncio.create_task(_stopper())
    rc = await asyncio.wait_for(runner_mod.run(args), timeout=20)
    await stopper
    assert rc == 0
    return {'primary': primary_det, 'seg': seg, 'ocr': ocr, 'vlm': vlm}


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
        """W8: a combined-verify reject on the existing candidate is a
        terminal ``verify_rejected`` write for this pass -- there is no
        intra-pass fallback to the secondary segmenter (pre-W8's
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
        """Same W8 architecture note as the pending_verify case: a primary-
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
        # touched by this pass -- see the W8 handback report).
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
        a.update_doc = {F.status: 'detected', F.max_score: 0.9}
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

    @pytest.mark.asyncio
    async def test_vlm_unmatched_write_clears_the_prior_class(self) -> None:
        """IT-2: a vlm_unmatched write must not keep the class_id/class_name
        the VLM's answer just contradicted -- and must reset a class-range
        cluster_id so the residual pass re-clusters the item."""
        F = get_region_fields()
        current_doc = {
            F.status: 'pending',
            'class_id': 5,
            'class_name': 'foo',
            'class_source': 'coco_yolo11_model',
            'cluster_id': 5,
        }
        a = _make_task(crop_id='a', status='pending')
        a.class_token = class_state_token(current_doc)
        a.update_doc = {'class_source': 'vlm_unmatched', 'vlm_raw_class': 'zzz'}

        async def _fake_mget(*, body: dict[str, Any]) -> dict[str, Any]:
            return make_mget_response({d['_id']: current_doc for d in body['docs']})

        captured: dict[str, dict[str, Any]] = {}

        async def _fake_bulk(*, body: list[dict[str, Any]], **kw: Any) -> dict[str, Any]:
            for action, doc in zip(body[0::2], body[1::2], strict=True):
                captured[action['update']['_id']] = doc['doc']
            items = [
                make_bulk_update_item(action['update']['_id'], status=200) for action in body[0::2]
            ]
            return make_bulk_response(items)

        opensearch = MagicMock()
        opensearch.mget = AsyncMock(side_effect=_fake_mget)
        opensearch.bulk = AsyncMock(side_effect=_fake_bulk)

        n_written, n_skipped = await worker._bulk_update(opensearch, [a])
        assert n_written == 1
        assert n_skipped == 0
        doc = captured['a']
        assert doc['class_id'] is None
        assert doc['class_name'] is None
        assert doc['class_detector'] is None
        assert doc['cluster_id'] == -1
        assert doc['cluster_subid'] is None

    @pytest.mark.asyncio
    async def test_vlm_unmatched_write_leaves_a_candidate_cluster_alone(self) -> None:
        """A candidate (residual) cluster isn't a class-cluster fact --
        clearing the class must not touch it."""
        F = get_region_fields()
        current_doc = {
            F.status: 'pending',
            'class_id': 5,
            'class_name': 'foo',
            'class_source': 'coco_yolo11_model',
            'cluster_id': 10042,
        }
        a = _make_task(crop_id='a', status='pending')
        a.class_token = class_state_token(current_doc)
        a.update_doc = {'class_source': 'vlm_unmatched', 'vlm_raw_class': 'zzz'}

        async def _fake_mget(*, body: dict[str, Any]) -> dict[str, Any]:
            return make_mget_response({d['_id']: current_doc for d in body['docs']})

        captured: dict[str, dict[str, Any]] = {}

        async def _fake_bulk(*, body: list[dict[str, Any]], **kw: Any) -> dict[str, Any]:
            for action, doc in zip(body[0::2], body[1::2], strict=True):
                captured[action['update']['_id']] = doc['doc']
            items = [
                make_bulk_update_item(action['update']['_id'], status=200) for action in body[0::2]
            ]
            return make_bulk_response(items)

        opensearch = MagicMock()
        opensearch.mget = AsyncMock(side_effect=_fake_mget)
        opensearch.bulk = AsyncMock(side_effect=_fake_bulk)

        await worker._bulk_update(opensearch, [a])
        doc = captured['a']
        assert doc['class_id'] is None
        assert 'cluster_id' not in doc
        assert 'cluster_subid' not in doc

    @pytest.mark.asyncio
    async def test_vlm_unmatched_write_never_clears_a_validated_item(self) -> None:
        F = get_region_fields()
        current_doc = {
            F.status: 'pending',
            'class_id': 5,
            'class_name': 'foo',
            'class_source': 'human',
            'class_validated': True,
            'cluster_id': 5,
        }
        a = _make_task(crop_id='a', status='pending')
        a.class_token = class_state_token(current_doc)
        a.update_doc = {'class_source': 'vlm_unmatched', 'vlm_raw_class': 'zzz'}

        async def _fake_mget(*, body: dict[str, Any]) -> dict[str, Any]:
            return make_mget_response({d['_id']: current_doc for d in body['docs']})

        captured: dict[str, dict[str, Any]] = {}

        async def _fake_bulk(*, body: list[dict[str, Any]], **kw: Any) -> dict[str, Any]:
            for action, doc in zip(body[0::2], body[1::2], strict=True):
                captured[action['update']['_id']] = doc['doc']
            items = [
                make_bulk_update_item(action['update']['_id'], status=200) for action in body[0::2]
            ]
            return make_bulk_response(items)

        opensearch = MagicMock()
        opensearch.mget = AsyncMock(side_effect=_fake_mget)
        opensearch.bulk = AsyncMock(side_effect=_fake_bulk)

        n_written, n_skipped = await worker._bulk_update(opensearch, [a])
        # class_write_allowed() strips every CLASS_WRITE_FIELDS key before
        # the vlm_unmatched clear even runs, leaving an empty merge result
        # -- a documented noop (neither updated nor skipped), and no bulk
        # round trip at all -- for a validated item's class fields.
        assert n_written == 0
        assert n_skipped == 0
        opensearch.bulk.assert_not_awaited()
        assert 'a' not in captured


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

        # B1: the producer loop now iterates the real ProjectRegistry's
        # active_projects() every cycle, so the registry needs a real
        # 'default' doc to see -- previously the single global runtime
        # was built unconditionally from the env profile, regardless of
        # what the registry (a separate, only-used-for-fetch concern)
        # contained.
        from src.config.curation import base_curation_config
        from src.config.projects import resources_for_new
        from src.services.projects.registry import (
            REVISION_DOC_ID,
            ProjectRecord,
            projects_index,
            record_to_doc,
        )

        default_record = ProjectRecord(
            slug='default',
            display_name='Default',
            description='',
            status='active',
            revision=1,
            created_at='',
            updated_at='',
            origin=None,
            resources=resources_for_new('default', base_curation_config()),
        )

        async def _search(*, index: str, body: dict[str, Any]) -> dict[str, Any]:
            if index == projects_index():
                return {'hits': {'hits': [{'_source': record_to_doc(default_record)}]}}
            return {'hits': {'hits': []}}

        async def _get(*, index: str, id: str) -> dict[str, Any]:  # noqa: A002
            if index == projects_index() and id == REVISION_DOC_ID:
                return {'found': True, '_source': {'revision': 1}}
            return {'found': False}

        os_client = MagicMock()
        os_client.search = AsyncMock(side_effect=_search)
        os_client.get = AsyncMock(side_effect=_get)
        os_client.bulk = AsyncMock()
        os_client.close = AsyncMock()
        monkeypatch.setattr(worker, 'make_script_opensearch', MagicMock(return_value=os_client))

        segmenter = MagicMock()
        segmenter.aclose = AsyncMock()
        monkeypatch.setattr(worker, 'SegmenterClient', MagicMock(return_value=segmenter))

        vlm = MagicMock()
        vlm.aclose = AsyncMock()
        monkeypatch.setattr(worker, 'build_vlm_labeler', MagicMock(return_value=vlm))

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
        monkeypatch.setenv('OP_REGION_WORKER_METRICS_PORT', '0')

        sentinel = tmp_path / 'pause.sentinel'  # absent
        monkeypatch.setenv('OP_VLM_URL', 'http://vlm.local:8000')
        args = worker.parse_args(
            [
                '--opensearch=http://os.local:9200',
                '--triton=triton:8001',
                '--segmenter-url=http://segmenter.local:8000',
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
        segmenter.aclose.assert_awaited()
        vlm.aclose.assert_awaited()


# =============================================================================
# is_secondary_shape helper
# =============================================================================


class TestIsSecondaryShape:
    """Group resolution now comes from the class registry
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
        t = _make_task(class_name='class_b')
        assert not worker._is_secondary_shape(t)

    def test_registry_group_takes_priority_over_class_name(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The registry's group answer is the only source of truth --
        the class name itself is never inspected."""
        monkeypatch.setattr(worker_state, '_class_group', lambda _name: 'cars')
        t = _make_task(class_name='class_b')
        assert not worker._is_secondary_shape(t)
