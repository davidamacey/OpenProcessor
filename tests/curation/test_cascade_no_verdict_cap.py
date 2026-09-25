"""The per-crop cascade (``_process_crop``) bounds no-verdict retries too.

``_verify_with_vlm`` returns ``None`` when the region verifier gives no
usable answer and the cascade leaves the item pending (no write). At
temperature 0 that answer is often deterministic, so the same bound the
streaming worker uses applies here: after
``OP_REGION_WORKER_MAX_NO_VERDICT_ATTEMPTS`` no-verdict passes the item is
parked as a reviewable ``verify_rejected`` / ``verifier_no_verdict``. A VLM
transport failure is no reply at all -- retried, never counted.
"""

from __future__ import annotations

import io
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from PIL import Image

import scripts.curation.region_worker_main as worker
from scripts.curation.worker import combined as combined_mod, no_verdict
from src.config import get_region_fields
from src.config.region_rejection import REJECT_REASON_NO_VERDICT
from src.config.region_source import CANDIDATE_DETECTOR, CANDIDATE_DETECTOR_EXISTING
from src.services.detection.cascade_detect import RegionCandidate
from src.services.labeling.vlm_labeler import (
    RegionCrop,
    VlmCombinedReply,
    VlmLabeler,
    VlmRegionVerdict,
    VlmTransportError,
)


pytestmark = pytest.mark.usefixtures('reference_region_profile')

ENV = 'OP_REGION_WORKER_MAX_NO_VERDICT_ATTEMPTS'
EXISTING_BOX = (0.2, 0.2, 0.3, 0.22)


@pytest.fixture(autouse=True)
def _default_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    # reference_region_profile resets the process-wide count per test.
    monkeypatch.delenv(ENV, raising=False)


def _jpeg() -> bytes:
    buf = io.BytesIO()
    Image.new('RGB', (320, 240), (50, 80, 120)).save(buf, format='JPEG')
    return buf.getvalue()


def _task(**kw: Any) -> Any:
    base: dict[str, Any] = {
        'crop_id': 'crop-1',
        'image_path': '/dev/null/never-read',
        'vehicle_bbox_norm': (0.0, 0.0, 1.0, 1.0),
        'region_status': 'pending_verify',
        'class_name': 'audi',
        'group': 'cars',
        'detector_region_in_source': EXISTING_BOX,
        'detector_score': 0.7,
        'crop_jpeg': _jpeg(),
    }
    base.update(kw)
    return worker._ItemTask(**base)


def _vlm(*answers: Any) -> MagicMock:
    """``verify_region`` answering ``answers`` in turn (an exception type is
    raised), then the last answer forever."""
    seq = list(answers)

    async def verify_region(_crop: RegionCrop, **_kw: Any) -> Any:
        a = seq.pop(0) if len(seq) > 1 else seq[0]
        if isinstance(a, type) and issubclass(a, Exception):
            raise a('upstream down')
        return a

    g = MagicMock()
    g.verify_region = AsyncMock(side_effect=verify_region)
    g.aclose = AsyncMock()
    return g


def _accept() -> VlmRegionVerdict:
    return VlmRegionVerdict(crop_id='crop-1', is_region=True, confidence='high', reason='ok')


def _mocks() -> dict[str, Any]:
    sam3 = MagicMock()
    sam3.segment = AsyncMock(return_value=None)
    ocr = MagicMock()
    ocr.detect_regions = AsyncMock(return_value=[])
    ocr.pick_best_text_region = MagicMock(return_value=None)
    detector = MagicMock()
    detector.detect_batch = AsyncMock(return_value=[])
    return {'sam3': sam3, 'ocr_recognizer': ocr, 'detector': detector}


async def _pass(vlm: MagicMock, **task_kw: Any) -> Any:
    task = _task(**task_kw)
    await worker._process_crop(task, vlm=vlm, **_mocks())
    return task


class TestCascadeVerifyNoVerdictIsCapped:
    @pytest.mark.asyncio
    async def test_parks_the_existing_box_after_the_cap(self) -> None:
        F = get_region_fields()
        vlm = _vlm(None)
        for _ in range(no_verdict.DEFAULT_MAX_NO_VERDICT_ATTEMPTS - 1):
            assert (await _pass(vlm)).update_doc == {}
        task = await _pass(vlm)
        doc = task.update_doc
        assert doc[F.status] == 'verify_rejected'
        assert doc[F.rejection_reason] == REJECT_REASON_NO_VERDICT
        assert doc[F.bbox_correct] is None
        assert doc[F.bbox_norm] is None
        assert doc[F.candidate_bbox_norm] == list(EXISTING_BOX)
        assert doc[F.candidate_source] == CANDIDATE_DETECTOR_EXISTING
        det = worker.region_profile().detector_model
        assert (
            f'{det}:combined_verify_reject:{REJECT_REASON_NO_VERDICT}' not in doc[F.detector_chain]
        )
        assert f'{det}:vlm_reject:{REJECT_REASON_NO_VERDICT}' in doc[F.detector_chain]

    @pytest.mark.asyncio
    async def test_fresh_detector_box_is_parked_the_same_way(self) -> None:
        F = get_region_fields()
        cand = RegionCandidate(bbox_norm=(0.3, 0.4, 0.5, 0.45), score=0.82, source='det')
        vlm = _vlm(None)
        m = _mocks()
        m['detector'].detect_batch = AsyncMock(return_value=[cand])
        for i in range(no_verdict.DEFAULT_MAX_NO_VERDICT_ATTEMPTS):
            task = _task(region_status='pending', detector_region_in_source=None)
            await worker._process_crop(task, vlm=vlm, **m)
            if i < no_verdict.DEFAULT_MAX_NO_VERDICT_ATTEMPTS - 1:
                assert task.update_doc == {}
        assert task.update_doc[F.rejection_reason] == REJECT_REASON_NO_VERDICT
        assert task.update_doc[F.candidate_source] == CANDIDATE_DETECTOR
        assert task.update_doc[F.candidate_score] == pytest.approx(0.82)

    @pytest.mark.asyncio
    async def test_real_verdict_clears_the_count(self) -> None:
        F = get_region_fields()
        vlm = _vlm(None, None, _accept(), None)
        assert (await _pass(vlm)).update_doc == {}
        assert (await _pass(vlm)).update_doc == {}
        assert (await _pass(vlm)).update_doc[F.status] == 'detected'
        # Requeued later: a fresh count, not the stale two.
        assert (await _pass(vlm)).update_doc == {}
        assert (await _pass(vlm)).update_doc == {}
        assert (await _pass(vlm)).update_doc[F.rejection_reason] == REJECT_REASON_NO_VERDICT

    @pytest.mark.asyncio
    async def test_transport_failure_is_never_capped(self) -> None:
        vlm = _vlm(VlmTransportError)
        for _ in range(no_verdict.DEFAULT_MAX_NO_VERDICT_ATTEMPTS * 3):
            task = await _pass(vlm)
            assert task.update_doc == {}
        assert vlm.verify_region.await_args.kwargs == {'raise_on_transport': True}


class TestCombinedCohortNoVerdictIsCapped:
    @pytest.mark.asyncio
    async def test_null_box_verdict_is_parked_after_the_cap(self) -> None:
        F = get_region_fields()
        reply = VlmCombinedReply(img_id='c1', region_visible=True, region_bbox_correct=None)
        vlm = MagicMock()
        vlm.class_names = []
        vlm.label_combined = AsyncMock(return_value=reply)
        det = worker.region_profile().detector_model
        for i in range(no_verdict.DEFAULT_MAX_NO_VERDICT_ATTEMPTS):
            task = _task(
                crop_id='c1', region_status='pending_detection', detector_region_in_source=None
            )
            resolved = await combined_mod._try_combined_class_region(
                task,
                candidate_in_crop=(0.3, 0.6, 0.6, 0.75),
                candidate_in_source=(0.3, 0.6, 0.6, 0.75),
                candidate_score=0.9,
                detector=det,
                detector_version='1',
                detector_chain_tag=det,
                vlm=vlm,
                candidate_source=CANDIDATE_DETECTOR,
            )
            assert resolved is True
            if i < no_verdict.DEFAULT_MAX_NO_VERDICT_ATTEMPTS - 1:
                assert task.update_doc == {}
        doc = task.update_doc
        assert doc[F.rejection_reason] == REJECT_REASON_NO_VERDICT
        assert doc[F.bbox_correct] is None
        assert doc[F.candidate_source] == CANDIDATE_DETECTOR
        # The reply's class side lands with it.
        assert doc[F.visible] is True
        assert 'vlm_verify_completed_at' in doc


class TestVerifyPlateTransportSignal:
    def _labeler(self) -> VlmLabeler:
        lab = VlmLabeler(base_url='http://vlm.invalid/v1')
        lab._post_chat = AsyncMock(side_effect=httpx.ConnectError('down'))  # type: ignore[method-assign]
        return lab

    @pytest.mark.asyncio
    async def test_opt_in_raises_on_transport_failure(self) -> None:
        with pytest.raises(VlmTransportError):
            await self._labeler().verify_region(
                RegionCrop(crop_id='r1', jpeg_bytes=b'x'), raise_on_transport=True
            )

    @pytest.mark.asyncio
    async def test_default_still_returns_no_verdict(self) -> None:
        assert (
            await self._labeler().verify_region(RegionCrop(crop_id='r1', jpeg_bytes=b'x')) is None
        )

    @pytest.mark.asyncio
    async def test_unparseable_reply_is_no_verdict_even_with_opt_in(self) -> None:
        lab = VlmLabeler(base_url='http://vlm.invalid/v1')
        lab._post_chat = AsyncMock(  # type: ignore[method-assign]
            return_value={'choices': [{'message': {'content': 'not json'}}]}
        )
        crop = RegionCrop(crop_id='r1', jpeg_bytes=b'x')
        assert await lab.verify_region(crop, raise_on_transport=True) is None

    def test_combined_transport_failure_is_a_transport_error(self) -> None:
        from src.services.labeling.vlm_labeler import CombinedTransportError

        assert issubclass(CombinedTransportError, VlmTransportError)
