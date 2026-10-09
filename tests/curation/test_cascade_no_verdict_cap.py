"""The streaming worker's Stage B (combined VLM call) bounds no-verdict
retries.

W8: the pre-W8 per-crop cascade (``_process_crop``) and its
``combined.py`` cohort helper are deleted -- ``runner.py``'s
``stage_b_combined`` is the only production no-verdict-cap consumer now,
via :func:`scripts.curation.worker.verify.verdicts_to_boxes`'s
``force_resolve`` split. A box with no verdict at all (null / absent /
unparseable, on every candidate offered) leaves the item pending for a
retry instead of writing a reject; at
``OP_REGION_WORKER_MAX_NO_VERDICT_ATTEMPTS`` it is parked as a reviewable
``verify_rejected`` box (``rejection_reason=verifier_no_verdict``). A VLM
transport failure is no reply at all -- retried, never counted (see
``TestCascadeVerifyNoVerdictIsCapped`` in ``test_region_cascade_integrity
.py``-style ``_drive_worker`` harness, reused here).
"""

from __future__ import annotations

import io
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock

import httpx
import pytest
from PIL import Image

from src.config import get_region_fields
from src.config.region_rejection import REJECT_REASON_NO_VERDICT
from src.services.detection.cascade_detect.candidate import RegionCandidate
from src.services.labeling.region_overlay import VlmBoxVerdict
from src.services.labeling.vlm_labeler import VlmLabeler
from src.services.labeling.vlm_models import RegionCrop, VlmCombinedReply, VlmTransportError

from .test_region_cascade_integrity import _drive_worker, _FakeOpenSearch, _item


if TYPE_CHECKING:
    from pathlib import Path


pytestmark = pytest.mark.usefixtures('reference_region_profile')

ENV = 'OP_REGION_WORKER_MAX_NO_VERDICT_ATTEMPTS'


@pytest.fixture(autouse=True)
def _default_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    # reference_region_profile resets the process-wide count per test.
    monkeypatch.delenv(ENV, raising=False)


def _jpeg() -> bytes:
    buf = io.BytesIO()
    Image.new('RGB', (320, 240), (50, 80, 120)).save(buf, format='JPEG')
    return buf.getvalue()


def _no_verdict_reply() -> VlmCombinedReply:
    return VlmCombinedReply(
        img_id='c1',
        region_visible=True,
        region_boxes=[VlmBoxVerdict(box=1, bbox_correct=None, confidence=None)],
    )


class TestCombinedNoVerdictIsCapped:
    """Ported from the deleted ``TestCascadeVerifyNoVerdictIsCapped`` /
    ``TestCombinedCohortNoVerdictIsCapped`` -- same intent (bounded
    retries, then a reviewable park), proven against the live streaming
    pipeline via ``_drive_worker`` instead of the deleted per-crop
    cascade helpers."""

    @pytest.mark.asyncio
    async def test_null_box_verdict_is_parked_after_the_cap(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(ENV, '2')
        fake_os = _FakeOpenSearch({'c1': _item()}, search_delay=0.0, lag_searches=0)
        mocks = await _drive_worker(
            tmp_path,
            monkeypatch,
            fake_os=fake_os,
            primary=RegionCandidate(bbox_norm=(0.3, 0.6, 0.6, 0.75), score=0.9, source='det'),
            segmenter=None,
            reply=_no_verdict_reply(),
        )
        F = get_region_fields()
        doc = fake_os.live['c1']
        assert doc[F.status] == 'verify_rejected'
        assert doc[F.boxes][0]['rejection_reason'] == REJECT_REASON_NO_VERDICT
        assert doc[F.boxes][0]['bbox_correct'] is None
        # The reply's class side lands with it.
        assert doc[F.visible] is True
        assert 'vlm_verify_completed_at' in doc
        assert mocks['vlm'].label_combined_batch.await_count >= 2

    @pytest.mark.asyncio
    async def test_transport_failure_is_never_capped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(ENV, '2')

        async def _always_fails(crops: Any, **_kw: Any) -> Any:
            msg = 'upstream down'
            from src.services.labeling.vlm_models import CombinedTransportError

            raise CombinedTransportError(msg)

        fake_os = _FakeOpenSearch({'c1': _item()}, search_delay=0.0, lag_searches=0)
        mocks = await _drive_worker(
            tmp_path,
            monkeypatch,
            fake_os=fake_os,
            primary=RegionCandidate(bbox_norm=(0.3, 0.6, 0.6, 0.75), score=0.9, source='det'),
            segmenter=None,
            reply=_no_verdict_reply(),
            combined_side_effect=_always_fails,
            until_writes=0,
        )
        F = get_region_fields()
        # No write at all -- a transport failure is retried forever, never
        # counted toward the cap, never parked.
        assert fake_os.writes == []
        assert fake_os.live['c1'][F.status] == 'pending_detection'
        assert mocks['vlm'].label_combined_batch.await_count >= 2


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
        from src.services.labeling.vlm_models import CombinedTransportError

        assert issubclass(CombinedTransportError, VlmTransportError)
