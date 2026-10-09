"""The detection worker on a text-free region profile, driven end to end.

A text-free profile (``text_reader='none'``) stores a region box and no
region text: a VLM reading is dropped, the region OCR reader never runs,
the OCR text-hint re-pass is optional (``text_hint_enabled``), and an
empty ``detector_model`` means there is no detector leg at all.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.config import get_region_fields
from src.services.detection.cascade_detect.candidate import RegionCandidate
from src.services.detection.cascade_detect.region_detector import RegionDetector
from src.services.labeling.region_overlay import VlmBoxVerdict
from src.services.labeling.vlm_models import VlmCombinedReply

from .test_region_cascade_integrity import _FakeOpenSearch, _item, _jpeg, _profile
from .test_region_text_worker import _drive


if TYPE_CHECKING:
    from pathlib import Path


pytestmark = pytest.mark.usefixtures('reference_region_profile')

TEXT_FREE: dict[str, Any] = {'text_reader': 'none', 'text_hint_enabled': False}
SEG_BOX = RegionCandidate(bbox_norm=(0.3, 0.6, 0.6, 0.75), score=0.5, source='seg')
VLM_URL = 'http://vlm.invalid:8000'


def _text_keys(doc: dict[str, Any]) -> list[str]:
    """Item-level (flat) text keys -- there must be none: text lives on boxes."""
    return [k for k in doc if k.startswith('region_text')]


_BOX_TEXT_ATTRS = (
    'text',
    'text_raw',
    'text_source',
    'text_engine_version',
    'text_confidence',
    'text_vlm',
    'text_ocr',
    'text_choice',
    'text_vlm_invalid',
    'text_disagreement',
)


def _box_text_values(box: dict[str, Any]) -> list[Any]:
    """Every text-ish value set on one ``region_boxes`` element."""
    return [box.get(attr) for attr in _BOX_TEXT_ATTRS if box.get(attr) is not None]


def _fake_os() -> _FakeOpenSearch:
    return _FakeOpenSearch({'c1': _item()}, search_delay=0.0, lag_searches=0)


class TestTextFreeWrites:
    @pytest.mark.asyncio
    async def test_vlm_reading_is_dropped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_os = _fake_os()
        reply = VlmCombinedReply(
            img_id='c1',
            region_visible=True,
            region_boxes=[
                VlmBoxVerdict(box=1, bbox_correct=True, confidence='high', text_reply='ABC1234')
            ],
        )
        mocks = await _drive(
            tmp_path,
            monkeypatch,
            fake_os=fake_os,
            primary=None,
            segmenter=SEG_BOX,
            vlm_url=VLM_URL,
            reply=reply,
            profile_overrides=TEXT_FREE,
        )
        F = get_region_fields()
        doc = fake_os.live['c1']
        assert doc[F.status] == 'detected'
        box = doc[F.boxes][0]
        assert box['bbox_norm']
        assert _text_keys(doc) == []
        assert _box_text_values(box) == []
        mocks['ocr'].read_region_lines.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_rejected_box_vlm_reading_is_also_dropped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """W8 M6 fix (pipeline-wiring review, 2026-09-27): the text-free
        leak fix only ever covered ACCEPTED boxes
        (``_box_with_resolved_text`` runs on the per-box text-resolution
        loop's accepted branch only). A REJECTED box's ``text`` came
        straight from ``verdicts_to_boxes``'s raw ``verdict.text_reply``
        with no profile gate at all -- proven here with the same VLM
        text reading as the accepted-case test above, but with
        ``bbox_correct=False`` (rejected) instead of ``True``."""
        fake_os = _fake_os()
        reply = VlmCombinedReply(
            img_id='c1',
            region_visible=True,
            region_boxes=[
                VlmBoxVerdict(box=1, bbox_correct=False, confidence='high', text_reply='ABC1234')
            ],
        )
        await _drive(
            tmp_path,
            monkeypatch,
            fake_os=fake_os,
            primary=None,
            segmenter=SEG_BOX,
            vlm_url=VLM_URL,
            reply=reply,
            profile_overrides=TEXT_FREE,
        )
        F = get_region_fields()
        doc = fake_os.live['c1']
        assert doc[F.status] == 'verify_rejected'
        box = doc[F.boxes][0]
        assert box['state'] == 'rejected'
        assert _text_keys(doc) == []
        assert _box_text_values(box) == []

    @pytest.mark.asyncio
    async def test_no_vlm_writes_box_without_ocr(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_os = _fake_os()
        mocks = await _drive(
            tmp_path,
            monkeypatch,
            fake_os=fake_os,
            primary=None,
            segmenter=SEG_BOX,
            vlm_url='',
            profile_overrides={**TEXT_FREE, 'ocr_pipeline_model': ''},
        )
        F = get_region_fields()
        doc = fake_os.live['c1']
        assert doc[F.status] == 'detected'
        box = doc[F.boxes][0]
        assert box['bbox_norm']
        # W8: no verified/validated concept per box (RegionBox carries
        # none) -- state='accepted' is the machine-acceptance signal.
        assert box['state'] == 'accepted'
        assert _text_keys(doc) == []
        assert _box_text_values(box) == []
        mocks['ocr'].read_region_lines.assert_not_awaited()
        mocks['ocr'].read_lines.assert_not_awaited()
        mocks['ocr'].detect_regions.assert_not_awaited()


class TestTextHintOptional:
    @pytest.mark.asyncio
    async def test_segmenter_miss_without_hint_is_no_region_box(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_os = _fake_os()
        mocks = await _drive(
            tmp_path,
            monkeypatch,
            fake_os=fake_os,
            primary=None,
            segmenter=None,
            vlm_url='',
            profile_overrides={**TEXT_FREE, 'ocr_pipeline_model': ''},
        )
        F = get_region_fields()
        doc = fake_os.live['c1']
        seg = _profile().segmenter_name
        assert doc[F.status] == 'no_region_box'
        chain = doc[F.detector_chain]
        assert not any('text_hint' in e for e in chain)
        assert chain[-1] == f'{seg}:miss'
        mocks['ocr'].detect_regions.assert_not_awaited()
        mocks['ocr'].pick_best_text_region.assert_not_called()
        assert mocks['seg'].segment_multi.await_count == 1


class TestNoDetectorLeg:
    @pytest.mark.asyncio
    async def test_empty_detector_model_skips_the_detector(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_os = _fake_os()
        mocks = await _drive(
            tmp_path,
            monkeypatch,
            fake_os=fake_os,
            primary=None,
            segmenter=SEG_BOX,
            vlm_url='',
            profile_overrides={**TEXT_FREE, 'detector_model': ''},
        )
        F = get_region_fields()
        doc = fake_os.live['c1']
        seg = _profile().segmenter_name
        mocks['primary'].detect_batch_multi.assert_not_awaited()
        assert doc[F.status] == 'detected'
        box = doc[F.boxes][0]
        assert box['detector'] == seg
        chain = doc[F.detector_chain]
        assert not any(e.startswith(':') or e.endswith(':miss') for e in chain), chain

    @pytest.mark.asyncio
    async def test_detector_without_model_does_no_triton_io(self) -> None:
        pool = MagicMock()
        pool.infer = AsyncMock()
        detector = RegionDetector(pool, _profile().__class__(name='p', detector_model=''))
        assert await detector.detect(_jpeg()) is None
        assert await detector.detect_batch([_jpeg(), _jpeg()]) == [None, None]
        assert await detector.detect_multi(_jpeg()) == []
        assert await detector.detect_batch_multi([_jpeg(), _jpeg()]) == [[], []]
        pool.infer.assert_not_awaited()


class TestTextFreeWriteHelpers:
    @pytest.fixture
    def text_free(self, reference_region_profile: None) -> Any:  # noqa: ARG002
        import dataclasses

        from src.services.detection.profile_registry import register_profile

        profile = dataclasses.replace(_profile(), text_reader='none')
        register_profile(profile, default=True)
        return profile

    def test_text_hint_fallback_is_a_no_op(self, text_free: Any) -> None:
        from scripts.curation.worker.region_text_stage import apply_text_hint_fallback
        from src.services.detection.region_text_rules import region_text_rules

        doc: dict[str, Any] = {'text_choice': 'vlm_invalid', 'text_vlm': 'stale'}
        apply_text_hint_fallback(
            doc,
            text='ABC1234',
            confidence=0.9,
            profile=text_free,
            rules=region_text_rules(text_free),
        )
        assert doc == {}


class TestLegacyCascade:
    @pytest.mark.asyncio
    async def test_segmenter_only_text_free_miss(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No detector leg (text-free, ``detector_model=''``) and the
        segmenter also misses -> terminal ``no_region_box``, detector
        never called. W8: ported off the deleted per-crop
        ``_process_crop`` onto the real streaming pipeline via
        ``_drive``."""
        fake_os = _fake_os()
        mocks = await _drive(
            tmp_path,
            monkeypatch,
            fake_os=fake_os,
            primary=None,
            segmenter=None,
            vlm_url='',
            profile_overrides={**TEXT_FREE, 'detector_model': '', 'ocr_pipeline_model': ''},
        )
        F = get_region_fields()
        doc = fake_os.live['c1']
        assert doc[F.status] == 'no_region_box'
        assert doc[F.detector_chain] == [f'{_profile().segmenter_name}:miss']
        mocks['primary'].detect_batch_multi.assert_not_awaited()
        mocks['ocr'].detect_regions.assert_not_awaited()
