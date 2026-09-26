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
from src.services.detection.cascade_detect import RegionCandidate, RegionDetector
from src.services.labeling.vlm_labeler import VlmCombinedReply

from .test_region_cascade_integrity import _FakeOpenSearch, _item, _jpeg, _profile
from .test_region_text_worker import _drive


if TYPE_CHECKING:
    from pathlib import Path


pytestmark = pytest.mark.usefixtures('reference_region_profile')

TEXT_FREE: dict[str, Any] = {'text_reader': 'none', 'text_hint_enabled': False}
SEG_BOX = RegionCandidate(bbox_norm=(0.3, 0.6, 0.6, 0.75), score=0.5, source='seg')
VLM_URL = 'http://vlm.invalid:8000'


def _text_keys(doc: dict[str, Any]) -> list[str]:
    prefix = get_region_fields().text
    return [k for k in doc if k.startswith(prefix)]


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
            region_bbox_correct=True,
            region_text_reply='ABC1234',
            region_confidence='high',
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
        assert doc[F.bbox_norm]
        assert _text_keys(doc) == []
        mocks['ocr'].read_region_lines.assert_not_awaited()

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
        assert doc[F.bbox_norm]
        assert doc[F.verified] is False
        assert _text_keys(doc) == []
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
        assert mocks['seg'].segment.await_count == 1


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
        mocks['primary'].detect_batch.assert_not_awaited()
        assert doc[F.status] == 'detected'
        assert doc[F.detector] == seg
        chain = doc[F.detector_chain]
        assert not any(e.startswith(':') or e.endswith(':miss') for e in chain), chain

    @pytest.mark.asyncio
    async def test_detector_without_model_does_no_triton_io(self) -> None:
        pool = MagicMock()
        pool.infer = AsyncMock()
        detector = RegionDetector(pool, _profile().__class__(name='p', detector_model=''))
        assert await detector.detect(_jpeg()) is None
        assert await detector.detect_batch([_jpeg(), _jpeg()]) == [None, None]
        pool.infer.assert_not_awaited()


class TestTextFreeWriteHelpers:
    @pytest.fixture
    def text_free(self, reference_region_profile: None) -> Any:  # noqa: ARG002
        import dataclasses

        from src.services.detection.profile_registry import register_profile

        profile = dataclasses.replace(_profile(), text_reader='none')
        register_profile(profile, default=True)
        return profile

    @pytest.mark.usefixtures('text_free')
    def test_region_write_doc_drops_vlm_text(self) -> None:
        from scripts.curation.worker.verify import _region_write_doc

        doc = _region_write_doc(
            region_in_source=(0.1, 0.1, 0.2, 0.2),
            score=0.9,
            detector='seg',
            detector_version='1',
            chain=['seg:hit'],
            region_text_reply='ABC1234',
            region_text_confidence='high',
        )
        assert doc[get_region_fields().status] == 'detected'
        assert _text_keys(doc) == []

    def test_text_hint_fallback_is_a_no_op(self, text_free: Any) -> None:
        from scripts.curation.worker.region_text_stage import apply_text_hint_fallback
        from src.services.detection.region_text_rules import region_text_rules

        F = get_region_fields()
        doc: dict[str, Any] = {F.status: 'detected', F.text_vlm: 'stale'}
        apply_text_hint_fallback(
            doc,
            text='ABC1234',
            confidence=0.9,
            profile=text_free,
            rules=region_text_rules(text_free),
        )
        assert doc == {F.status: 'detected'}


class TestLegacyCascade:
    @pytest.mark.asyncio
    async def test_segmenter_only_text_free_miss(self) -> None:
        import dataclasses

        import scripts.curation.region_worker_main as worker
        from src.services.detection.profile_registry import register_profile

        from .test_region_worker import (
            _detector_mock,
            _make_task,
            _ocr_recognizer_mock,
            _segmenter_mock,
            _vlm_mock,
        )

        register_profile(
            dataclasses.replace(_profile(), detector_model='', **TEXT_FREE), default=True
        )
        detector = _detector_mock([None])
        ocr = _ocr_recognizer_mock()
        task = _make_task(status='pending_detection')
        await worker._process_crop(
            task,
            detector=detector,
            segmenter=_segmenter_mock(None),
            ocr_recognizer=ocr,
            vlm=_vlm_mock(is_region=True),
        )
        F = get_region_fields()
        assert task.update_doc[F.status] == 'no_region_box'
        assert task.update_doc[F.detector_chain] == [f'{_profile().segmenter_name}:miss']
        detector.detect_batch.assert_not_awaited()
        ocr.detect_regions.assert_not_awaited()
