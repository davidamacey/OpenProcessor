"""The detection worker on a text-free region profile, driven end to end.

A text-free profile (``text_reader='none'``) stores a region box and no
region text: a VLM reading is dropped, the region OCR reader never runs,
the OCR text-hint re-pass is optional (``text_hint_enabled``), and an
empty ``detector_model`` means there is no detector leg at all.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from src.config import get_region_fields
from src.services.detection.cascade_detect import RegionCandidate
from src.services.labeling.vlm_labeler import VlmCombinedReply

from .test_region_cascade_integrity import _FakeOpenSearch, _item, _profile
from .test_region_text_worker import _drive


if TYPE_CHECKING:
    from pathlib import Path


pytestmark = pytest.mark.usefixtures('reference_region_profile')

TEXT_FREE = {'text_reader': 'none', 'text_hint_enabled': False}
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


class TestTextFreeWriteHelpers:
    @pytest.fixture
    def text_free(self) -> Any:
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
