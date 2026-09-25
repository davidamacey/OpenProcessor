"""The detection worker's OCR text readers, driven end to end.

* No VLM configured: detector regions are accepted unverified and their
  text is read by OCR; the VLM client is never built.
* ``text_reader='both'``: the VLM and OCR readings are stored side by side
  and a disagreement is flagged.
* Item text: every OCR line on the item crop lands on the item.
"""

from __future__ import annotations

import asyncio
import dataclasses
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock

import pytest

import scripts.curation.region_worker_main as worker
from scripts.curation.worker import runner as runner_mod
from src.config import get_region_fields
from src.services.detection.cascade_detect import RegionCandidate
from src.services.detection.region_text import OcrLine
from src.services.labeling.vlm_labeler import VlmCombinedReply

from .test_region_cascade_integrity import (
    _capture_signal_handler,
    _FakeOpenSearch,
    _item,
    _jpeg,
    _profile,
)


if TYPE_CHECKING:
    from pathlib import Path


pytestmark = pytest.mark.usefixtures('reference_region_profile')


REGION_LINES = [
    OcrLine('Ohio', (0.35, 0.12, 0.65, 0.24), 0.95),
    OcrLine('ABC-1234', (0.12, 0.30, 0.88, 0.72), 0.91),
    OcrLine('Birthplace of Aviation', (0.20, 0.75, 0.80, 0.85), 0.80),
]
ITEM_LINES = [
    OcrLine('ABC-1234', (0.30, 0.60, 0.55, 0.70), 0.93),
    OcrLine('Smith Motors', (0.28, 0.72, 0.58, 0.76), 0.81),
]


async def _drive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    fake_os: _FakeOpenSearch,
    primary: RegionCandidate | None,
    segmenter: RegionCandidate | None,
    vlm_url: str,
    reply: VlmCombinedReply | None = None,
    text_reader: str | None = None,
) -> dict[str, Any]:
    handlers = _capture_signal_handler(monkeypatch)
    monkeypatch.setenv('SAM_WORKER_METRICS_PORT', '0')
    if text_reader is not None:
        profile = dataclasses.replace(_profile(), text_reader=text_reader)
        monkeypatch.setattr(runner_mod, 'get_active_region_profile', lambda: profile)

    pool = MagicMock(initialize=AsyncMock(), close=AsyncMock())
    monkeypatch.setattr(worker, 'AsyncTritonPool', MagicMock(return_value=pool))
    monkeypatch.setattr(worker, 'AsyncOpenSearch', MagicMock(return_value=fake_os))

    primary_det = MagicMock()
    primary_det.detect_batch = AsyncMock(return_value=[primary])
    monkeypatch.setattr(runner_mod, 'RegionDetector', MagicMock(return_value=primary_det))

    ocr = MagicMock()
    ocr.read_lines = AsyncMock(return_value=ITEM_LINES)
    ocr.read_region_lines = AsyncMock(return_value=REGION_LINES)
    ocr.regions_from_lines = MagicMock(return_value=[])
    ocr.detect_regions = AsyncMock(return_value=[])
    ocr.pick_best_text_region = MagicMock(return_value=None)
    monkeypatch.setattr(runner_mod, 'PaddleOcrTextRecognizer', MagicMock(return_value=ocr))
    monkeypatch.setattr(runner_mod, '_crop_jpeg_for_task', lambda *_a: _jpeg())

    seg = MagicMock(aclose=AsyncMock())
    seg.segment = AsyncMock(return_value=segmenter)
    monkeypatch.setattr(worker, 'SegmenterClient', MagicMock(return_value=seg))

    vlm = MagicMock(aclose=AsyncMock())
    vlm.class_names = []
    vlm.label_combined_batch = AsyncMock(
        side_effect=lambda crops, **_kw: {c.crop_id: reply for c in crops}
    )
    vlm.region_visible_batch = AsyncMock(
        side_effect=lambda crops, **_kw: dict.fromkeys((c.crop_id for c in crops), True)
    )
    vlm_cls = MagicMock(return_value=vlm)
    monkeypatch.setattr(worker, 'VlmLabeler', vlm_cls)
    monkeypatch.setattr(
        'src.clients.curation_opensearch.ClassRegistry',
        MagicMock(side_effect=RuntimeError('no registry in test')),
    )
    # Stage A resolves the class group through the process-wide registry
    # singleton; without this the test depends on another test having
    # loaded it first.
    monkeypatch.setattr('scripts.curation.worker.state._class_group', lambda _name: None)

    args = worker.parse_args(
        [
            '--opensearch=http://os.invalid:9200',
            '--triton=triton.invalid:8001',
            '--sam3-url=http://seg.invalid:8000',
            f'--vlm-url={vlm_url}',
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
    return {'primary': primary_det, 'seg': seg, 'vlm': vlm, 'vlm_cls': vlm_cls, 'ocr': ocr}


BOX = RegionCandidate(bbox_norm=(0.3, 0.6, 0.6, 0.75), score=0.9, source='det')


class TestNoVlmDeployment:
    @pytest.mark.asyncio
    async def test_primary_region_accepted_with_ocr_text(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_os = _FakeOpenSearch({'c1': _item()}, search_delay=0.0, lag_searches=0)
        mocks = await _drive(
            tmp_path, monkeypatch, fake_os=fake_os, primary=BOX, segmenter=None, vlm_url=''
        )
        mocks['vlm_cls'].assert_not_called()
        F = get_region_fields()
        doc = fake_os.live['c1']
        det = _profile().detector_model
        assert doc[F.status] == 'detected'
        assert doc[F.verified] is False
        assert doc[F.validated] is False
        assert F.verifier not in doc
        assert doc[F.detector] == det
        assert doc[F.detector_chain] == [f'{det}:hit', f'{det}:accepted_unverified']
        assert doc[F.text] == 'ABC1234'
        assert doc[F.text_source] == 'ocr'
        assert doc[F.text_ocr] == 'ABC1234'
        assert doc[F.text_raw] == 'Ohio ABC-1234 Birthplace of Aviation'
        assert doc[F.text_confidence] == pytest.approx(0.91)
        assert doc[F.text_engine_version] == 'paddleocr_det_trt:1+paddleocr_rec_trt:1'
        assert F.text_vlm not in doc
        assert F.text_disagreement not in doc
        assert [ln['text'] for ln in doc['item_text_lines']] == ['ABC-1234', 'Smith Motors']
        assert 'SMITH' in doc['item_text_tokens']

    @pytest.mark.asyncio
    async def test_segmenter_region_accepted_without_visibility_call(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_os = _FakeOpenSearch({'c1': _item()}, search_delay=0.0, lag_searches=0)
        seg_box = RegionCandidate(bbox_norm=(0.3, 0.6, 0.6, 0.75), score=0.5, source='seg')
        await _drive(
            tmp_path, monkeypatch, fake_os=fake_os, primary=None, segmenter=seg_box, vlm_url=''
        )
        F = get_region_fields()
        doc = fake_os.live['c1']
        seg = _profile().segmenter_name
        assert doc[F.status] == 'detected'
        assert doc[F.detector] == seg
        assert f'{seg}:accepted_unverified' in doc[F.detector_chain]
        assert not any(e.startswith('vlm_visible') for e in doc[F.detector_chain])
        assert doc[F.text] == 'ABC1234'


class TestBothReaders:
    @pytest.mark.asyncio
    async def test_disagreement_is_flagged_and_vlm_reading_chosen(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        assert _profile().text_reader == 'both'
        fake_os = _FakeOpenSearch({'c1': _item()}, search_delay=0.0, lag_searches=0)
        reply = VlmCombinedReply(
            img_id='c1',
            region_visible=True,
            region_bbox_correct=True,
            region_text_reply='ABC 1284',
            region_confidence='high',
        )
        mocks = await _drive(
            tmp_path,
            monkeypatch,
            fake_os=fake_os,
            primary=BOX,
            segmenter=None,
            vlm_url='http://vlm.invalid:8000',
            reply=reply,
        )
        F = get_region_fields()
        doc = fake_os.live['c1']
        assert doc[F.status] == 'detected'
        assert doc[F.text] == 'ABC 1284'
        assert doc[F.text_source] == 'vlm'
        assert doc[F.text_vlm] == 'ABC 1284'
        assert doc[F.text_ocr] == 'ABC1234'
        assert doc[F.text_disagreement] is True
        # The region OCR ran on the candidate box, framed for small text.
        call = mocks['ocr'].read_region_lines.await_args
        assert call.kwargs['min_height'] == _profile().text_crop_min_height

    @pytest.mark.asyncio
    async def test_vlm_then_ocr_falls_back_when_vlm_reads_nothing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_os = _FakeOpenSearch({'c1': _item()}, search_delay=0.0, lag_searches=0)
        reply = VlmCombinedReply(
            img_id='c1',
            region_visible=True,
            region_bbox_correct=True,
            region_text_reply=None,
            region_confidence='high',
        )
        await _drive(
            tmp_path,
            monkeypatch,
            fake_os=fake_os,
            primary=BOX,
            segmenter=None,
            vlm_url='http://vlm.invalid:8000',
            reply=reply,
            text_reader='vlm_then_ocr',
        )
        F = get_region_fields()
        doc = fake_os.live['c1']
        assert doc[F.text] == 'ABC1234'
        assert doc[F.text_source] == 'ocr'
        assert F.text_disagreement not in doc

    @pytest.mark.asyncio
    async def test_vlm_mode_never_reads_region_ocr(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_os = _FakeOpenSearch({'c1': _item()}, search_delay=0.0, lag_searches=0)
        reply = VlmCombinedReply(
            img_id='c1',
            region_visible=True,
            region_bbox_correct=True,
            region_text_reply='ABC1234',
            region_confidence='medium',
        )
        mocks = await _drive(
            tmp_path,
            monkeypatch,
            fake_os=fake_os,
            primary=BOX,
            segmenter=None,
            vlm_url='http://vlm.invalid:8000',
            reply=reply,
            text_reader='vlm',
        )
        mocks['ocr'].read_region_lines.assert_not_awaited()
        doc = fake_os.live['c1']
        assert doc[get_region_fields().text_source] == 'vlm'
