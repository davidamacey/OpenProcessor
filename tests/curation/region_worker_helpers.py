"""Builders and drivers shared by the region worker tests."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock

from _fake_project_registry import install_static_project_registry

import scripts.curation.worker.state as worker_state
from curation.region_cascade_helpers import _capture_signal_handler, _FakeOpenSearch, _item, _jpeg
from scripts.curation.worker import runner as runner_mod, stage_a as stage_a_mod
from scripts.curation.worker.__main__ import parse_args
from src.config.project_context import current_project
from src.services.curation.region_boxes import RegionBox, boxes_write_fields
from src.services.labeling.region_overlay import VlmBoxVerdict
from src.services.labeling.vlm_models import VlmCombinedReply


if TYPE_CHECKING:
    from pathlib import Path

    import pytest

    from src.services.detection.cascade_detect.candidate import RegionCandidate


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
) -> worker_state._ItemTask:
    """Build a fully populated ``_ItemTask`` for the (non-cascade)
    bulk-write / secondary-shape tests below."""
    return worker_state._ItemTask(
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
    monkeypatch.setattr(runner_mod, 'AsyncTritonPool', MagicMock(return_value=pool))
    monkeypatch.setattr(runner_mod, 'make_script_opensearch', MagicMock(return_value=fake_os))
    install_static_project_registry(monkeypatch)

    primary_det = MagicMock()
    primary_det.confidence_floor = 0.0
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
    monkeypatch.setattr(runner_mod, 'SegmenterClient', MagicMock(return_value=seg))

    vlm = MagicMock(aclose=AsyncMock())
    vlm.class_names = []
    vlm.label_combined_batch = AsyncMock(
        side_effect=lambda crops, **_kw: {c.crop_id: reply for c in crops}
    )
    vlm.region_visible_batch = AsyncMock(
        side_effect=lambda crops, **_kw: dict.fromkeys((c.crop_id for c in crops), True)
    )
    monkeypatch.setattr(runner_mod, 'build_vlm_labeler', MagicMock(return_value=vlm))
    monkeypatch.setattr(
        'src.clients.curation_opensearch.registry.ClassRegistry',
        MagicMock(side_effect=RuntimeError('no registry in test')),
    )
    monkeypatch.setattr('scripts.curation.worker.state._class_group', lambda _name: None)

    monkeypatch.setenv('OP_VLM_URL', 'http://vlm.invalid:8000')
    args = parse_args(
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
