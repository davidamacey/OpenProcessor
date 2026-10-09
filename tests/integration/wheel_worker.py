"""Drive the region detection worker in-process over the routed fake cluster.

The worker is production code (``runner.run``); only its network boundaries
are replaced: OpenSearch (the shared in-memory cluster), Triton, the
segmenter service client and the VLM client. The same pattern as
``tests/curation/test_region_text_worker._drive``, but over a project that
was created through the lifecycle route instead of a single hard-wired one.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock

from _fake_project_registry import install_static_project_registry

import scripts.curation.region_worker_main as worker
from scripts.curation.worker import runner as runner_mod
from src.services.labeling.region_overlay import VlmBoxVerdict
from src.services.labeling.vlm_client import VlmIdentity
from src.services.labeling.vlm_models import VlmCombinedReply


if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    import pytest

    from src.config.projects import ProjectRecord
    from src.services.detection.cascade_detect.candidate import RegionCandidate


def _capture_signal_handler(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    handlers: list[Any] = []

    def _record(_self: Any, _sig: Any, cb: Any, *_a: Any) -> None:
        handlers.append(cb)

    monkeypatch.setattr(
        asyncio.get_event_loop().__class__, 'add_signal_handler', _record, raising=False
    )
    return handlers


def accept_lower_half(crops: list[Any]) -> dict[str, VlmCombinedReply]:
    """A VLM stand-in with one visible rule: a numbered box is a wheel only
    if its centre is in the lower half of the crop (a roof box is not)."""
    out: dict[str, VlmCombinedReply] = {}
    for crop in crops:
        verdicts = [
            VlmBoxVerdict(
                box=i,
                bbox_correct=(box[1] + box[3]) / 2 > 0.5,
                confidence='high',
            )
            for i, box in enumerate(crop.region_bboxes_norm, start=1)
        ]
        out[crop.crop_id] = VlmCombinedReply(
            img_id=crop.crop_id, region_visible=True, region_boxes=verdicts
        )
    return out


class WorkerMocks:
    def __init__(self) -> None:
        self.segment_calls: list[str] = []
        self.vlm_batches: list[list[str]] = []


async def run_region_worker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    cluster: Any,
    records: list[ProjectRecord],
    segmenter: Callable[[], list[RegionCandidate]],
    done: Callable[[], bool],
    vlm: Callable[[list[Any]], dict[str, VlmCombinedReply]] = accept_lower_half,
) -> WorkerMocks:
    mocks = WorkerMocks()
    handlers = _capture_signal_handler(monkeypatch)
    monkeypatch.setenv('OP_REGION_WORKER_METRICS_PORT', '0')
    monkeypatch.setenv('OP_VLM_URL', 'http://vlm.invalid:8000')

    pool = MagicMock(initialize=AsyncMock(), close=AsyncMock())
    monkeypatch.setattr(worker, 'AsyncTritonPool', MagicMock(return_value=pool))
    monkeypatch.setattr(worker, 'make_script_opensearch', MagicMock(return_value=cluster))
    install_static_project_registry(monkeypatch, records)

    primary_det = MagicMock()
    primary_det.confidence_floor = 0.0
    primary_det.detect_batch = AsyncMock(return_value=[None])
    primary_det.detect_batch_multi = AsyncMock(return_value=[[]])
    monkeypatch.setattr(runner_mod, 'RegionDetector', MagicMock(return_value=primary_det))
    ocr = MagicMock()
    ocr.detect_regions = AsyncMock(return_value=[])
    ocr.pick_best_text_region = MagicMock(return_value=None)
    monkeypatch.setattr(runner_mod, 'PaddleOcrTextRecognizer', MagicMock(return_value=ocr))

    seg = MagicMock(aclose=AsyncMock())

    async def _segment_multi(*_a: Any, **_k: Any) -> list[RegionCandidate]:
        mocks.segment_calls.append('segment_multi')
        return segmenter()

    seg.segment_multi = AsyncMock(side_effect=_segment_multi)

    async def _segment(*_a: Any, **_k: Any) -> RegionCandidate | None:
        found = segmenter()
        return found[0] if found else None

    seg.segment = AsyncMock(side_effect=_segment)
    monkeypatch.setattr(worker, 'SegmenterClient', MagicMock(return_value=seg))

    labeler = MagicMock(aclose=AsyncMock())
    labeler.class_names = []
    labeler.identity = VlmIdentity(endpoint_ref='fake-vlm@1', model='fake-vlm')

    async def _combined(crops: list[Any], **_kw: Any) -> dict[str, VlmCombinedReply]:
        mocks.vlm_batches.append([c.crop_id for c in crops])
        return vlm(crops)

    labeler.label_combined_batch = AsyncMock(side_effect=_combined)
    labeler.region_visible_batch = AsyncMock(
        side_effect=lambda crops, **_kw: dict.fromkeys((c.crop_id for c in crops), True)
    )
    monkeypatch.setattr(worker, 'build_vlm_labeler', MagicMock(return_value=labeler))
    monkeypatch.setattr('scripts.curation.worker.state._class_group', lambda _name: None)

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
        for _ in range(1500):
            await asyncio.sleep(0.02)
            if done():
                break
        await asyncio.sleep(0.3)
        handlers[0]()

    stopper = asyncio.create_task(_stopper())
    rc = await asyncio.wait_for(runner_mod.run(args), timeout=60)
    await stopper
    assert rc == 0
    return mocks


__all__ = ['WorkerMocks', 'accept_lower_half', 'run_region_worker']
