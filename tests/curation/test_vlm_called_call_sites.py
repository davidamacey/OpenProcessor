"""m3 (W2-finish review, 2026-09-27): each ``vlm_called`` write site in
the detection worker actually gets exercised by a test that would fail if
that flag write were removed or set wrong.

Before this file, only the bulk-writer *gate* was tested
(``test_worker_hot_reload.py::test_bulk_write_does_not_stamp_pack_when_no_vlm_call_happened``)
-- proving the writer reads ``task.vlm_called`` correctly, never that any
of the five production call sites (the reviewer's mutation removed all
six ``vlm_called = True`` assignments and the full-suite pass/fail
outcome did not change outside that one test) actually *sets* it right.

Call sites covered here:
* the cascade verify path (``verify.py::_verify_with_vlm``, reached
  through all four of ``cascade.py``'s branches -- exercised here via the
  ``pending_verify`` leg),
* the combined single-crop path (``combined.py::_try_combined_class_region``),
* the Stage A visibility batch (``runner.py``'s ``region_visible_batch``
  call), and
* the Stage B combined batch (``runner.py``'s ``label_combined_batch`` call),

plus one negative case: the secondary-segmenter's high-confidence
auto-skip path (``cascade.py``'s ``_SKIP_VLM_VERIFY_SECONDARY_SCORE``
fast path) must NOT set ``vlm_called``, even with a VLM configured,
because it never calls it -- stamping ``vlm_prompt_pack`` there would
falsely claim a VLM ran.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

import pytest

import scripts.curation.region_worker_main as worker
from scripts.curation.worker import combined as combined_mod
from src.config import get_region_fields
from src.config.region_source import CANDIDATE_DETECTOR
from src.config.region_state import RegionStatus
from src.services.detection.cascade_detect import RegionCandidate
from src.services.labeling.vlm_labeler import VlmCombinedReply

from .test_cascade_no_verdict_cap import _accept, _mocks, _pass, _task, _vlm
from .test_region_cascade_integrity import (
    _accept as _combined_accept,
    _drive_worker,
    _FakeOpenSearch,
    _item,
)


if TYPE_CHECKING:
    from pathlib import Path


pytestmark = pytest.mark.usefixtures('reference_region_profile')


@pytest.mark.asyncio
async def test_cascade_verify_path_sets_vlm_called() -> None:
    """``_verify_with_vlm`` (verify.py:80), reached from cascade.py's
    ``pending_verify`` leg here, stamps ``task.vlm_called`` on every
    answered round trip -- proven by an accepted verdict producing a
    real write."""
    vlm = _vlm(_accept())
    task = await _pass(vlm)
    assert task.vlm_called is True
    assert task.update_doc  # a real write happened -- the round trip counted


@pytest.mark.asyncio
async def test_combined_single_crop_path_sets_vlm_called() -> None:
    """``combined.py::_try_combined_class_region`` (combined.py:141) sets
    ``task.vlm_called`` once ``vlm.label_combined`` actually answers,
    whatever the verdict."""
    F = get_region_fields()
    task = _task(region_status='pending_detection', detector_region_in_source=None)
    reply = VlmCombinedReply(
        img_id=task.crop_id,
        region_visible=True,
        region_bbox_correct=True,
        region_confidence='high',
    )
    vlm = MagicMock()
    vlm.class_names = []
    vlm.label_combined = AsyncMock(return_value=reply)
    det = worker.region_profile().detector_model

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
    assert task.vlm_called is True
    assert task.update_doc.get(F.status) == RegionStatus.DETECTED


@pytest.mark.asyncio
async def test_stage_a_visibility_batch_sets_vlm_called(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Runner Stage A's ``region_visible_batch`` call (runner.py:919-924)
    stamps every chunked task with a jpeg ``vlm_called = True`` before it
    even knows the verdict. Driven with ``visible=False`` so the item
    terminates at Stage A (``no_region_visible``) without ever reaching
    Stage B's ``label_combined_batch`` -- the resulting write's
    ``vlm_prompt_pack`` stamp can only be Stage A's doing."""
    F = get_region_fields()
    fake_os = _FakeOpenSearch({'c1': _item()}, search_delay=0.0, lag_searches=0)
    await _drive_worker(
        tmp_path,
        monkeypatch,
        fake_os=fake_os,
        primary=None,
        segmenter=None,
        reply=_combined_accept(),
        visible=False,
    )
    doc = fake_os.live['c1']
    assert doc[F.status] == RegionStatus.NO_REGION_VISIBLE
    assert doc.get('vlm_prompt_pack')


@pytest.mark.asyncio
async def test_stage_b_combined_batch_sets_vlm_called(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Runner Stage B's ``label_combined_batch`` call (runner.py:1286-1294)
    stamps every chunked task with a jpeg ``vlm_called = True`` -- the
    resulting ``detected`` write carries ``vlm_prompt_pack``."""
    F = get_region_fields()
    fake_os = _FakeOpenSearch({'c1': _item()}, search_delay=0.0, lag_searches=0)
    await _drive_worker(
        tmp_path,
        monkeypatch,
        fake_os=fake_os,
        primary=RegionCandidate(bbox_norm=(0.3, 0.6, 0.6, 0.75), score=0.9, source='det'),
        segmenter=None,
        reply=_combined_accept(),
    )
    doc = fake_os.live['c1']
    assert doc[F.status] == 'detected'
    assert doc.get('vlm_prompt_pack')


@pytest.mark.asyncio
async def test_segmenter_auto_skip_does_not_set_vlm_called() -> None:
    """cascade.py's high-confidence secondary-segmenter fast path
    (``_SKIP_VLM_VERIFY_SECONDARY_SCORE``) bypasses the VLM verify round
    trip entirely -- ``task.vlm_called`` must stay False even with a VLM
    configured, since it never actually calls it. A regression that set
    the flag here would falsely claim a VLM ran on this write."""
    F = get_region_fields()
    task = _task(region_status='pending_detection', detector_region_in_source=None)
    vlm = _vlm(None)
    mocks = _mocks()
    mocks['segmenter'].segment = AsyncMock(
        return_value=RegionCandidate(bbox_norm=(0.3, 0.6, 0.6, 0.75), score=0.99, source='sam')
    )

    await worker._process_crop(task, vlm=vlm, **mocks)

    assert task.update_doc.get(F.skip_verify) is True  # confirms the fast path actually fired
    assert task.vlm_called is False
    vlm.verify_region.assert_not_awaited()
    assert 'vlm_prompt_pack' not in task.update_doc
