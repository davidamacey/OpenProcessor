"""m3 (W2-finish review, 2026-09-27): each ``vlm_called`` write site in
the detection worker actually gets exercised by a test that would fail if
that flag write were removed or set wrong.

Before this file, only the bulk-writer *gate* was tested
(``test_worker_hot_reload.py::test_bulk_write_does_not_stamp_pack_when_no_vlm_call_happened``)
-- proving the writer reads ``task.vlm_called`` correctly, never that any
of the four production call sites (the reviewer's mutation removed all
four ``vlm_called = True`` assignments and the full-suite pass/fail
outcome did not change outside that one test) actually *sets* it right.

Call sites covered here (post-W8: the pre-W8 per-crop cascade
``_process_crop`` / ``combined.py`` cohort path was deleted once
``runner.py``'s streaming pipeline became the only production cascade --
its two ``vlm_called`` sites are gone with it, ported below onto the two
that remain):
* the Stage A visibility batch (``runner.py``'s ``region_visible_batch``
  call), and
* the Stage B combined batch (``runner.py``'s ``label_combined_batch`` call),

plus one negative case: the secondary-segmenter's high-confidence
auto-skip path (``runner.py``'s ``_SKIP_VLM_VERIFY_SECONDARY_SCORE``
fast path) must NOT set ``vlm_called``, even with a VLM configured,
because it never calls it -- stamping ``vlm_prompt_pack`` there would
falsely claim a VLM ran.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from src.config import get_region_fields
from src.config.region_state import RegionStatus
from src.services.detection.cascade_detect import RegionCandidate

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
async def test_segmenter_auto_skip_does_not_call_the_combined_vlm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """runner.py's high-confidence secondary-segmenter fast path
    (``stage_a_sam_consumer``'s ``_SKIP_VLM_VERIFY_SECONDARY_SCORE``
    branch) bypasses the combined (class + region-verify) VLM call
    entirely for this box -- a regression that routed it through
    ``combined_q`` anyway would introduce a redundant VLM round trip
    ``_SKIP_VLM_VERIFY_SECONDARY_SCORE`` exists to avoid.

    Note: this crop still visits the cheap Stage A visibility pre-filter
    first (``vlm_visible_q`` — every primary-miss crop does, whether a
    VLM is configured or not), so ``vlm_prompt_pack`` legitimately IS
    stamped from that call; it is not evidence the combined call ran.
    """
    F = get_region_fields()
    fake_os = _FakeOpenSearch({'c1': _item()}, search_delay=0.0, lag_searches=0)
    mocks = await _drive_worker(
        tmp_path,
        monkeypatch,
        fake_os=fake_os,
        primary=None,
        segmenter=RegionCandidate(bbox_norm=(0.3, 0.6, 0.6, 0.75), score=0.99, source='sam'),
        reply=_combined_accept(),
        visible=True,
    )
    doc = fake_os.live['c1']
    assert doc.get(F.skip_verify) is True  # confirms the fast path actually fired
    assert doc[F.status] == 'detected'
    mocks['vlm'].label_combined_batch.assert_not_awaited()
