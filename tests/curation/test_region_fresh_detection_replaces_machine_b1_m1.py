"""W8c B1 + M1 regression (2026-09-28 re-review, "fix pass" for the
review at docs/design/openprocessor_internal/w8c_slice1_review_2026-09-28.md).

``pending_merge`` used to mean two different things at once:

1. Path 1 (human re-verification of a stored ``proposed`` box) needing
   its resolved verdict MERGED onto the stored list, preserving any
   untouched sibling by id (the real B1 fix target,
   ``test_region_pending_verification_b1.py`` / ``test_region_not_visible_
   terminal_r_b1.py``).
2. Every fresh-detection pass (Path 2/3) ALSO setting the same flag, for
   an entirely different (M1) reason: keeping a stored HUMAN box while
   the pass's own new candidates replace whatever MACHINE boxes were
   there.

Reusing one flag for both broke (1)'s specific downstream consumer at
``runner.py``'s ``region_visible=False`` branch, which read
``pending_merge`` to mean "this is Path 1" -- a fresh item (no stored
boxes at all) that got ``region_visible=False`` was misrouted into the
re-verify branch and wrote a phantom ``rejected`` box instead of the
correct empty ``no_region_visible`` (B1). Separately, treating a fresh
detection as a "merge" onto the FULL stored list (not just its human
boxes) let a stale MACHINE box accumulate forever and override the new
pass's own derived status (M1).

This module reproduces both fixed bugs directly, plus the combined case
(human box + stale machine box, both present) the fix must get right at
the same time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from src.config import get_region_fields
from src.config.region_state import RegionStatus
from src.services.detection.cascade_detect.candidate import RegionCandidate
from src.services.labeling.vlm_models import VlmCombinedReply

from .test_region_cascade_integrity import _accept, _drive_worker, _FakeOpenSearch, _item


if TYPE_CHECKING:
    from pathlib import Path


pytestmark = pytest.mark.usefixtures('reference_region_profile')

F = get_region_fields()


def _stale_machine_box(box_id: str = 'b_stale') -> dict[str, Any]:
    """A rejected box a PRIOR detection pass wrote and a
    ``clear_detection=False`` requeue left behind (the documented default
    requeue mode, ``region_requeue.apply_requeue``) -- never human-owned."""
    return {
        'box_id': box_id,
        'bbox_norm': [0.2, 0.2, 0.4, 0.3],
        'state': 'rejected',
        'score': 0.4,
        'detector': 'old_det',
        'detector_version': '1',
        'source': 'detector',
        'rejection_reason': 'sanity_reject:aspect_ratio',
    }


def _human_box(box_id: str = 'b_human') -> dict[str, Any]:
    """A box a human created via ``PUT /crops/{id}/regions`` -- must
    never be touched by a later fresh-detection pass."""
    return {
        'box_id': box_id,
        'bbox_norm': [0.05, 0.05, 0.15, 0.15],
        'state': 'accepted',
        'score': 1.0,
        'detector': 'human',
        'source': 'human',
    }


class TestFreshDetectionNotVisibleNeverFabricatesARejectedBox:
    """Reproduces the review's exact B1 probe: a fresh `pending_detection`
    item with NO stored boxes, a primary-detector hit, and a VLM reply of
    `region_visible=False`."""

    @pytest.mark.asyncio
    async def test_no_stored_boxes_not_visible_ends_no_region_visible_empty(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_os = _FakeOpenSearch({'c1': _item()}, search_delay=0.0, lag_searches=0)
        reply = VlmCombinedReply(img_id='c1', region_visible=False, region_boxes=[])

        await _drive_worker(
            tmp_path,
            monkeypatch,
            fake_os=fake_os,
            primary=RegionCandidate(bbox_norm=(0.3, 0.6, 0.6, 0.75), score=0.9, source='det'),
            segmenter=None,
            reply=reply,
        )

        doc = fake_os.live['c1']
        # Before the fix: this landed `verify_rejected` with a phantom
        # rejected box (the flag was misread as "Path 1 re-verify").
        assert doc[F.status] == RegionStatus.NO_REGION_VISIBLE.value
        assert doc[F.boxes] == []
        assert doc[F.rejected_count] == 0
        assert doc[F.verified] is False


class TestFreshDetectionReplacesStaleMachineBox:
    """Reproduces the review's exact M1 probe: a requeued item carrying a
    stale rejected MACHINE box from a prior pass; the new pass finds
    nothing (both detector and segmenter miss)."""

    @pytest.mark.asyncio
    async def test_stale_machine_box_dropped_when_fresh_pass_finds_nothing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seeded = {
            **_item(),
            F.boxes: [_stale_machine_box()],
            F.rejected_count: 1,
        }
        fake_os = _FakeOpenSearch({'c1': seeded}, search_delay=0.0, lag_searches=0)

        await _drive_worker(
            tmp_path,
            monkeypatch,
            fake_os=fake_os,
            primary=None,
            segmenter=None,
            reply=_accept(),  # never consulted: no candidate ever reaches the VLM
            visible=True,
        )

        doc = fake_os.live['c1']
        # Before the fix: `merge_boxes_for_write` kept the untouched stale
        # box (its id never appeared in this pass's own empty candidate
        # list), and `derive_status` re-derived `verify_rejected` from it
        # forever -- a verifier verdict that never actually happened this
        # pass.
        assert doc[F.status] == RegionStatus.NO_REGION_BOX.value
        assert doc[F.boxes] == []
        assert doc[F.rejected_count] == 0

    @pytest.mark.asyncio
    async def test_stale_machine_box_replaced_by_a_fresh_accepted_one(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Same stale box, but this time the fresh pass DOES find (and the
        VLM accepts) a new region -- the stale box must not linger
        alongside it either."""
        seeded = {
            **_item(),
            F.boxes: [_stale_machine_box()],
            F.rejected_count: 1,
        }
        fake_os = _FakeOpenSearch({'c1': seeded}, search_delay=0.0, lag_searches=0)

        await _drive_worker(
            tmp_path,
            monkeypatch,
            fake_os=fake_os,
            primary=RegionCandidate(bbox_norm=(0.3, 0.6, 0.6, 0.75), score=0.9, source='det'),
            segmenter=None,
            reply=_accept(),
        )

        doc = fake_os.live['c1']
        box_ids = {b['box_id'] for b in doc[F.boxes]}
        assert 'b_stale' not in box_ids, f'stale machine box survived: {doc[F.boxes]}'
        assert doc[F.status] == RegionStatus.DETECTED.value
        assert doc[F.rejected_count] == 0
        assert any(b['state'] == 'accepted' for b in doc[F.boxes])


class TestFreshDetectionKeepsHumanBoxAndReplacesMachineBox:
    """Both a human-owned box and a stale machine box are stored; a fresh
    detection pass must preserve the former untouched and replace the
    latter."""

    @pytest.mark.asyncio
    async def test_human_box_untouched_machine_box_replaced(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seeded = {
            **_item(),
            F.boxes: [_human_box(), _stale_machine_box()],
            F.count: 1,
            F.rejected_count: 1,
        }
        fake_os = _FakeOpenSearch({'c1': seeded}, search_delay=0.0, lag_searches=0)

        await _drive_worker(
            tmp_path,
            monkeypatch,
            fake_os=fake_os,
            primary=RegionCandidate(bbox_norm=(0.3, 0.6, 0.6, 0.75), score=0.9, source='det'),
            segmenter=None,
            reply=_accept(),
        )

        doc = fake_os.live['c1']
        boxes_by_id = {b['box_id']: b for b in doc[F.boxes]}

        # Human box: untouched (the stored doc round-trips through
        # RegionBox.to_doc/from_doc, which fills in every other field with
        # its None default -- compare only the fields the seed set).
        human = boxes_by_id['b_human']
        seed = _human_box()
        assert {k: human[k] for k in seed} == seed
        # Stale machine box: gone.
        assert 'b_stale' not in boxes_by_id
        # This pass's own fresh candidate: present, accepted.
        fresh_ids = set(boxes_by_id) - {'b_human'}
        assert len(fresh_ids) == 1
        fresh = boxes_by_id[next(iter(fresh_ids))]
        assert fresh['state'] == 'accepted'
        assert fresh['source'] != 'human'
        assert doc[F.status] == RegionStatus.DETECTED.value
