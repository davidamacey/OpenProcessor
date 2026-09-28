"""W8 M3 + M8 regressions: multi-candidate (N>1) pipeline coverage.

The review's mutation testing found the existing suite could not tell
N=1 from N>1 through the real streaming runner -- ``_drive_worker`` /
``_drive`` only ever fed one candidate per leg, so mutations like
forcing the selection cap to 1, misaligning verdicts with candidates, or
resetting the box-list revision at every write all survived the FULL
test suite. This module drives the real runner with 2-3 raw candidates
per leg and asserts PER-BOX outcomes distinctly.

Also covers M3: the region-embedding source must come from an ACCEPTED
box, never the top-scored (but possibly-rejected) one.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from scripts.curation.worker import runner as runner_mod
from scripts.curation.worker.state import _ItemTask
from scripts.curation.worker.verify import TaskBoxInput
from src.config import get_region_fields
from src.services.curation.region_boxes import RegionBox
from src.services.detection.cascade_detect import RegionCandidate
from src.services.labeling.region_overlay import VlmBoxVerdict
from src.services.labeling.vlm_labeler import VlmCombinedReply

from .test_region_cascade_integrity import _drive_worker, _FakeOpenSearch, _item, _profile


if TYPE_CHECKING:
    from pathlib import Path


pytestmark = pytest.mark.usefixtures('reference_region_profile')

F = get_region_fields()

MULTI_BOX_PROFILE = {'max_regions_per_item': 3}

# Three non-overlapping crop-frame boxes so class-agnostic NMS never
# merges them away -- the selection cap (max_regions_per_item=3) is the
# only thing gating how many survive.
BOX_A = (0.05, 0.05, 0.15, 0.15)
BOX_B = (0.40, 0.40, 0.55, 0.55)
BOX_C = (0.70, 0.70, 0.85, 0.85)


def _mixed_reply() -> VlmCombinedReply:
    """Box 1 rejected, boxes 2-3 accepted -- one candidate of each verdict
    kind, aligned by position."""
    return VlmCombinedReply(
        img_id='c1',
        region_visible=True,
        region_boxes=[
            VlmBoxVerdict(box=1, bbox_correct=False, confidence='high', text_reply=None),
            VlmBoxVerdict(box=2, bbox_correct=True, confidence='high', text_reply='DNV20'),
            VlmBoxVerdict(box=3, bbox_correct=True, confidence='medium', text_reply='XYZ99'),
        ],
    )


class TestDetectorLegMultiCandidate:
    @pytest.mark.asyncio
    async def test_three_raw_candidates_survive_selection_and_write_distinct_outcomes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_os = _FakeOpenSearch({'c1': _item()}, search_delay=0.0, lag_searches=0)
        candidates = [
            RegionCandidate(bbox_norm=BOX_A, score=0.9, source='det'),
            RegionCandidate(bbox_norm=BOX_B, score=0.8, source='det'),
            RegionCandidate(bbox_norm=BOX_C, score=0.7, source='det'),
        ]
        await _drive_worker(
            tmp_path,
            monkeypatch,
            fake_os=fake_os,
            primary=candidates,
            segmenter=None,
            reply=_mixed_reply(),
            profile_overrides=MULTI_BOX_PROFILE,
        )
        doc = fake_os.live['c1']
        boxes = doc[F.boxes]
        # Mutation 1 (cap forced to 1): this alone kills it -- all 3
        # raw candidates must survive selection, not just the top-scored.
        assert len(boxes) == 3
        # Mutation 2 (verdicts misaligned with candidates): each box's
        # own state/text must match ITS OWN verdict, not a neighbour's.
        assert boxes[0]['state'] == 'rejected'
        assert boxes[0]['rejection_reason'] == 'region_visible_elsewhere'
        assert boxes[1]['state'] == 'accepted'
        assert boxes[1]['text'] == 'DNV20'
        assert boxes[2]['state'] == 'accepted'
        assert boxes[2]['text'] == 'XYZ99'
        # Distinct ids, scores preserved in selection (score) order.
        assert [b['box_id'] for b in boxes] == ['b1', 'b2', 'b3']
        assert [b['score'] for b in boxes] == [0.9, 0.8, 0.7]
        assert doc[F.count] == 2
        assert doc[F.rejected_count] == 1
        # Mutation 4 (current_src={} / revision reset every write): a
        # fresh item's first write is revision 1, box_seq 3 -- not reset.
        assert doc[F.revision] == 1
        assert doc[F.box_seq] == 3
        assert doc[F.status] == 'detected'


class TestSegmenterLegMultiCandidate:
    @pytest.mark.asyncio
    async def test_three_raw_candidates_via_segmenter_leg(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_os = _FakeOpenSearch({'c1': _item()}, search_delay=0.0, lag_searches=0)
        # Top candidate's score (0.97) is ABOVE the skip-verify threshold
        # (_SKIP_VLM_VERIFY_SECONDARY_SCORE, default 0.95) on purpose --
        # with N=1 this score alone would bypass the VLM entirely
        # (skip_vlm_verify). Mutation 5 (skip-verify guard `len==1`
        # removed) would let that auto-skip fire here too; the guard
        # must keep it gated to the VLM whenever N>1.
        candidates = [
            RegionCandidate(bbox_norm=BOX_A, score=0.97, source='seg'),
            RegionCandidate(bbox_norm=BOX_B, score=0.8, source='seg'),
            RegionCandidate(bbox_norm=BOX_C, score=0.7, source='seg'),
        ]
        mocks = await _drive_worker(
            tmp_path,
            monkeypatch,
            fake_os=fake_os,
            primary=None,
            segmenter=candidates,
            reply=_mixed_reply(),
            profile_overrides=MULTI_BOX_PROFILE,
        )
        doc = fake_os.live['c1']
        boxes = doc[F.boxes]
        # Mutation 5 (skip-verify guard `len==1` removed): with N=3
        # candidates the VLM must still be the adjudicator -- the
        # high-confidence auto-skip never applies to a multi-candidate
        # set, so the combined VLM call must have actually run, and
        # 'skip_vlm_verify' must never appear in the chain.
        assert mocks['vlm'].label_combined_batch.await_count >= 1
        assert not any('skip_vlm_verify' in e for e in doc[F.detector_chain])
        assert len(boxes) == 3
        assert boxes[0]['state'] == 'rejected'
        assert boxes[1]['state'] == 'accepted'
        assert boxes[2]['state'] == 'accepted'
        assert all(b['detector'] == _profile().segmenter_name for b in boxes)
        assert doc[F.count] == 2
        assert doc[F.rejected_count] == 1


class TestCombinedVerdictAlignment:
    @pytest.mark.asyncio
    async def test_verdicts_align_by_position_not_reversed_or_shifted(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Mutation 2 dedicated check: swap the verdict order and every
        box's identity (bbox + text) must still track ITS candidate."""
        fake_os = _FakeOpenSearch({'c1': _item()}, search_delay=0.0, lag_searches=0)
        candidates = [
            RegionCandidate(bbox_norm=BOX_A, score=0.9, source='det'),
            RegionCandidate(bbox_norm=BOX_B, score=0.8, source='det'),
            RegionCandidate(bbox_norm=BOX_C, score=0.7, source='det'),
        ]
        # Only box 1 (BOX_A) is accepted; 2 and 3 rejected -- the
        # opposite pattern from the other tests in this module, so a
        # reversed/shifted alignment bug would flip which bbox ends up
        # 'accepted'.
        reply = VlmCombinedReply(
            img_id='c1',
            region_visible=True,
            region_boxes=[
                VlmBoxVerdict(box=1, bbox_correct=True, confidence='high', text_reply='DNV20'),
                VlmBoxVerdict(box=2, bbox_correct=False, confidence='high'),
                VlmBoxVerdict(box=3, bbox_correct=False, confidence='high'),
            ],
        )
        await _drive_worker(
            tmp_path,
            monkeypatch,
            fake_os=fake_os,
            primary=candidates,
            segmenter=None,
            reply=reply,
            profile_overrides=MULTI_BOX_PROFILE,
        )
        doc = fake_os.live['c1']
        boxes = {tuple(b['bbox_norm']): b for b in doc[F.boxes]}
        source_a = tuple(runner_mod.crop_norm_to_source_norm(BOX_A, _item()['bbox_norm']))
        source_b = tuple(runner_mod.crop_norm_to_source_norm(BOX_B, _item()['bbox_norm']))
        source_c = tuple(runner_mod.crop_norm_to_source_norm(BOX_C, _item()['bbox_norm']))
        assert boxes[source_a]['state'] == 'accepted'
        assert boxes[source_a]['text'] == 'DNV20'
        assert boxes[source_b]['state'] == 'rejected'
        assert boxes[source_c]['state'] == 'rejected'


class TestM3EmbeddingSourceIsAcceptedBox:
    def test_sync_accepted_candidate_prefers_accepted_over_top_scored(self) -> None:
        """Unit-level: `_sync_accepted_candidate` is the exact function
        the M3 fix added. `t.candidates[0]` (score 0.9, the top-scored
        raw candidate) is REJECTED; `t.candidates[1]` (score 0.7) is
        ACCEPTED. The region-embedding source (and the legacy singular
        candidate_* mirror) must point at the accepted one."""
        t = _ItemTask(
            crop_id='c1',
            image_path='',
            item_bbox_norm=(0.0, 0.0, 1.0, 1.0),
            region_status='pending_detection',
            class_name='',
        )
        rejected = TaskBoxInput(
            bbox_in_crop=(0.1, 0.1, 0.2, 0.2),
            bbox_in_source=(0.1, 0.1, 0.2, 0.2),
            score=0.9,
            detector='det',
            detector_version='1',
            source='det',
        )
        accepted = TaskBoxInput(
            bbox_in_crop=(0.5, 0.5, 0.6, 0.6),
            bbox_in_source=(0.5, 0.5, 0.6, 0.6),
            score=0.7,
            detector='det',
            detector_version='1',
            source='det',
        )
        t.candidates = [rejected, accepted]
        # _sync_singular_candidate (the pre-M3 behaviour) would mirror
        # `rejected` here -- the bug this fix targets.
        runner_mod._sync_singular_candidate(t)
        assert t.candidate_in_crop == rejected.bbox_in_crop

        boxes = [
            RegionBox(box_id='b1', bbox_norm=rejected.bbox_in_source, state='rejected'),
            RegionBox(box_id='b2', bbox_norm=accepted.bbox_in_source, state='accepted'),
        ]
        runner_mod._sync_accepted_candidate(t, boxes)
        assert t.candidate_in_crop == accepted.bbox_in_crop
        assert t.candidate_in_source == accepted.bbox_in_source
        assert t.candidate_score == accepted.score
        assert t.candidate_source == accepted.source

    @pytest.mark.asyncio
    async def test_embedding_eligibility_source_is_the_accepted_box_end_to_end(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """End-to-end: drive the real runner with one rejected + one
        accepted candidate and confirm the item's `candidate_in_crop`
        mirror -- what `embed_written_regions`'s eligibility filter and
        crop-extraction both key off -- lands on the ACCEPTED box, by
        capturing every `_sync_accepted_candidate` call the pipeline
        makes."""
        fake_os = _FakeOpenSearch({'c1': _item()}, search_delay=0.0, lag_searches=0)
        rejected_cand = RegionCandidate(bbox_norm=BOX_A, score=0.9, source='det')
        accepted_cand = RegionCandidate(bbox_norm=BOX_B, score=0.7, source='det')
        reply = VlmCombinedReply(
            img_id='c1',
            region_visible=True,
            region_boxes=[
                VlmBoxVerdict(box=1, bbox_correct=False, confidence='high'),
                VlmBoxVerdict(box=2, bbox_correct=True, confidence='high'),
            ],
        )
        calls: list[tuple[float, float, float, float] | None] = []
        real_sync = runner_mod._sync_accepted_candidate

        def _spy(t: Any, boxes: Any) -> None:
            real_sync(t, boxes)
            calls.append(t.candidate_in_crop)

        monkeypatch.setattr(runner_mod, '_sync_accepted_candidate', _spy)

        await _drive_worker(
            tmp_path,
            monkeypatch,
            fake_os=fake_os,
            primary=[rejected_cand, accepted_cand],
            segmenter=None,
            reply=reply,
            profile_overrides=MULTI_BOX_PROFILE,
        )

        assert calls, 'the write path never re-synced the embedding source'
        # The accepted candidate's crop-frame bbox, never the rejected
        # (higher-scored, candidates[0]) one's.
        assert calls[-1] == BOX_B
        assert calls[-1] != BOX_A
        doc = fake_os.live['c1']
        assert doc[F.status] == 'detected'
