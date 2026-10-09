"""W8 M3 + M8 regressions: multi-candidate (N>1) pipeline coverage.

The review's mutation testing found the existing suite could not tell
N=1 from N>1 through the real streaming runner -- ``_drive_worker`` /
``_drive`` only ever fed one candidate per leg, so mutations like
forcing the selection cap to 1, misaligning verdicts with candidates, or
resetting the box-list revision at every write all survived the FULL
test suite. This module drives the real runner with 2-3 raw candidates
per leg and asserts PER-BOX outcomes distinctly.

Also covers the per-box embeddings: every ACCEPTED box gets its own vector
in ``region_box_embeddings`` (keyed by its final id and the geometry it was
computed from), never a rejected one.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

import pytest

from src.config import get_region_fields
from src.services.detection.cascade_detect import RegionCandidate, crop_norm_to_source_norm
from src.services.labeling.region_overlay import VlmBoxVerdict
from src.services.labeling.vlm_models import VlmCombinedReply

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
        source_a = tuple(crop_norm_to_source_norm(BOX_A, _item()['bbox_norm']))
        source_b = tuple(crop_norm_to_source_norm(BOX_B, _item()['bbox_norm']))
        source_c = tuple(crop_norm_to_source_norm(BOX_C, _item()['bbox_norm']))
        assert boxes[source_a]['state'] == 'accepted'
        assert boxes[source_a]['text'] == 'DNV20'
        assert boxes[source_b]['state'] == 'rejected'
        assert boxes[source_c]['state'] == 'rejected'


class TestPerBoxEmbeddings:
    @pytest.mark.asyncio
    async def test_embedding_and_region_verified_event_are_written_end_to_end(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """R-M1 fix gate (2026-09-27 re-review): assert the real written
        output, not an internal helper -- moving status-writing into
        `bulk_writer._merge` once silently stopped the embed stage and
        `bulk_writer._publish_region_events` from seeing a `DETECTED`
        status, so no embedding and no `crop.region_verified` event were
        produced for any output.

        Drives the real runner with the embed stage genuinely enabled
        (``region_embed_ready=True`` + a stubbed ``PEEncoder``) and event
        publishing genuinely enabled (a stubbed event client), then
        asserts directly on the item doc's ``region_box_embeddings`` and
        the published event body: ONLY the accepted box (the lower-scored
        second candidate here; the first is rejected) carries a vector,
        keyed by its final box id and the geometry it was cropped from.
        """
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

        import numpy as np

        import scripts.curation.worker.region_embed_stage as region_embed_stage_mod
        import src.clients.pe_encoder as pe_encoder_mod
        from scripts.curation.worker import bulk_writer as bulk_writer_mod

        class _FakePE:
            async def embed_crops(self, crops: list[Any], max_batch: int = 32) -> Any:  # noqa: ARG002
                return np.tile(np.array([0.0, 1.0, 0.0], dtype=np.float32), (len(crops), 1))

        monkeypatch.setattr(pe_encoder_mod, 'PEEncoder', MagicMock(return_value=_FakePE()))

        # Which crop-frame bbox the embed stage actually cropped.
        crop_calls: list[tuple[float, float, float, float]] = []
        real_crop_region_jpeg = region_embed_stage_mod._crop_region_jpeg

        def _recording_crop(jpeg: bytes, region_in_crop: Any) -> bytes:
            crop_calls.append(tuple(region_in_crop))
            return real_crop_region_jpeg(jpeg, region_in_crop)

        monkeypatch.setattr(region_embed_stage_mod, '_crop_region_jpeg', _recording_crop)

        published: list[dict[str, Any]] = []

        class _FakeEventClient:
            async def post(self, url: str, json: Any = None, timeout: Any = None) -> None:  # noqa: ARG002
                published.append({'url': url, 'body': json})

        monkeypatch.setattr(bulk_writer_mod, '_EVENT_API_URL', 'http://fake-event-api')
        monkeypatch.setattr(bulk_writer_mod, '_EVENT_CLIENT', _FakeEventClient())

        await _drive_worker(
            tmp_path,
            monkeypatch,
            fake_os=fake_os,
            primary=[rejected_cand, accepted_cand],
            segmenter=None,
            reply=reply,
            profile_overrides=MULTI_BOX_PROFILE,
            region_embed_ready=True,
        )

        doc = fake_os.live['c1']
        assert doc[F.status] == 'detected'

        # The real written per-box embedding -- absent entirely under R-M1.
        assert doc.get(F.box_embeddings), 'no box embedding was written (R-M1 regression)'
        by_geometry = {tuple(b['bbox_norm']): b for b in doc[F.boxes]}
        source_b = tuple(crop_norm_to_source_norm(BOX_B, _item()['bbox_norm']))
        accepted_box = by_geometry[source_b]
        assert accepted_box['state'] == 'accepted'
        (entry,) = doc[F.box_embeddings]
        assert entry['box_id'] == accepted_box['box_id']
        assert entry['bbox_norm'] == pytest.approx(list(source_b))
        assert list(entry['embedding'][:3]) == pytest.approx([0.0, 1.0, 0.0])

        # Embedded from the ACCEPTED box's crop only: the rejected
        # (higher-scored) candidate is never cropped for an embedding.
        assert crop_calls == [pytest.approx(BOX_B)]

        # The real published event -- absent entirely under R-M1.
        assert published, 'crop.region_verified event was never published (R-M1 regression)'
        assert published[-1]['body']['type'] == 'crop.region_verified'
        assert published[-1]['body']['crop_id'] == 'c1'
