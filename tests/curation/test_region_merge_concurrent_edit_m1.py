"""M1 residual / R-M3 regression: a box a human moves or deletes DURING
its own VLM re-verification call must never be silently reverted or
resurrected by that pass's now-stale merge.

Reproduces the review's exact probes (docs/design/openprocessor_internal/
w8_pipeline_review_2026-09-27.md, "Re-review 2026-09-27" section, finding
R-M3): the B1 fix's ``merge_boxes_for_write`` only guarded the ITEM-level
``region_revision`` (via the status staleness check in
``bulk_writer._merge``), never the specific box a pass's own verdict
targets. A human editing that SAME box while its VLM call is in flight
(the item's status never changes -- it's still `proposed` either way, so
the status guard never fires) had its edit silently overwritten (moved
case) or resurrected (deleted case) by the pass's pre-edit snapshot.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

import pytest

from src.config import get_region_fields
from src.services.labeling.region_overlay import VlmBoxVerdict
from src.services.labeling.vlm_models import VlmCombinedReply

from .test_region_cascade_integrity import _drive_worker, _FakeOpenSearch
from .test_region_pending_verification_b1 import SIBLING_BBOX, _seed_via_real_put_route


if TYPE_CHECKING:
    from pathlib import Path


pytestmark = pytest.mark.usefixtures('reference_region_profile')

F = get_region_fields()

MOVED_BBOX = [0.31, 0.61, 0.62, 0.77]


def _accept_reply() -> VlmCombinedReply:
    return VlmCombinedReply(
        img_id='c1',
        region_visible=True,
        region_boxes=[
            VlmBoxVerdict(box=1, bbox_correct=True, confidence='high', text_reply='DNV20')
        ],
    )


class TestConcurrentMoveDuringVerification:
    @pytest.mark.asyncio
    async def test_human_move_mid_verification_survives_the_stale_merge(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seeded = _seed_via_real_put_route()
        fake_os = _FakeOpenSearch({'c1': seeded}, search_delay=0.0, lag_searches=0)

        calls: list[int] = []

        def combined_side_effect(crops: list[Any], **_kw: Any) -> dict[str, Any]:
            # Only the FIRST VLM call races with a human edit -- later
            # (re-fetch) passes must not keep "re-racing" so the test stays
            # deterministic regardless of how many polls run in the
            # _drive_worker tail window.
            if not calls:
                calls.append(1)
                live_boxes = fake_os.live['c1'][F.boxes]
                for b in live_boxes:
                    if b['box_id'] == 'b3':
                        b['bbox_norm'] = list(MOVED_BBOX)
                fake_os.live['c1'][F.revision] = fake_os.live['c1'].get(F.revision, 1) + 1
                fake_os.searchable = copy.deepcopy(fake_os.live)
            return {c.crop_id: _accept_reply() for c in crops}

        await _drive_worker(
            tmp_path,
            monkeypatch,
            fake_os=fake_os,
            primary=None,
            segmenter=None,
            reply=_accept_reply(),
            combined_side_effect=combined_side_effect,
            until_writes=1,
        )

        # The pass whose verdict was computed against the OLD geometry writes
        # NOTHING (no box state, no verified/verifier, no embedding, no event):
        # the next poll re-verifies the box where the human put it.
        assert fake_os.writes, 'the re-verification of the moved box never wrote'
        for _doc_id, written in fake_os.writes:
            boxes_by_id = {b['box_id']: b for b in written[F.boxes]}
            # R-M3: the human's newer position survives -- never reverted to
            # PROPOSED_BBOX (what the dropped pass's VLM call verified).
            assert boxes_by_id['b3']['bbox_norm'] == MOVED_BBOX
            assert boxes_by_id['b2']['state'] == 'rejected'
            assert boxes_by_id['b2']['bbox_norm'] == SIBLING_BBOX
            # `verified` only describes a verdict on the geometry that is stored.
            assert not written.get(F.verified) or boxes_by_id['b3']['state'] == 'accepted'
        first = {b['box_id']: b for b in fake_os.writes[0][1][F.boxes]}
        assert first['b3']['state'] == 'accepted', 'only the second, fresh verdict may land'


class TestConcurrentDeleteDuringVerification:
    @pytest.mark.asyncio
    async def test_human_delete_mid_verification_stays_deleted(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seeded = _seed_via_real_put_route()
        fake_os = _FakeOpenSearch({'c1': seeded}, search_delay=0.0, lag_searches=0)

        calls: list[int] = []

        def combined_side_effect(crops: list[Any], **_kw: Any) -> dict[str, Any]:
            if not calls:
                calls.append(1)
                live = fake_os.live['c1']
                live[F.boxes] = [b for b in live[F.boxes] if b['box_id'] != 'b3']
                # What the real delete route also does: re-derive the status.
                live[F.status] = 'verify_rejected'
                live[F.revision] = live.get(F.revision, 1) + 1
                fake_os.searchable = copy.deepcopy(fake_os.live)
            return {c.crop_id: _accept_reply() for c in crops}

        await _drive_worker(
            tmp_path,
            monkeypatch,
            fake_os=fake_os,
            primary=None,
            segmenter=None,
            reply=_accept_reply(),
            combined_side_effect=combined_side_effect,
            until_writes=0,
        )

        doc = fake_os.live['c1']
        box_ids = {b['box_id'] for b in doc[F.boxes]}
        # R-M3: a box deleted during its own re-verification call must
        # never be resurrected by that pass's stale merge.
        assert 'b3' not in box_ids, f'deleted box b3 was resurrected: {doc[F.boxes]}'
        assert 'b2' in box_ids
        # The item is terminal (`verify_rejected`) and the pass's verdict for
        # the deleted box was dropped: nothing was written at all.
        assert doc[F.status] == 'verify_rejected'
        assert not fake_os.writes
