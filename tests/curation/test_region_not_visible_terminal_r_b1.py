"""R-B1 regression: a re-verified box the VLM says isn't visible must
reach a TERMINAL outcome, never stay indefinitely re-queued.

Reproduces the review's exact probe (docs/design/openprocessor_internal/
w8_pipeline_review_2026-09-27.md, "Re-review 2026-09-27" section, finding
R-B1): the B1 fix made Path 1 (``pending_verification``) merge its
resolved verdict back into the stored box list instead of replacing it
outright. But when the VLM answers ``region_visible=False``, the pre-fix
code wrote an EMPTY box list -- and ``merge_boxes_for_write`` never
touches a stored box whose id doesn't appear in the new (empty) list, so
the human's `proposed` box stayed `proposed` forever. Every poll then
made another VLM call and bumped the revision, unbounded. ``fake_vlm.py``
defaults ``region_visible`` to False, so this would livelock on the
first human-proposed box in the real dev/test stack, not just an edge
case.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from src.config import TERMINAL_STATUSES, get_region_fields
from src.config.region_state import RegionStatus
from src.services.labeling.vlm_labeler import VlmCombinedReply

from .test_region_cascade_integrity import _drive_worker, _FakeOpenSearch
from .test_region_pending_verification_b1 import (
    PROPOSED_BBOX,
    SIBLING_BBOX,
    _seed_via_real_put_route,
)


if TYPE_CHECKING:
    from pathlib import Path


pytestmark = pytest.mark.usefixtures('reference_region_profile')

F = get_region_fields()


class TestNotVisibleReachesTerminalStatus:
    @pytest.mark.asyncio
    async def test_region_visible_false_terminates_instead_of_livelocking(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seeded = _seed_via_real_put_route()
        fake_os = _FakeOpenSearch({'c1': seeded}, search_delay=0.0, lag_searches=0)

        reply = VlmCombinedReply(img_id='c1', region_visible=False, region_boxes=[])

        mocks = await _drive_worker(
            tmp_path,
            monkeypatch,
            fake_os=fake_os,
            primary=None,
            segmenter=None,
            reply=reply,
        )

        doc = fake_os.live['c1']

        # Terminal outcome -- never left `proposed`/`pending_verification`.
        assert doc[F.status] in {s.value for s in TERMINAL_STATUSES}
        assert doc[F.status] == RegionStatus.VERIFY_REJECTED.value

        boxes_by_id = {b['box_id']: b for b in doc[F.boxes]}
        # The re-verified box is resolved (rejected), keeping its id --
        # never silently dropped, never left `proposed`.
        assert boxes_by_id['b3']['state'] == 'rejected'
        assert boxes_by_id['b3']['bbox_norm'] == PROPOSED_BBOX
        # Untouched sibling unaffected.
        assert boxes_by_id['b2']['state'] == 'rejected'
        assert boxes_by_id['b2']['bbox_norm'] == SIBLING_BBOX

        # Not re-queued for another VLM call: the item reached a terminal
        # status on the FIRST call, so the drive's tail polling window
        # (well past the write) never triggers a second one.
        assert mocks['vlm'].label_combined_batch.await_count == 1

        # verified is never true on a rejection -- R-M4.
        assert doc[F.verified] is False
