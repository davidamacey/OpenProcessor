"""R-M2 regression: the no-VLM Path 1 accept must reuse the STORED box's
own id, never mint a duplicate sibling for the same geometry.

Reproduces the review's exact probe (docs/design/openprocessor_internal/
w8_pipeline_review_2026-09-27.md, "Re-review 2026-09-27" section, finding
R-M2): with no VLM configured, ``accept_without_vlm`` always minted
``new_box_placeholder(0)`` for its box, even when the candidate came from
a stored `proposed` box (``cand.box_id`` set by ``_task_box_from_stored``,
W8 B1). In merge mode the stored `proposed` box was never touched (its id
never appears in the fresh placeholder-based write) and the placeholder
was appended as a brand-new sibling -- leaving BOTH the original
`proposed` box (forever unresolved) AND a duplicate `accepted` box with
the exact same geometry.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from src.config import get_region_fields

from .test_region_cascade_integrity import _FakeOpenSearch
from .test_region_pending_verification_b1 import (
    PROPOSED_BBOX,
    SIBLING_BBOX,
    _seed_via_real_put_route,
)
from .test_region_text_worker import _drive


if TYPE_CHECKING:
    from pathlib import Path


pytestmark = pytest.mark.usefixtures('reference_region_profile')

F = get_region_fields()


class TestNoVlmPathReusesStoredBoxId:
    @pytest.mark.asyncio
    async def test_accept_without_vlm_reuses_the_stored_proposed_box_id(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seeded = _seed_via_real_put_route()
        fake_os = _FakeOpenSearch({'c1': seeded}, search_delay=0.0, lag_searches=0)

        await _drive(
            tmp_path,
            monkeypatch,
            fake_os=fake_os,
            primary=None,
            segmenter=None,
            vlm_url='',
        )

        doc = fake_os.live['c1']
        assert doc[F.status] == 'detected'
        boxes = doc[F.boxes]
        # R-M2: exactly the untouched sibling (b2) plus the re-verified
        # box REUSING its stored id (b3) -- never a THIRD, duplicate box
        # for the same geometry under a fresh placeholder id.
        assert len(boxes) == 2, f'expected 2 boxes (b2, b3), got {boxes}'
        boxes_by_id = {b['box_id']: b for b in boxes}
        assert set(boxes_by_id) == {'b2', 'b3'}
        assert boxes_by_id['b3']['state'] == 'accepted'
        assert boxes_by_id['b3']['bbox_norm'] == PROPOSED_BBOX
        assert boxes_by_id['b2']['state'] == 'rejected'
        assert boxes_by_id['b2']['bbox_norm'] == SIBLING_BBOX
