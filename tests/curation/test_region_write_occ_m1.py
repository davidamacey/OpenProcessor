"""W8 M1 regression: the box-list write must never use a stale
``region_revision`` / ``region_box_seq`` snapshot.

Reproduces the review's exact probe (docs/design/openprocessor_internal/
w8_pipeline_review_2026-09-27.md, finding M1): something else (a human
PUT) bumps the item's stored ``region_revision`` / ``region_box_seq`` and
replaces its box list while the worker's own VLM call is still in
flight. Before the M1 fix, the worker's write built ``region_revision`` /
minted box ids from the pipeline's fetch-time snapshot (revision 0,
box_seq 0) instead of the CURRENT stored values, so the write reset the
revision backwards and reused ids the concurrent write had already
claimed.

W8c (r1 wipe-on-replace fix): a fresh-detection pass now also MERGES onto
whatever is live at write time (``_ItemTask.pending_merge``), rather than
replacing the box list wholesale -- so the concurrently-added box (``b7``)
must survive alongside this pass's own freshly-detected box, not just
avoid an id collision with it.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

import pytest

from src.config import get_region_fields
from src.services.detection.cascade_detect import RegionCandidate

from .test_region_cascade_integrity import _accept, _drive_worker, _FakeOpenSearch, _item


if TYPE_CHECKING:
    from pathlib import Path


pytestmark = pytest.mark.usefixtures('reference_region_profile')

F = get_region_fields()


class TestConcurrentRevisionBumpDuringVlmCall:
    @pytest.mark.asyncio
    async def test_write_advances_from_current_revision_never_resets_and_never_reuses_ids(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_os = _FakeOpenSearch({'c1': _item()}, search_delay=0.0, lag_searches=0)

        def combined_side_effect(crops: list[Any], **_kw: Any) -> dict[str, Any]:
            # Simulate a human PUT landing while this VLM call is in
            # flight -- AFTER the worker's own producer-side fetch (which
            # saw region_revision=0, region_box_seq=0), but BEFORE the
            # worker's own write-time re-read (bulk_writer._merge's mget,
            # which runs once this VLM call returns).
            fake_os.live['c1'][F.revision] = 5
            fake_os.live['c1'][F.box_seq] = 7
            fake_os.live['c1'][F.boxes] = [
                {
                    'box_id': 'b7',
                    'bbox_norm': [0.2, 0.2, 0.3, 0.3],
                    'state': 'accepted',
                    'score': 0.9,
                }
            ]
            fake_os.searchable = copy.deepcopy(fake_os.live)
            return {c.crop_id: _accept() for c in crops}

        await _drive_worker(
            tmp_path,
            monkeypatch,
            fake_os=fake_os,
            primary=RegionCandidate(bbox_norm=(0.3, 0.6, 0.6, 0.75), score=0.9, source='det'),
            segmenter=None,
            reply=_accept(),
            combined_side_effect=combined_side_effect,
        )

        doc = fake_os.live['c1']
        # M1: revision advances from the CURRENT (5) value -- 6, never
        # reset back to 1 against the task's stale fetch-time snapshot.
        assert doc[F.revision] == 6
        # W8c: the concurrently-added box (b7) is a sibling this pass
        # never touched -- merged back in, not silently discarded.
        by_id = {b['box_id']: b for b in doc[F.boxes]}
        assert by_id['b7']['bbox_norm'] == [0.2, 0.2, 0.3, 0.3]
        assert by_id['b7']['state'] == 'accepted'
        # M1: the freshly-written box's id must never collide with
        # anything already claimed under the CURRENT box_seq (7) -- b8,
        # never a low id like b1 minted off the task's stale seq=0
        # snapshot.
        ids = list(by_id)
        assert len(ids) == len(set(ids)), f'duplicate/reused box id in {ids}'
        new_ids = [i for i in ids if i != 'b7']
        assert new_ids == ['b8'], f'expected exactly one fresh box (b8), got {new_ids}'
        assert doc[F.box_seq] >= 8
