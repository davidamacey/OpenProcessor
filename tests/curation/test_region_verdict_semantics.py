"""A missing verifier answer is "no verdict", never a reject (DQ-M10).

The combined VLM reply's ``region_bbox_correct`` decides whether the
candidate box is accepted. Only an explicit ``true`` accepts and only an
explicit ``false`` rejects; ``null``, an absent key, or a null-like string
("null", "none", "") is no verdict at all -- the item stays pending for a
retry instead of landing ``verify_rejected``.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

from src.config import get_region_fields
from src.services.detection.cascade_detect import RegionCandidate
from src.services.labeling.region_overlay import VlmBoxVerdict
from src.services.labeling.vlm_labeler import VlmCombinedReply, _coerce_bool

from .test_region_cascade_integrity import (
    _combined,
    _drive_worker,
    _FakeOpenSearch,
    _item,
    _labeler,
)


if TYPE_CHECKING:
    from pathlib import Path


pytestmark = pytest.mark.usefixtures('reference_region_profile')

BOX = RegionCandidate(bbox_norm=(0.3, 0.6, 0.6, 0.75), score=0.9, source='det')


class TestNullIsNoVerdict:
    @pytest.mark.parametrize('value', [None, 'null', 'None', 'NULL', '', '  ', 'maybe'])
    def test_null_like_values_are_no_verdict(self, value: Any) -> None:
        assert _coerce_bool(value) is None

    @pytest.mark.parametrize(('value', 'expected'), [('false', False), ('no', False), (0, False)])
    def test_explicit_negatives_still_reject(self, value: Any, expected: bool) -> None:
        assert _coerce_bool(value) is expected

    @pytest.mark.asyncio
    @pytest.mark.parametrize('bbox_answer', [None, 'null', 'none', ''])
    async def test_combined_reply_keeps_null_bbox_answer_as_none(self, bbox_answer: Any) -> None:
        reply = await _labeler(_combined(region_bbox_correct=bbox_answer)).label_combined(
            'c1', b'x', region_bboxes_norm=[(0.1, 0.1, 0.5, 0.5)], draw_overlay=False
        )
        assert reply.region_boxes[0].bbox_correct is None

    @pytest.mark.asyncio
    async def test_absent_bbox_answer_is_none(self) -> None:
        entry = _combined()
        del entry['region_boxes'][0]['region_bbox_correct']
        reply = await _labeler(json.dumps(entry)).label_combined(
            'c1', b'x', region_bboxes_norm=[(0.1, 0.1, 0.5, 0.5)], draw_overlay=False
        )
        assert reply.region_boxes[0].bbox_correct is None


def _reply(bbox_correct: bool | None) -> VlmCombinedReply:
    return VlmCombinedReply(
        img_id='c1',
        region_visible=True,
        region_boxes=[
            VlmBoxVerdict(box=1, bbox_correct=bbox_correct, confidence='high', text_reply='DNV20')
        ],
    )


class TestStreamingWorker:
    @pytest.mark.asyncio
    async def test_null_bbox_verdict_leaves_the_item_pending(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Below the no-verdict cap (test_region_no_verdict_cap.py covers it).
        monkeypatch.setenv('OP_REGION_WORKER_MAX_NO_VERDICT_ATTEMPTS', '100000')
        fake_os = _FakeOpenSearch({'c1': _item()}, search_delay=0.0, lag_searches=0)
        mocks = await _drive_worker(
            tmp_path,
            monkeypatch,
            fake_os=fake_os,
            primary=BOX,
            segmenter=None,
            reply=_reply(None),
        )
        F = get_region_fields()
        assert mocks['vlm'].label_combined_batch.await_count >= 1
        assert fake_os.writes == []
        assert fake_os.live['c1'][F.status] == 'pending_detection'

    @pytest.mark.asyncio
    async def test_explicit_false_still_rejects(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_os = _FakeOpenSearch({'c1': _item()}, search_delay=0.0, lag_searches=0)
        await _drive_worker(
            tmp_path,
            monkeypatch,
            fake_os=fake_os,
            primary=BOX,
            segmenter=None,
            reply=_reply(False),
        )
        F = get_region_fields()
        assert fake_os.live['c1'][F.status] == 'verify_rejected'


# test_null_bbox_verdict_writes_nothing (pre-W8 unit test of the deleted
# combined.py::_try_combined_class_region) is superseded by
# TestStreamingWorker.test_null_bbox_verdict_leaves_the_item_pending above
# -- same assertion (a null box verdict writes nothing, the item stays
# pending) proven at the real runner.py pipeline level.
