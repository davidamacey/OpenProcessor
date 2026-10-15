"""Strict (fail-closed) parsing of the combined VLM reply: region text, accept fields and batch alignment."""

from __future__ import annotations

import json
from typing import Any

import pytest

from curation.region_cascade_helpers import CLASS_NAMES, _combined, _jpeg, _labeler
from src.config import get_region_fields
from src.services.labeling.vlm_labeler import VlmLabeler
from src.services.labeling.vlm_models import CombinedCrop, CombinedParseFailure, RegionCrop
from src.services.labeling.vlm_reply_parse import _align_batch_entries


pytestmark = pytest.mark.usefixtures('reference_region_profile')


# =============================================================================
# 1. region_text + accept decision
# =============================================================================


class TestCombinedSingle:
    @pytest.mark.asyncio
    async def test_region_text_is_the_transcribed_text(self) -> None:
        reply = await _labeler(_combined()).label_combined(
            'c1', _jpeg(), class_names=CLASS_NAMES, region_bboxes_norm=[(0.2, 0.6, 0.5, 0.8)]
        )
        assert reply.region_boxes[0].text_reply == 'DNV20'
        assert reply.region_visible is True
        assert reply.region_boxes[0].bbox_correct is True

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        'echo',
        ['chevycar', 'CHEVYCAR', 'Chevy Car'],
    )
    async def test_class_echo_is_not_region_text(self, echo: str) -> None:
        reply = await _labeler(_combined(region_text=echo)).label_combined(
            'c1', _jpeg(), class_names=CLASS_NAMES, region_bboxes_norm=[(0.2, 0.6, 0.5, 0.8)]
        )
        # box_verdicts' text cleaning (region_overlay._clean_text_reply)
        # now suppresses an item-answer echo (the picked class name) the same way the old flat parser's
        # _clean_combined_region_text did. The class side still resolves
        # regardless.
        assert reply.region_boxes[0].text_reply is None
        assert reply.class_id == 1

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        'sentinel',
        # 'unreadable' / '-' are sentinels too: region_overlay._clean_text_reply
        # must drop all 7 the old flat parser dropped, not just 4.
        [None, '', 'unknown', 'N/A', 'null', 'unreadable', '-'],
    )
    async def test_no_text_is_null(self, sentinel: str | None) -> None:
        reply = await _labeler(_combined(region_text=sentinel)).label_combined(
            'c1', _jpeg(), class_names=CLASS_NAMES, region_bboxes_norm=[(0.2, 0.6, 0.5, 0.8)]
        )
        assert reply.region_boxes[0].text_reply is None

    @pytest.mark.asyncio
    async def test_quoted_false_booleans_do_not_accept(self) -> None:
        reply = await _labeler(
            _combined(region_visible='false', region_bbox_correct='false')
        ).label_combined('c1', _jpeg(), class_names=CLASS_NAMES, region_bboxes_norm=[(0, 0, 1, 1)])
        assert reply.region_visible is False
        assert reply.region_boxes[0].bbox_correct is False

    @pytest.mark.asyncio
    async def test_unrecognized_bbox_answer_is_not_an_accept(self) -> None:
        reply = await _labeler(_combined(region_bbox_correct='maybe')).label_combined(
            'c1', _jpeg(), class_names=CLASS_NAMES, region_bboxes_norm=[(0, 0, 1, 1)]
        )
        assert reply.region_boxes[0].bbox_correct is None

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        'content',
        [
            '',
            json.dumps({k: v for k, v in _combined().items() if k != 'region_visible'}),
            json.dumps(_combined(region_visible='perhaps')),
        ],
    )
    async def test_missing_or_garbled_visible_answer_is_a_parse_failure(self, content: str) -> None:
        with pytest.raises(CombinedParseFailure):
            await _labeler(content).label_combined(
                'c1', _jpeg(), class_names=CLASS_NAMES, region_bboxes_norm=[(0, 0, 1, 1)]
            )


class TestCombinedBatch:
    def _crops(self, n: int) -> list[CombinedCrop]:
        return [
            CombinedCrop(
                crop_id=f'c{i}', jpeg_bytes=b'', region_bboxes_norm=[(0, 0, 1, 1)], classify=True
            )
            for i in range(1, n + 1)
        ]

    def _parse(self, entries: list[dict[str, Any]], n: int) -> dict[str, Any]:
        raw = json.dumps({'results': entries})
        return VlmLabeler._parse_combined_batch_response(
            raw, self._crops(n), get_region_fields(), class_names=CLASS_NAMES
        )

    def test_each_crop_gets_its_own_text(self) -> None:
        out = self._parse(
            [
                {'img': 2, **_combined(region_text='XYZ789')},
                {'img': 1, **_combined(region_text='DNV20')},
                {'img': 3, **_combined(region_text=None)},
            ],
            3,
        )
        assert out['c1'].region_boxes[0].text_reply == 'DNV20'
        assert out['c2'].region_boxes[0].text_reply == 'XYZ789'
        assert out['c3'].region_boxes[0].text_reply is None

    def test_zero_based_indices_are_shifted_not_misassigned(self) -> None:
        out = self._parse(
            [
                {'img': 0, **_combined(region_text='AAA111', region_bbox_correct=True)},
                {'img': 1, **_combined(region_text='BBB222', region_bbox_correct=False)},
            ],
            2,
        )
        assert out['c1'].region_boxes[0].text_reply == 'AAA111'
        assert out['c1'].region_boxes[0].bbox_correct is True
        assert out['c2'].region_boxes[0].bbox_correct is False

    @pytest.mark.parametrize(
        'entries',
        [
            # duplicate index
            [{'img': 1, **_combined()}, {'img': 1, **_combined()}],
            # out of range
            [{'img': 1, **_combined()}, {'img': 7, **_combined()}],
            # indexed + unindexed mix
            [{'img': 1, **_combined()}, _combined()],
            # unindexed, wrong count
            [_combined()],
        ],
    )
    def test_untrustworthy_alignment_fails_the_whole_chunk_closed(
        self, entries: list[dict[str, Any]]
    ) -> None:
        out = self._parse(entries, 2)
        assert out == {'c1': None, 'c2': None}

    def test_missing_entry_is_none_not_a_neighbours_verdict(self) -> None:
        out = self._parse([{'img': 2, **_combined()}], 2)
        assert out['c1'] is None
        assert out['c2'] is not None

    def test_entry_without_visible_answer_is_none(self) -> None:
        bad = {k: v for k, v in _combined().items() if k != 'region_visible'}
        out = self._parse([{'img': 1, **bad}, {'img': 2, **_combined()}], 2)
        assert out['c1'] is None
        assert out['c2'].region_visible is True


class TestStandaloneVerify:
    def test_single_text_comes_from_text_not_reason(self) -> None:
        raw = json.dumps(
            {
                'is_region': True,
                'confidence': 'high',
                'reason': 'a red taillight next to it',
                'text': 'DNV20',
                'text_confidence': 'high',
            }
        )
        v = VlmLabeler._parse_region_response(raw, RegionCrop(crop_id='c1', jpeg_bytes=b''))
        assert v is not None
        assert v.is_region is True
        assert v.text == 'DNV20'

    def test_single_quoted_false_is_a_reject(self) -> None:
        raw = json.dumps({'is_region': 'false', 'confidence': 'high', 'text': 'X'})
        v = VlmLabeler._parse_region_response(raw, RegionCrop(crop_id='c1', jpeg_bytes=b''))
        assert v is not None
        assert v.is_region is False
        assert v.text is None

    def test_batch_misaligned_yields_no_verdicts_not_rejects(self) -> None:
        crops = [RegionCrop(crop_id=f'c{i}', jpeg_bytes=b'') for i in (1, 2)]
        raw = json.dumps(
            [
                {'img': 1, 'is_region': True, 'confidence': 'high', 'text': 'A1'},
                {'img': 1, 'is_region': True, 'confidence': 'high', 'text': 'B2'},
            ]
        )
        out = VlmLabeler._parse_region_batch_response(raw, crops)
        assert out == []

    def test_batch_missing_entry_is_omitted_not_a_reject(self) -> None:
        crops = [RegionCrop(crop_id=f'c{i}', jpeg_bytes=b'') for i in (1, 2)]
        raw = json.dumps([{'img': 2, 'is_region': True, 'confidence': 'high', 'text': 'B2'}])
        out = VlmLabeler._parse_region_batch_response(raw, crops)
        by_id = {v.crop_id: v for v in out}
        assert 'c1' not in by_id
        assert by_id['c2'].is_region is True

    def test_batch_empty_reply_yields_no_verdicts(self) -> None:
        crops = [RegionCrop(crop_id=f'c{i}', jpeg_bytes=b'') for i in (1, 2)]
        assert VlmLabeler._parse_region_batch_response('', crops) == []

    def test_batch_aligned_by_index(self) -> None:
        crops = [RegionCrop(crop_id=f'c{i}', jpeg_bytes=b'') for i in (1, 2)]
        raw = json.dumps(
            [
                {'img': 2, 'is_region': False, 'confidence': 'high', 'text': None},
                {'img': 1, 'is_region': True, 'confidence': 'high', 'text': 'A1'},
            ]
        )
        out = VlmLabeler._parse_region_batch_response(raw, crops)
        assert (out[0].crop_id, out[0].is_region, out[0].text) == ('c1', True, 'A1')
        assert (out[1].crop_id, out[1].is_region) == ('c2', False)


class TestAlignBatchEntries:
    def test_positional_only_when_counts_match(self) -> None:
        a, b = {'x': 1}, {'x': 2}
        assert _align_batch_entries([a, b], 2) == [a, b]
        assert _align_batch_entries([a], 2) is None

    def test_bool_index_rejected(self) -> None:
        assert _align_batch_entries([{'img': True}], 1) is None
