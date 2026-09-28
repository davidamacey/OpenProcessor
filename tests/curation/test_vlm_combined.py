"""Tests for ``VlmLabeler.label_combined``.

Mocks the httpx client; verifies JSON-mode
flag, parse success, and that
parse failures raise :class:`CombinedParseFailure` so the caller can fall
back to the separate-call paths.

The upstream VLM's own wire-reply keys (what the reply's JSON object is
keyed by) are read via ``RegionFields`` in ``vlm_labeler.py`` rather than
hardcoded — these fixtures use the generic ``RegionFields``
defaults (``region_visible``, ``region_boxes`` -- W8 list shape, one
verdict object per candidate box: ``box`` (1-based), ``region_bbox_correct``,
``region_text``, ``region_confidence``) so the tests stay domain-neutral.

Single-box overlay-drawing coverage (numbered tag rendering) lives in
``tests/curation/test_region_overlay.py`` -- this file only exercises
``VlmLabeler``'s own request/response wiring.
"""

from __future__ import annotations

import io
import json
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from PIL import Image

from src.services.labeling.vlm_labeler import (
    CombinedCrop,
    CombinedParseFailure,
    CombinedTransportError,
    VlmCombinedReply,
    VlmLabeler,
)


def _make_labeler(http_response_json: dict | list | str | None = None) -> VlmLabeler:
    """Build a labeler whose httpx client returns a canned chat-completion."""
    fake = MagicMock(spec=httpx.AsyncClient)
    content = (
        http_response_json
        if isinstance(http_response_json, str)
        else json.dumps(http_response_json)
    )
    resp = MagicMock()
    resp.status_code = 200
    resp.json = MagicMock(
        return_value={
            'choices': [{'message': {'content': content}}],
        }
    )
    resp.raise_for_status = MagicMock()
    fake.post = AsyncMock(return_value=resp)
    return VlmLabeler(client=fake)


def _make_jpeg(size: tuple[int, int] = (200, 100)) -> bytes:
    buf = io.BytesIO()
    Image.new('RGB', size, color=(10, 20, 30)).save(buf, format='JPEG', quality=85)
    return buf.getvalue()


class TestLabelCombined:
    @pytest.mark.asyncio
    async def test_parses_full_reply(self) -> None:
        labeler = _make_labeler(
            {
                'class_id': 12,
                'class_confidence': 'high',
                'region_visible': True,
                'region_boxes': [
                    {
                        'box': 1,
                        'region_bbox_correct': True,
                        'region_text': 'ABC1234',
                        'region_confidence': 'medium',
                    }
                ],
            }
        )
        reply = await labeler.label_combined(
            'crop-1',
            _make_jpeg(),
            class_names=['widget', 'gadget'],
            region_bboxes_norm=[(0.1, 0.7, 0.9, 0.95)],
        )
        assert isinstance(reply, VlmCombinedReply)
        assert reply.img_id == 'crop-1'
        assert reply.class_id == 12
        assert reply.class_confidence == 'high'
        assert reply.region_visible is True
        assert len(reply.region_boxes) == 1
        assert reply.region_boxes[0].bbox_correct is True
        assert reply.region_boxes[0].text_reply == 'ABC1234'
        assert reply.region_boxes[0].confidence == 'medium'
        # JSON-mode flag must be set so the upstream server grammar-
        # constrains output.
        sent = labeler._client.post.await_args.kwargs['json']
        assert sent['response_format'] == {'type': 'json_object'}

    @pytest.mark.asyncio
    async def test_trusted_class_cohort_skips_class(self) -> None:
        labeler = _make_labeler(
            {
                'class_id': None,
                'region_visible': True,
                'region_boxes': [
                    {'box': 1, 'region_bbox_correct': False, 'region_text': None},
                ],
            }
        )
        reply = await labeler.label_combined(
            'crop-2',
            _make_jpeg(),
            class_names=None,
            region_bboxes_norm=[(0.1, 0.7, 0.9, 0.95)],
        )
        assert reply.class_id is None
        assert reply.region_visible is True
        assert reply.region_boxes[0].bbox_correct is False
        assert reply.region_boxes[0].text_reply is None

    @pytest.mark.asyncio
    async def test_invalid_json_raises_combined_parse_failure(self) -> None:
        labeler = _make_labeler('this is not json at all')
        with pytest.raises(CombinedParseFailure):
            await labeler.label_combined('crop-3', _make_jpeg(), class_names=['widget'])

    @pytest.mark.asyncio
    async def test_non_object_response_raises_combined_parse_failure(self) -> None:
        labeler = _make_labeler('[1, 2, 3]')
        with pytest.raises(CombinedParseFailure):
            await labeler.label_combined('crop-4', _make_jpeg(), class_names=['widget'])

    @pytest.mark.asyncio
    async def test_http_error_raises_combined_parse_failure(self) -> None:
        labeler = _make_labeler({})
        labeler._client.post = AsyncMock(side_effect=httpx.ConnectError('boom'))
        with pytest.raises(CombinedParseFailure):
            await labeler.label_combined('crop-5', _make_jpeg(), class_names=['widget'])

    @pytest.mark.asyncio
    async def test_plate_text_truncated_to_32(self) -> None:
        labeler = _make_labeler(
            {
                'class_id': 1,
                'region_visible': True,
                'region_boxes': [{'box': 1, 'region_text': 'A' * 100}],
            }
        )
        reply = await labeler.label_combined(
            'crop-6',
            _make_jpeg(),
            class_names=['widget'],
            region_bboxes_norm=[(0.1, 0.7, 0.9, 0.95)],
        )
        assert reply.region_boxes[0].text_reply is not None
        assert len(reply.region_boxes[0].text_reply) == 32

    @pytest.mark.asyncio
    async def test_no_candidate_boxes_skips_box_verdict_parsing(self) -> None:
        """N=0: no ``region_boxes`` key required in the reply at all."""
        labeler = _make_labeler({'class_id': 0, 'region_visible': False})
        reply = await labeler.label_combined('crop-7', _make_jpeg(), class_names=['widget'])
        assert reply.region_boxes == []

    @pytest.mark.asyncio
    async def test_missing_region_boxes_key_raises_when_candidates_offered(self) -> None:
        """D-B: N>=1 candidates offered but the reply lacks the list key
        is a hard parse failure -- no flat-shape fallback."""
        labeler = _make_labeler({'class_id': 0, 'region_visible': True})
        with pytest.raises(CombinedParseFailure):
            await labeler.label_combined(
                'crop-8',
                _make_jpeg(),
                class_names=['widget'],
                region_bboxes_norm=[(0.1, 0.7, 0.9, 0.95)],
            )


class TestLabelCombinedBatch:
    """Tests for the batched variant that collapses region-visibility +
    region-verify into ONE upstream call per chunk.

    The batched method is the critical prerequisite for a high-
    throughput cascade worker: swapping the per-crop ``label_combined``
    into a batch stage without this batching would multiply upstream
    traffic proportionally to the chunk size. These tests guard against
    regressions in the chunking, multi-image payload shape, per-entry
    parse, and sentinel-on-failure contract.
    """

    @pytest.mark.asyncio
    async def test_returns_one_reply_per_input_crop(self) -> None:
        labeler = _make_labeler(
            [
                {
                    'img': 1,
                    'class_id': 0,
                    'class_confidence': 'high',
                    'region_visible': True,
                    'region_boxes': [
                        {
                            'box': 1,
                            'region_bbox_correct': True,
                            'region_text': 'ABC123',
                            'region_confidence': 'high',
                        }
                    ],
                    'make': 'Ford',
                    'model': 'F150',
                },
                {
                    'img': 2,
                    'class_id': 1,
                    'class_confidence': 'medium',
                    'region_visible': True,
                    'region_boxes': [
                        {'box': 1, 'region_bbox_correct': False, 'region_text': None},
                    ],
                    'make': '',
                    'model': '',
                },
                {
                    'img': 3,
                    'class_id': None,
                    'region_visible': False,
                    'region_boxes': [{'box': 1, 'region_bbox_correct': None}],
                    'make': '',
                    'model': '',
                },
            ]
        )
        crops = [
            CombinedCrop(
                crop_id=f'c{i}',
                jpeg_bytes=_make_jpeg(),
                region_bboxes_norm=[(0.1, 0.7, 0.9, 0.95)],
            )
            for i in range(1, 4)
        ]
        out = await labeler.label_combined_batch(crops, class_names=['widget', 'gadget'])
        assert set(out.keys()) == {'c1', 'c2', 'c3'}
        # Happy path: detected.
        assert out['c1'] is not None
        assert out['c1'].region_boxes[0].bbox_correct is True
        assert out['c1'].region_boxes[0].text_reply == 'ABC123'
        assert out['c1'].make == 'Ford'
        # Bbox-wrong-but-region-visible.
        assert out['c2'] is not None
        assert out['c2'].region_visible is True
        assert out['c2'].region_boxes[0].bbox_correct is False
        # No-region-visible.
        assert out['c3'] is not None
        assert out['c3'].region_visible is False

    @pytest.mark.asyncio
    async def test_payload_is_multi_image_with_per_image_directives(self) -> None:
        labeler = _make_labeler(
            [
                {
                    'img': i,
                    'class_id': 0,
                    'region_visible': True,
                    'region_boxes': [{'box': 1, 'region_bbox_correct': True}],
                }
                for i in (1, 2)
            ]
        )
        crops = [
            CombinedCrop(
                crop_id='a',
                jpeg_bytes=_make_jpeg(),
                region_bboxes_norm=[(0.1, 0.7, 0.9, 0.95)],
                classify=True,
            ),
            CombinedCrop(
                crop_id='b', jpeg_bytes=_make_jpeg(), region_bboxes_norm=[], classify=False
            ),
        ]
        await labeler.label_combined_batch(crops, class_names=['widget'])
        sent = labeler._client.post.await_args.kwargs['json']
        user_msg = sent['messages'][-1]['content']
        # Header + 2 (text directive, image) pairs.
        assert len(user_msg) == 1 + 2 * len(crops)
        text_parts = [p['text'] for p in user_msg if p['type'] == 'text']
        # Per-image directives present in order.
        assert any('Image 1' in t for t in text_parts)
        assert any('Image 2' in t for t in text_parts)
        # Crop b asked to skip classification.
        assert any('skip classification' in t for t in text_parts)

    @pytest.mark.asyncio
    async def test_short_array_returns_none_for_missing_entries(self) -> None:
        # Only 1 entry in the response — second crop should sentinel to None.
        labeler = _make_labeler(
            [
                {
                    'img': 1,
                    'class_id': 0,
                    'region_visible': True,
                    'region_boxes': [{'box': 1, 'region_bbox_correct': True}],
                }
            ]
        )
        crops = [
            CombinedCrop(
                crop_id='c1', jpeg_bytes=_make_jpeg(), region_bboxes_norm=[(0.1, 0.7, 0.9, 0.95)]
            ),
            CombinedCrop(
                crop_id='c2', jpeg_bytes=_make_jpeg(), region_bboxes_norm=[(0.1, 0.7, 0.9, 0.95)]
            ),
        ]
        out = await labeler.label_combined_batch(crops, class_names=['widget'])
        assert out['c1'] is not None
        assert out['c2'] is None  # per-crop parse failure sentinel

    @pytest.mark.asyncio
    async def test_empty_response_fails_all_crops_closed(self) -> None:
        # Empty raw from the VLM → every crop in chunk gets None so the
        # caller can leave them in pending for the next poll.
        labeler = _make_labeler('')
        crops = [
            CombinedCrop(crop_id=f'c{i}', jpeg_bytes=_make_jpeg(), region_bboxes_norm=[])
            for i in range(1, 4)
        ]
        out = await labeler.label_combined_batch(crops, class_names=['widget'])
        assert out == {'c1': None, 'c2': None, 'c3': None}

    @pytest.mark.asyncio
    async def test_http_failure_raises_transport_failure(self) -> None:
        # No reply at all is not a reply without a verdict: the caller
        # retries an outage but caps no-verdict replies.
        labeler = _make_labeler({})
        labeler._client.post = AsyncMock(side_effect=httpx.ConnectError('boom'))
        crops = [
            CombinedCrop(crop_id='c1', jpeg_bytes=_make_jpeg(), region_bboxes_norm=[]),
            CombinedCrop(crop_id='c2', jpeg_bytes=_make_jpeg(), region_bboxes_norm=[]),
        ]
        with pytest.raises(CombinedTransportError):
            await labeler.label_combined_batch(crops, class_names=['widget'])

    @pytest.mark.asyncio
    async def test_single_crop_chunk_delegates_to_label_combined(self) -> None:
        # Length-1 chunks go through the single-image path (better
        # accuracy, less prompt overhead than the numbered-image path).
        labeler = _make_labeler(
            {
                'class_id': 0,
                'class_confidence': 'high',
                'region_visible': True,
                'region_boxes': [{'box': 1, 'region_bbox_correct': True, 'region_text': 'XYZ789'}],
            }
        )
        crops = [
            CombinedCrop(
                crop_id='only',
                jpeg_bytes=_make_jpeg(),
                region_bboxes_norm=[(0.1, 0.7, 0.9, 0.95)],
            ),
        ]
        out = await labeler.label_combined_batch(crops, class_names=['widget'])
        assert out['only'] is not None
        assert out['only'].region_boxes[0].text_reply == 'XYZ789'
        sent = labeler._client.post.await_args.kwargs['json']
        # Single-image path uses response_format json_object.
        assert sent.get('response_format') == {'type': 'json_object'}

    @pytest.mark.asyncio
    async def test_chunks_into_multiple_calls_when_over_limit(self) -> None:
        # 8 crops with max_images_per_call=4 → 2 upstream calls.
        labeler = _make_labeler(
            [
                {
                    'img': i + 1,
                    'class_id': 0,
                    'region_visible': True,
                    'region_boxes': [{'box': 1, 'region_bbox_correct': True}],
                }
                for i in range(4)
            ]
        )
        labeler.max_images_per_call = 4
        crops = [
            CombinedCrop(crop_id=f'c{i}', jpeg_bytes=_make_jpeg(), region_bboxes_norm=[])
            for i in range(8)
        ]
        out = await labeler.label_combined_batch(crops, class_names=['widget'])
        assert len(out) == 8
        # Both chunks made one upstream call each.
        assert labeler._client.post.await_count == 2

    @pytest.mark.asyncio
    async def test_multi_box_reply_aligns_by_position(self) -> None:
        """W8: 2 candidate boxes on one crop -> 2 verdicts, aligned by
        1-based ``box`` number."""
        labeler = _make_labeler(
            {
                'class_id': 0,
                'region_visible': True,
                'region_boxes': [
                    {'box': 1, 'region_bbox_correct': True, 'region_confidence': 'high'},
                    {'box': 2, 'region_bbox_correct': False, 'region_confidence': 'low'},
                ],
            }
        )
        reply = await labeler.label_combined(
            'crop-multi',
            _make_jpeg(),
            class_names=['widget'],
            region_bboxes_norm=[(0.1, 0.1, 0.4, 0.4), (0.5, 0.5, 0.9, 0.9)],
        )
        assert len(reply.region_boxes) == 2
        assert reply.region_boxes[0].bbox_correct is True
        assert reply.region_boxes[1].bbox_correct is False
