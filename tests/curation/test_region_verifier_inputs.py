"""What the verifier is shown and how a non-answer is read (MODEL notes, DQ §6).

* The candidate-box overlay is drawn *around* the region, never over it:
  typical regions are ~25 px tall, and a 3 px outline drawn inside the box
  covered a quarter of the text the verifier is asked to read and judge.
* The per-image directive says what the drawn rectangle is.
* An empty visibility reply is no verdict -- the items are retried, not
  stamped "no region visible".
"""

from __future__ import annotations

import io
import json
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock

import pytest
from PIL import Image

from src.config import get_region_fields
from src.services.detection.cascade_detect import RegionCandidate
from src.services.labeling.vlm_labeler import (
    CombinedCrop,
    RegionCrop,
    VlmCombinedReply,
    VlmLabeler,
    _draw_bbox_overlay,
)

from .test_region_cascade_integrity import _chat, _drive_worker, _FakeOpenSearch, _item


if TYPE_CHECKING:
    from pathlib import Path


F = get_region_fields()


def _gray_jpeg(w: int = 200, h: int = 100) -> bytes:
    buf = io.BytesIO()
    Image.new('RGB', (w, h), (128, 128, 128)).save(buf, format='JPEG', quality=95)
    return buf.getvalue()


def _is_red(px: tuple[int, int, int]) -> bool:
    r, g, b = px
    return r > 180 and g < 90 and b < 90


class TestOverlay:
    def test_outline_is_drawn_outside_the_region(self) -> None:
        # Region x 50..150, y 40..60 (a 20 px tall region).
        out = _draw_bbox_overlay(_gray_jpeg(), (0.25, 0.4, 0.75, 0.6))
        assert out is not None
        im = Image.open(io.BytesIO(out)).convert('RGB')
        # Just outside the region's edges: the outline.
        assert _is_red(im.getpixel((48, 50)))
        assert _is_red(im.getpixel((100, 38)))
        # The region's own edge pixels stay as they were.
        for xy in ((51, 50), (100, 41), (149, 50), (100, 59)):
            assert not _is_red(im.getpixel(xy)), xy


class TestDirective:
    @pytest.mark.asyncio
    async def test_batch_directive_names_the_red_rectangle(self) -> None:
        lab = VlmLabeler(base_url='http://vlm.invalid/v1')
        post = AsyncMock(return_value=_chat(json.dumps({'results': []})))
        lab._post_chat = post  # type: ignore[method-assign]
        crops = [
            CombinedCrop(
                crop_id=f'c{i}', jpeg_bytes=_gray_jpeg(), region_bbox_norm=(0.2, 0.2, 0.6, 0.6)
            )
            for i in range(2)
        ]
        await lab.label_combined_batch(crops)
        assert post.await_args is not None
        texts = [
            part['text']
            for part in post.await_args.args[0]['messages'][1]['content']
            if part['type'] == 'text'
        ]
        assert any('red rectangle' in t for t in texts[1:])

    @pytest.mark.asyncio
    async def test_single_call_names_the_red_rectangle(self) -> None:
        lab = VlmLabeler(base_url='http://vlm.invalid/v1')
        reply = {'region_visible': True, 'region_bbox_correct': True}
        post = AsyncMock(return_value=_chat(json.dumps(reply)))
        lab._post_chat = post  # type: ignore[method-assign]
        await lab.label_combined('c1', _gray_jpeg(), region_bbox_norm=(0.2, 0.2, 0.6, 0.6))
        assert post.await_args is not None
        user = post.await_args.args[0]['messages'][1]['content'][0]['text']
        assert 'red rectangle' in user


class TestVisibilityNoVerdict:
    def test_empty_reply_is_no_verdict(self) -> None:
        crops = [RegionCrop(crop_id=c, jpeg_bytes=b'x') for c in ('a', 'b')]
        assert VlmLabeler._parse_region_visible_response('', crops, F) == {}

    @pytest.mark.asyncio
    @pytest.mark.usefixtures('reference_region_profile')
    async def test_worker_retries_an_item_without_a_visibility_verdict(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Below the no-verdict cap (test_region_no_verdict_cap.py covers it).
        monkeypatch.setenv('OP_REGION_WORKER_MAX_NO_VERDICT_ATTEMPTS', '100000')
        fake_os = _FakeOpenSearch({'c1': _item()}, search_delay=0.0, lag_searches=0)
        mocks: dict[str, Any] = await _drive_worker(
            tmp_path,
            monkeypatch,
            fake_os=fake_os,
            primary=None,
            segmenter=RegionCandidate(bbox_norm=(0.3, 0.6, 0.6, 0.75), score=0.5, source='seg'),
            reply=VlmCombinedReply(img_id='c1', region_visible=True, region_bbox_correct=True),
            visible=None,
        )
        assert mocks['vlm'].region_visible_batch.await_count >= 2
        assert fake_os.writes == []
        assert fake_os.live['c1'][F.status] == 'pending_detection'
