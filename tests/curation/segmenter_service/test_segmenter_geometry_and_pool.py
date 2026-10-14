"""Box normalization, rectangularity, the processor pool and mask polygons."""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from curation.segmenter_service._helpers import (
    _asgi_client,
    _disc_mask,
    _FakeProcessor,
    _full_mask,
    _segment,
    sam3_backend,
    segmenter,
)


# =============================================================================
# Backend units that the HTTP path cannot reach
# =============================================================================


class TestBoxNormalization:
    def test_pixel_space_boxes_are_divided_by_the_image_size(self) -> None:
        assert sam3_backend._bbox_to_norm([64.0, 48.0, 128.0, 96.0], 256, 192) == pytest.approx(
            (0.25, 0.25, 0.5, 0.5)
        )

    def test_normalized_boxes_pass_through(self) -> None:
        assert sam3_backend._bbox_to_norm([0.25, 0.25, 0.5, 0.5], 256, 192) == pytest.approx(
            (0.25, 0.25, 0.5, 0.5)
        )

    def test_inverted_and_out_of_range_boxes_are_canonicalized(self) -> None:
        assert sam3_backend._bbox_to_norm([0.9, 0.9, -0.2, 0.4], 100, 100) == pytest.approx(
            (0.0, 0.4, 0.9, 0.9)
        )

    def test_short_box_is_rejected_as_degenerate(self) -> None:
        assert sam3_backend._bbox_to_norm([0.1, 0.2], 100, 100) == (0.0, 0.0, 0.0, 0.0)


class TestRectangularity:
    def test_a_filled_bbox_scores_one(self) -> None:
        assert sam3_backend._rectangularity(_full_mask(4, 4)) == pytest.approx(1.0)

    def test_a_half_filled_bbox_scores_half(self) -> None:
        mask = np.zeros((4, 4), dtype=np.float32)
        mask[:2, :] = 1.0
        mask[2:, :2] = 1.0  # bbox is the full 4x4; 8+4 of 16 pixels set
        assert sam3_backend._rectangularity(mask) == pytest.approx(0.75)

    def test_an_empty_mask_scores_zero(self) -> None:
        assert sam3_backend._rectangularity(np.zeros((4, 4), dtype=np.float32)) == 0.0


class TestProcessorPool:
    @pytest.mark.asyncio
    async def test_concurrent_callers_get_different_processors(self) -> None:
        pool = sam3_backend.ProcessorPool(['a', 'b'])
        held: list[str] = []
        release = asyncio.Event()

        async def hold() -> None:
            async with pool.acquire() as p:
                held.append(p)
                await release.wait()

        tasks = [asyncio.create_task(hold()) for _ in range(2)]
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        release.set()
        await asyncio.gather(*tasks)

        assert sorted(held) == ['a', 'b']

    @pytest.mark.asyncio
    async def test_a_busy_pool_serializes_rather_than_dropping(self) -> None:
        pool = sam3_backend.ProcessorPool(['only'])
        order: list[str] = []

        async def use(tag: str) -> None:
            async with pool.acquire():
                order.append(f'{tag}:in')
                await asyncio.sleep(0.01)
                order.append(f'{tag}:out')

        await asyncio.gather(use('a'), use('b'))

        # Never interleaved: one holder at a time.
        assert order in (
            ['a:in', 'a:out', 'b:in', 'b:out'],
            ['b:in', 'b:out', 'a:in', 'a:out'],
        )

    @pytest.mark.asyncio
    async def test_an_empty_pool_refuses_rather_than_dividing_by_zero(self) -> None:
        pool = sam3_backend.ProcessorPool([])
        assert not pool
        with pytest.raises(RuntimeError, match='empty'):
            async with pool.acquire():
                pass  # pragma: no cover


class TestMaskPolygons:
    @pytest.mark.asyncio
    async def test_return_masks_serves_a_normalized_bounded_polygon(self) -> None:
        proc = _FakeProcessor(boxes=[[0.1, 0.2, 0.5, 0.8]], scores=[0.9], masks=_disc_mask())
        previous = segmenter._pool
        segmenter._pool = sam3_backend.ProcessorPool([proc])
        try:
            async with _asgi_client() as http:
                body = await _segment(http, return_masks=True)
        finally:
            segmenter._pool = previous

        polygon = body['candidates'][0]['mask_polygon']
        assert 3 <= len(polygon) <= 256
        xs = [p[0] for p in polygon]
        ys = [p[1] for p in polygon]
        assert all(0.0 <= v <= 1.0 for v in xs + ys)
        # The disc (centre 24/96, 32/64; radius 20px) and not the small blob
        # at x ~ 0.9: the largest external contour only.
        assert min(xs) == pytest.approx(4 / 96, abs=0.03)
        assert max(xs) == pytest.approx(44 / 96, abs=0.03)
        assert max(xs) < 0.6
        assert min(ys) == pytest.approx(12 / 64, abs=0.03)
        assert max(ys) == pytest.approx(52 / 64, abs=0.03)

    @pytest.mark.asyncio
    async def test_the_polygon_is_not_computed_unless_asked_for(self) -> None:
        proc = _FakeProcessor(boxes=[[0.1, 0.2, 0.5, 0.8]], scores=[0.9], masks=_disc_mask())
        previous = segmenter._pool
        segmenter._pool = sam3_backend.ProcessorPool([proc])
        try:
            async with _asgi_client() as http:
                body = await _segment(http)
        finally:
            segmenter._pool = previous

        cand = body['candidates'][0]
        assert cand['mask_polygon'] is None
        assert cand['mask_iou'] is not None  # the rectangularity is unaffected

    @pytest.mark.asyncio
    async def test_return_masks_with_the_mask_head_off_is_null_not_fabricated(self) -> None:
        proc = _FakeProcessor(boxes=[[0.1, 0.2, 0.5, 0.8]], scores=[0.9], masks=None)
        previous = segmenter._pool
        segmenter._pool = sam3_backend.ProcessorPool([proc])
        try:
            async with _asgi_client() as http:
                body = await _segment(http, return_masks=True)
        finally:
            segmenter._pool = previous

        assert body['candidates'][0]['mask_polygon'] is None

    def test_a_ragged_mask_is_coarsened_to_the_point_ceiling_never_truncated(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        h, w = 400, 400
        yy, xx = np.mgrid[0:h, 0:w]
        theta = np.arctan2(yy - 200, xx - 200)
        radius = 120 + 60 * np.sin(37 * theta)
        mask = (np.hypot(yy - 200, xx - 200) <= radius).astype(np.float32)[None]

        detailed = sam3_backend.mask_polygon(mask)
        assert detailed is not None
        assert len(detailed) > 40, 'the fixture must need more points than the ceiling below'
        monkeypatch.setattr(sam3_backend, '_POLYGON_MAX_POINTS', 24)

        coarse = sam3_backend.mask_polygon(mask)

        assert coarse is not None
        assert 3 <= len(coarse) <= 24
        # Coarsened, not cut off: it still spans the whole star.
        xs = [p[0] for p in coarse]
        assert max(xs) - min(xs) > 0.5

    def test_the_shipped_ceiling_is_256(self) -> None:
        assert sam3_backend._POLYGON_MAX_POINTS == 256

    def test_an_empty_mask_has_no_polygon(self) -> None:
        assert sam3_backend.mask_polygon(np.zeros((1, 8, 8), dtype=np.float32)) is None
