"""CC-10 evidence: one batched VLM call never has more than the bound in flight, whatever the cap."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from src.services.labeling.vlm_chunks import VLM_CHUNK_CONCURRENCY, run_chunks
from src.services.labeling.vlm_labeler import VlmLabeler
from src.services.labeling.vlm_models import RegionCrop


class _StubVlm(VlmLabeler):
    """A labeler whose upstream call is a stub that counts calls in flight."""

    def __init__(self) -> None:
        super().__init__(
            base_url='http://vlm.invalid/v1', model='m', max_images_per_call=1, usage_source=None
        )
        self.live = 0
        self.peak = 0
        self.calls = 0

    async def _post_chat(self, _payload: dict[str, Any], **_: Any) -> dict[str, Any]:
        self.live += 1
        self.calls += 1
        self.peak = max(self.peak, self.live)
        await asyncio.sleep(0.005)
        self.live -= 1
        return {'choices': [{'message': {'content': '{"results": []}'}}]}


@pytest.mark.asyncio
async def test_64_crops_at_one_image_per_call_never_exceed_the_bound() -> None:
    vlm = _StubVlm()
    crops = [RegionCrop(crop_id=f'c{i}', jpeg_bytes=b'x') for i in range(64)]
    await vlm.verify_region_batch(crops)
    assert vlm.calls == 64
    assert vlm.peak == VLM_CHUNK_CONCURRENCY


@pytest.mark.asyncio
async def test_results_keep_chunk_order_and_a_failing_chunk_cancels_its_siblings() -> None:
    async def slow(n: int) -> int:
        await asyncio.sleep(0.01 * (5 - n))
        return n

    assert await run_chunks([slow(n) for n in range(5)], limit=2) == [0, 1, 2, 3, 4]

    started: list[int] = []

    async def boom() -> None:
        raise RuntimeError('upstream')

    async def waits(n: int) -> None:
        started.append(n)
        await asyncio.sleep(10)

    with pytest.raises(ExceptionGroup):
        await run_chunks([boom(), waits(1), waits(2)], limit=1)
