"""The thumbnail and source-image handlers decode with PIL off the event loop."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from PIL import Image

from src.routers.curation import images as images_router
from src.services.curation.image_serving import ThumbnailCache


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from pathlib import Path

# Loop-lag budget while a large decode runs; the in-place PIL call stalls the loop for
# several hundred ms on this image, the offloaded one for a few ms.
MAX_LAG_S = 0.1


@pytest.fixture(scope='module')
def big_image(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp('big') / 'big.jpg'
    rng = np.random.default_rng(0)
    Image.fromarray(rng.integers(0, 255, (6000, 6000, 3), dtype=np.uint8)).save(path, quality=60)
    return path


@pytest.fixture
def handlers(monkeypatch: pytest.MonkeyPatch, big_image: Path) -> None:
    async def fetch_crop(_crop_id: str, _opensearch: Any) -> dict[str, Any]:
        return {'bbox_norm': [0.0, 0.0, 1.0, 1.0]}

    monkeypatch.setattr(images_router, '_fetch_crop', fetch_crop)
    monkeypatch.setattr(images_router, '_resolve_image_for_crop', lambda _crop: big_image)
    monkeypatch.setattr(
        images_router,
        'read_boxes',
        lambda *_a, **_k: [SimpleNamespace(box_id='b1', bbox_norm=[0.0, 0.0, 1.0, 1.0])],
    )
    monkeypatch.setattr(images_router, 'THUMBNAIL_CACHE', ThumbnailCache(maxsize=8))


async def _max_lag(work: Callable[[], Awaitable[Any]]) -> float:
    loop = asyncio.get_running_loop()
    worst = 0.0
    stop = asyncio.Event()

    async def ticker() -> None:
        nonlocal worst
        while not stop.is_set():
            before = loop.time()
            await asyncio.sleep(0.005)
            worst = max(worst, loop.time() - before - 0.005)

    task = asyncio.create_task(ticker())
    await asyncio.sleep(0.02)
    await work()
    stop.set()
    await task
    return worst


@pytest.mark.asyncio
async def test_crop_thumbnail_does_not_block_the_loop(handlers: None) -> None:
    lag = await _max_lag(lambda: images_router.crop_thumbnail('c1', None, size=128))

    assert lag < MAX_LAG_S


@pytest.mark.asyncio
async def test_region_thumbnail_does_not_block_the_loop(handlers: None) -> None:
    request: Any = SimpleNamespace(headers={})
    lag = await _max_lag(
        lambda: images_router.crop_region_thumbnail('c1', request, None, box_id='b1', size=128)
    )

    assert lag < MAX_LAG_S


@pytest.mark.asyncio
async def test_full_image_does_not_block_the_loop(handlers: None) -> None:
    lag = await _max_lag(lambda: images_router.crop_full_image('c1', None, max_dim=None))

    assert lag < MAX_LAG_S


@pytest.mark.asyncio
async def test_cached_thumbnail_still_served(handlers: None) -> None:
    first = await images_router.crop_thumbnail('c1', None, size=128)
    second = await images_router.crop_thumbnail('c1', None, size=128)

    assert first.body == second.body
    assert images_router.THUMBNAIL_CACHE.cache_info()['hits'] == 1
