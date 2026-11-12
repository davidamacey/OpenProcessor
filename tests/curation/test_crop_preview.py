"""``GET /crops/{id}/preview``: a padded context crop, enlarged with LANCZOS, cached.

The thumbnail never upscales (``Image.thumbnail``), so a 7 px box came back 7 px wide; the preview
serves it large enough to see.
"""

from __future__ import annotations

import io
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from PIL import Image

from src.services.curation import crop_preview
from src.services.curation.crop_preview import (
    OUTLINE_RGB,
    PreviewCache,
    padded_pixel_box,
    render_preview,
    target_scale,
)
from src.services.curation.image_serving import ThumbnailCache


if TYPE_CHECKING:
    from pathlib import Path

# A 7 x 7 px box in a 400 x 300 photo, the size the audit saw.
TINY = (0.5, 0.5, 0.5 + 7 / 400, 0.5 + 7 / 300)


@pytest.fixture
def photo(tmp_path: Path) -> Path:
    path = tmp_path / 'photo.jpg'
    rng = np.random.default_rng(1)
    Image.fromarray(rng.integers(0, 255, (300, 400, 3), dtype=np.uint8)).save(path, quality=95)
    return path


def _size(jpeg: bytes) -> tuple[int, int]:
    with Image.open(io.BytesIO(jpeg)) as img:
        return img.size


def test_the_thumbnail_of_a_tiny_box_stays_tiny(photo: Path) -> None:
    """The behaviour the preview exists to fix."""
    thumb = ThumbnailCache._render_thumbnail(photo, TINY, 384)
    assert max(_size(thumb)) <= 8


def test_a_tiny_box_is_enlarged_to_the_minimum_side(photo: Path) -> None:
    jpeg = render_preview(photo, TINY, pad=0.6, min_side=320, max_side=1024, outline=False)
    assert max(_size(jpeg)) == 320


def test_a_large_context_is_shrunk_to_the_maximum_side(photo: Path) -> None:
    jpeg = render_preview(
        photo, (0.0, 0.0, 1.0, 1.0), pad=0.6, min_side=100, max_side=200, outline=False
    )
    assert max(_size(jpeg)) == 200


def test_a_context_inside_the_limits_keeps_its_size(photo: Path) -> None:
    jpeg = render_preview(
        photo, (0.25, 0.25, 0.75, 0.75), pad=0.0, min_side=100, max_side=1024, outline=False
    )
    assert _size(jpeg) == (200, 150)


def test_the_padding_is_a_share_of_the_box_and_stops_at_the_image_edge() -> None:
    context, box = padded_pixel_box((0.5, 0.5, 0.6, 0.6), 400, 300, 0.5)
    assert box == (200, 150, 240, 180)
    assert context == (180, 135, 260, 195)
    corner, _ = padded_pixel_box((0.0, 0.0, 0.1, 0.1), 400, 300, 5.0)
    assert corner[:2] == (0, 0)
    far, _ = padded_pixel_box((0.9, 0.9, 1.0, 1.0), 400, 300, 5.0)
    assert far[2:] == (400, 300)


def test_a_box_with_no_area_is_refused() -> None:
    with pytest.raises(ValueError, match='invalid bbox'):
        padded_pixel_box((0.5, 0.5, 0.5, 0.5), 400, 300, 0.6)


@pytest.mark.parametrize(('longest', 'expected'), [(10, 32.0), (500, 1.0), (4000, 0.25)])
def test_the_scale_brings_the_longest_side_into_range(longest: int, expected: float) -> None:
    assert target_scale(longest, 320, 1000) == pytest.approx(expected)


def test_the_outline_is_drawn_around_the_box_in_the_highlight_colour(photo: Path) -> None:
    box = (0.25, 0.25, 0.75, 0.75)
    plain = render_preview(photo, box, pad=0.5, min_side=100, max_side=1024, outline=False)
    marked = render_preview(photo, box, pad=0.5, min_side=100, max_side=1024, outline=True)
    with Image.open(io.BytesIO(plain)) as a, Image.open(io.BytesIO(marked)) as b:
        w, h = b.size
        edge = (w // 4 + 1, h // 2)  # the left edge of the box inside the padded context
        r, g, blue = b.getpixel(edge)  # type: ignore[misc]
        assert abs(r - OUTLINE_RGB[0]) < 40
        assert abs(g - OUTLINE_RGB[1]) < 40
        assert blue < 80
        assert a.getpixel(edge) != b.getpixel(edge)


@pytest.mark.asyncio
async def test_a_repeat_is_served_from_the_cache_and_a_moved_box_is_not(
    photo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[Any] = []
    real = crop_preview.render_preview

    def counting(*args: Any, **kwargs: Any) -> bytes:
        calls.append(args[1])
        return real(*args, **kwargs)

    monkeypatch.setattr(crop_preview, 'render_preview', counting)
    cache = PreviewCache(max_bytes=1 << 20)
    params: dict[str, Any] = {'pad': 0.6, 'min_side': 320, 'max_side': 1024, 'outline': False}
    first = await cache.get_or_compute(photo, TINY, **params)
    again = await cache.get_or_compute(photo, TINY, **params)
    assert first == again
    assert len(calls) == 1
    await cache.get_or_compute(photo, (0.1, 0.1, 0.2, 0.2), **params)
    await cache.get_or_compute(photo, TINY, **{**params, 'outline': True})
    assert len(calls) == 3


@pytest.mark.asyncio
async def test_the_cache_is_bounded_by_bytes_not_entries(photo: Path) -> None:
    params: dict[str, Any] = {'pad': 0.6, 'min_side': 320, 'max_side': 1024, 'outline': False}
    one = len(await PreviewCache(max_bytes=1 << 20).get_or_compute(photo, TINY, **params))
    cache = PreviewCache(max_bytes=one * 2 + 1)
    for i in range(5):
        await cache.get_or_compute(photo, (0.1 + i / 20, 0.1, 0.2 + i / 20, 0.2), **params)
    assert cache.size_bytes <= one * 2 + 1
    # An entry larger than the whole budget is served, never stored (and never raises).
    tiny = PreviewCache(max_bytes=one - 1)
    assert await tiny.get_or_compute(photo, TINY, **params)
    assert tiny.size_bytes == 0


# ---------------------------------------------------------------------------
# The route
# ---------------------------------------------------------------------------


@pytest.fixture
def client(photo: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from src.routers.curation import images as images_router

    async def fetch_crop(_crop_id: str, _opensearch: Any) -> dict[str, Any]:
        return {'bbox_norm': list(TINY)}

    monkeypatch.setattr(images_router, '_fetch_crop', fetch_crop)
    monkeypatch.setattr(images_router, '_resolve_image_for_crop', lambda _crop: photo)
    monkeypatch.setattr(images_router, 'PREVIEW_CACHE', PreviewCache(max_bytes=1 << 20))
    app = FastAPI()
    app.include_router(images_router.crops_router)
    app.dependency_overrides[images_router._raw_opensearch_dep] = lambda: object()
    with TestClient(app) as c:
        yield c


def test_the_route_serves_an_enlarged_jpeg(client: Any) -> None:
    r = client.get('/crops/c1/preview')
    assert r.status_code == 200, r.text
    assert r.headers['content-type'] == 'image/jpeg'
    assert max(_size(r.content)) == 320  # the defaults: pad 0.6, min_side 320


def test_the_route_revalidates_with_an_etag_that_names_the_parameters(client: Any) -> None:
    first = client.get('/crops/c1/preview')
    again = client.get('/crops/c1/preview', headers={'If-None-Match': first.headers['etag']})
    assert again.status_code == 304
    other = client.get('/crops/c1/preview', params={'outline': 'true'})
    assert other.headers['etag'] != first.headers['etag']
    assert (
        client.get(
            '/crops/c1/preview',
            params={'outline': 'true'},
            headers={'If-None-Match': first.headers['etag']},
        ).status_code
        == 200
    )


def test_min_side_above_max_side_is_refused(client: Any) -> None:
    r = client.get('/crops/c1/preview', params={'min_side': 500, 'max_side': 400})
    assert r.status_code == 422
    assert r.json()['detail']['error'] == 'preview_sides'


def test_the_plain_thumbnail_may_be_asked_up_to_1024(client: Any) -> None:
    assert client.get('/crops/c1/thumbnail', params={'size': 1024}).status_code == 200
    assert client.get('/crops/c1/thumbnail', params={'size': 1025}).status_code == 422
