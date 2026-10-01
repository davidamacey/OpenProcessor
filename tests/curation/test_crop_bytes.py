"""W5: one function builds the item-crop JPEG, for the worker and for the
test-on-crop routes, so a preview crop is the crop the worker sees."""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from PIL import Image

from scripts.curation.worker import state
from src.services.curation import crop_bytes


@pytest.fixture
def source_image(tmp_path: Path) -> Path:
    img = Image.new('RGB', (200, 100), (10, 20, 30))
    for x in range(100, 200):  # the right half is a different colour
        for y in range(100):
            img.putpixel((x, y), (200, 40, 40))
    path = tmp_path / 'frame.jpg'
    img.save(path, format='JPEG', quality=95)
    return path


@pytest.fixture
def cache_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    cache = tmp_path / 'crop_cache'
    cache.mkdir()
    monkeypatch.setenv('OP_CROP_CACHE_DIR', str(cache))
    import src.config.curation as curation_config_mod

    monkeypatch.setattr(curation_config_mod, '_default_curation_config', None)
    return cache


def _size(jpeg: bytes) -> tuple[int, int]:
    return Image.open(io.BytesIO(jpeg)).size


def test_a_cache_hit_is_returned_verbatim(cache_dir: Path, source_image: Path) -> None:
    (cache_dir / 'c1.jpg').write_bytes(b'cached-bytes')

    out = crop_bytes.load_item_crop_jpeg('c1', str(source_image), (0.0, 0.0, 0.5, 1.0))

    assert out == b'cached-bytes'


def test_a_cache_miss_cuts_the_crop_from_the_source_image(
    cache_dir: Path, source_image: Path
) -> None:
    out = crop_bytes.load_item_crop_jpeg('absent', str(source_image), (0.5, 0.0, 1.0, 1.0))

    assert out is not None
    assert _size(out) == (100, 100)
    red, _green, _blue = Image.open(io.BytesIO(out)).convert('RGB').getpixel((50, 50))
    assert red > 150, 'the right half of the frame was cropped'


def test_the_cache_counters_follow_the_lookups(cache_dir: Path, source_image: Path) -> None:
    (cache_dir / 'hit.jpg').write_bytes(b'x')
    hits, misses = crop_bytes.cache_stats()

    crop_bytes.load_item_crop_jpeg('hit', str(source_image), (0, 0, 1, 1))
    crop_bytes.load_item_crop_jpeg('miss', str(source_image), (0, 0, 1, 1))

    assert crop_bytes.cache_stats() == (hits + 1, misses + 1)


def test_an_unreadable_source_is_none_not_an_exception(cache_dir: Path, tmp_path: Path) -> None:
    junk = tmp_path / 'junk.jpg'
    junk.write_bytes(b'not an image')

    assert crop_bytes.load_item_crop_jpeg('x', str(junk), (0, 0, 1, 1)) is None
    assert crop_bytes.load_item_crop_jpeg('x', str(tmp_path / 'gone.jpg'), (0, 0, 1, 1)) is None


def test_the_worker_gets_exactly_the_bytes_the_shared_function_returns(
    cache_dir: Path, source_image: Path
) -> None:
    bbox = (0.25, 0.1, 0.9, 0.8)
    expected = crop_bytes.load_item_crop_jpeg('w1', str(source_image), bbox)
    (cache_dir / 'w2.jpg').write_bytes(b'from-cache')

    assert state._crop_jpeg_for_task('w1', str(source_image), bbox) == expected
    assert state._crop_jpeg_for_task('w2', str(source_image), bbox) == b'from-cache'


@pytest.fixture
def servable(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Every stored image path is servable (the path guard has its own tests)."""
    from src.services.curation import image_serving

    monkeypatch.setattr(image_serving, 'resolve_crop_root', lambda _p: tmp_path)
    monkeypatch.setattr(image_serving, 'resolve_safe_path', lambda p, _root: Path(p))
    image_serving.THUMBNAIL_CACHE.clear()


def test_the_vlm_crop_from_the_cache_is_shrunk_to_vlm_size(
    cache_dir: Path, servable: None, tmp_path: Path
) -> None:
    big = io.BytesIO()
    Image.new('RGB', (900, 600), (1, 2, 3)).save(big, format='JPEG')
    (cache_dir / 'c1.jpg').write_bytes(big.getvalue())

    out = crop_bytes.load_vlm_item_jpeg('c1', '/gone.jpg', (0, 0, 1, 1), cache_dir=cache_dir)

    assert out is not None
    assert max(_size(out)) == crop_bytes.VLM_CROP_SIZE


def test_the_vlm_crop_falls_back_to_a_thumbnail_of_the_source(
    cache_dir: Path, source_image: Path, servable: None
) -> None:
    out = crop_bytes.load_vlm_item_jpeg(
        'absent', str(source_image), (0.0, 0.0, 0.5, 1.0), cache_dir=cache_dir
    )

    assert out is not None
    assert max(_size(out)) <= crop_bytes.VLM_CROP_SIZE
    assert crop_bytes.load_vlm_item_jpeg('x', '/gone.jpg', (0, 0, 1, 1), cache_dir=None) is None


def test_a_region_close_up_is_cut_from_the_source_at_vlm_size(
    source_image: Path, servable: None
) -> None:
    out = crop_bytes.load_region_jpeg(str(source_image), (0.5, 0.0, 1.0, 1.0))

    assert max(_size(out)) <= crop_bytes.VLM_CROP_SIZE
    red, _g, _b = Image.open(io.BytesIO(out)).convert('RGB').getpixel((10, 10))
    assert red > 150
