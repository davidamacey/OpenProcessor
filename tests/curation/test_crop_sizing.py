"""The crop cap (``OP_CROP_MAX_SIDE``) and the VLM cap (``OP_VLM_CROP_MAX_SIDE``).

Cap: long side at most the cap, never upscaled, aspect kept; a cache hit and a miss give the same
dimensions; with the cap 0 the bytes are today's."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.services.curation import crop_bytes
from src.services.curation.source_image_cache import write_crop_cache
from src.services.detection.cascade_detect.sanity import crop_norm_to_source_norm
from src.services.detection.crop_sizing import cap_crop


def _photo(w: int, h: int) -> Image.Image:
    """A smooth gradient with noise, so the JPEG payload scales with the pixel count."""
    rng = np.random.default_rng(7)
    x = np.linspace(0, 255, w, dtype=np.float32)[None, :, None]
    y = np.linspace(0, 255, h, dtype=np.float32)[:, None, None]
    base = (x + y) / 2 + rng.normal(0, 12, (h, w, 3))
    return Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))


@pytest.fixture(scope='module')
def big_source(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp('src') / 'big.jpg'
    _photo(4000, 3000).save(path, format='JPEG', quality=90)
    return path


@pytest.fixture
def config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Set the caps through the env and rebuild the default config; returns the cache dir."""
    import src.config.curation_accessors as accessors

    cache = tmp_path / 'crop_cache'
    cache.mkdir()
    monkeypatch.setenv('OP_CROP_CACHE_DIR', str(cache))

    def apply(crop: int = 0, vlm: int = 0) -> Path:
        monkeypatch.setenv('OP_CROP_MAX_SIDE', str(crop))
        monkeypatch.setenv('OP_VLM_CROP_MAX_SIDE', str(vlm))
        monkeypatch.setattr(accessors, '_default_curation_config', None)
        return cache

    apply()
    return apply


def _size(jpeg: bytes) -> tuple[int, int]:
    return Image.open(io.BytesIO(jpeg)).size


def test_cap_long_side_never_exceeds_the_cap_and_keeps_aspect() -> None:
    out = cap_crop(Image.new('RGB', (4000, 3000)), 1024)
    assert out.size == (1024, 768)
    tall = cap_crop(Image.new('RGB', (900, 3000)), 1000)
    assert max(tall.size) == 1000
    assert abs(tall.width / tall.height - 900 / 3000) < 0.005


def test_cap_never_upscales_and_zero_is_off() -> None:
    small = Image.new('RGB', (300, 200))
    assert cap_crop(small, 1024) is small
    assert cap_crop(Image.new('RGB', (4000, 3000)), 0).size == (4000, 3000)


def test_cap_does_not_modify_its_argument() -> None:
    img = Image.new('RGB', (2000, 1000))
    cap_crop(img, 500)
    assert img.size == (2000, 1000)


def test_miss_path_is_capped(config, big_source: Path) -> None:
    config(crop=1024)
    jpeg = crop_bytes.crop_jpeg_from_disk(str(big_source), (0.0, 0.0, 1.0, 1.0))
    assert jpeg is not None
    assert _size(jpeg) == (1024, 768)


def test_hit_and_miss_have_identical_dimensions(config, big_source: Path) -> None:
    cache = config(crop=1024)
    bbox = (0.1, 0.1, 0.9, 0.8)
    miss = crop_bytes.crop_jpeg_from_disk(str(big_source), bbox)
    assert miss is not None

    src = Image.open(big_source)
    w, h = src.size
    from src.domain.image_geometry import crop_rect

    crop = src.crop(crop_rect((bbox[0] * w, bbox[1] * h, bbox[2] * w, bbox[3] * h), w, h))
    project_dir = cache / 'default'
    write_crop_cache('c1', crop, project_dir, max_side=1024)
    hit = crop_bytes.crop_jpeg_from_cache('c1')
    assert hit is not None
    assert _size(hit) == _size(miss)


def test_cap_zero_miss_bytes_are_todays_bytes(config, big_source: Path) -> None:
    config(crop=0)
    bbox = (0.2, 0.2, 0.6, 0.7)
    got = crop_bytes.crop_jpeg_from_disk(str(big_source), bbox)

    src = Image.open(big_source).convert('RGB')
    from src.domain.image_geometry import crop_rect

    w, h = src.size
    buf = io.BytesIO()
    src.crop(crop_rect((bbox[0] * w, bbox[1] * h, bbox[2] * w, bbox[3] * h), w, h)).save(
        buf, format='JPEG', quality=crop_bytes.CROP_JPEG_QUALITY
    )
    assert got == buf.getvalue()


def test_cap_zero_cache_write_bytes_are_todays_bytes(tmp_path: Path) -> None:
    crop = _photo(1500, 1000)
    write_crop_cache('a', crop, tmp_path)
    buf = io.BytesIO()
    crop.save(buf, format='JPEG', quality=85)
    from src.services.curation.source_image_cache import DEFAULT_CROP_CACHE_QUALITY

    expected = io.BytesIO()
    crop.save(expected, format='JPEG', quality=DEFAULT_CROP_CACHE_QUALITY)
    assert (tmp_path / 'a.jpg').read_bytes() == expected.getvalue()


def test_cache_write_is_capped(tmp_path: Path) -> None:
    write_crop_cache('a', _photo(4000, 3000), tmp_path, max_side=1024)
    assert _size((tmp_path / 'a.jpg').read_bytes()) == (1024, 768)


def test_a_large_crop_payload_is_smaller_with_the_cap(config, big_source: Path) -> None:
    config(crop=0)
    full = crop_bytes.crop_jpeg_from_disk(str(big_source), (0.0, 0.0, 1.0, 1.0))
    config(crop=1024)
    capped = crop_bytes.crop_jpeg_from_disk(str(big_source), (0.0, 0.0, 1.0, 1.0))
    assert full is not None
    assert capped is not None
    assert len(capped) < len(full) / 4


def test_vlm_payload_long_side_equals_the_cap_for_a_large_crop(
    config, big_source: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(crop_bytes, 'resolved_image_path', lambda p: Path(p))
    cache = config(vlm=640)
    miss = crop_bytes.load_vlm_item_jpeg('m', str(big_source), (0, 0, 1, 1), cache_dir=None)
    assert miss is not None
    assert max(_size(miss)) == 640

    write_crop_cache('h', _photo(2000, 1500), cache / 'default')
    hit = crop_bytes.load_vlm_item_jpeg('h', str(big_source), (0, 0, 1, 1), cache_dir=cache)
    assert hit is not None
    assert _size(hit) == (640, 480)


def test_vlm_cap_zero_keeps_the_built_in_size(
    config, big_source: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(crop_bytes, 'resolved_image_path', lambda p: Path(p))
    config(vlm=0)
    assert crop_bytes.vlm_crop_size() == crop_bytes.VLM_CROP_SIZE == 224
    out = crop_bytes.load_vlm_item_jpeg('m', str(big_source), (0, 0, 1, 1), cache_dir=None)
    assert out is not None
    assert _size(out) == (224, 168)


def test_crop_norm_to_source_norm_round_trip_is_unchanged() -> None:
    parent = (0.2, 0.1, 0.8, 0.7)
    region = (0.25, 0.5, 0.75, 1.0)
    got = crop_norm_to_source_norm(region, parent)
    assert got == pytest.approx((0.35, 0.4, 0.65, 0.7))
    inv = tuple(
        (g - parent[i % 2]) / (parent[2 + i % 2] - parent[i % 2]) for i, g in enumerate(got)
    )
    assert inv == pytest.approx(region)
