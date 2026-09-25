"""Unit tests for the curation image-serving service.

Covered:
- Path-traversal rejection (``..``, absolute, escapes-root, NT-style backslash)
- ``resolve_crop_root``'s configured-alias matching (absolute + relative)
- Thumbnail LRU cache hit/miss accounting
- Bbox overlay produces a real JPEG (magic-byte check)
- Multiple-bbox overlay tolerates empty list and labels
- ``_fetch_crop`` against a mocked OpenSearch client

These tests are pure-Python; they don't need triton-api running.
"""

from __future__ import annotations

import asyncio

# Module under test. Import the leaf module directly so collection
# doesn't trigger ``src.services.__init__`` (which pulls Triton client
# deps that aren't installed in slim test environments).
import importlib.util as _ilu
import io
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from fastapi import HTTPException
from PIL import Image

from src.config import CurationConfig, get_curation_config, get_region_fields


_spec = _ilu.spec_from_file_location(
    'image_serving',
    Path(__file__).resolve().parent.parent.parent
    / 'src'
    / 'services'
    / 'curation'
    / 'image_serving.py',
)
assert _spec is not None
assert _spec.loader is not None
svc: Any = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(svc)


# =============================================================================
# Fixtures
# =============================================================================


FIXTURE_DIR = Path(__file__).parent.parent / 'fixtures'
FIXTURE_IMAGE = FIXTURE_DIR / 'sample_image.jpg'


def _ensure_sample_image() -> Path:
    """Create a tiny synthetic JPEG once for the test session."""
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    if not FIXTURE_IMAGE.exists():
        img = Image.new('RGB', (640, 480), color=(40, 80, 120))
        # Add a contrasting square so crops aren't pure-color noise.
        for x in range(160, 320):
            for y in range(120, 280):
                img.putpixel((x, y), (255, 200, 50))
        img.save(FIXTURE_IMAGE, format='JPEG', quality=88)
    return FIXTURE_IMAGE


@pytest.fixture(scope='module')
def sample_image() -> Path:
    return _ensure_sample_image()


@pytest.fixture
def fresh_cache() -> svc.ThumbnailCache:
    return svc.ThumbnailCache(maxsize=8)


# =============================================================================
# Path traversal
# =============================================================================


@pytest.mark.parametrize(
    'bad_path',
    [
        '../etc/passwd',
        '../../etc/passwd',
        '/etc/passwd',
        # This is the one proprietary-absolute-path site in the whole
        # reference test corpus (see plan §0.8) — a path-traversal
        # REJECTION fixture, not a data dependency, so it is trivially
        # portable by swapping in the configured source_root instead of
        # the hardcoded proprietary root.
        f'{get_curation_config().source_root}/../../../etc/passwd',
        '..\\windows\\system32\\config',
        'hdd01/../../etc/passwd',
        'foo/bar/../../../../../etc/passwd',
        '~/.ssh/id_rsa',
    ],
)
def test_resolve_safe_path_rejects_traversal(tmp_path: Path, bad_path: str) -> None:
    with pytest.raises(HTTPException) as exc:
        svc.resolve_safe_path(bad_path, tmp_path)
    assert exc.value.status_code == 400


def test_resolve_safe_path_rejects_empty(tmp_path: Path) -> None:
    with pytest.raises(HTTPException) as exc:
        svc.resolve_safe_path('', tmp_path)
    assert exc.value.status_code == 400


def test_resolve_safe_path_404_for_missing(tmp_path: Path) -> None:
    with pytest.raises(HTTPException) as exc:
        svc.resolve_safe_path('does_not_exist.jpg', tmp_path)
    assert exc.value.status_code == 404


def test_resolve_safe_path_accepts_clean_relative(tmp_path: Path, sample_image: Path) -> None:
    # Place the sample under a sub-folder of tmp_path.
    target = tmp_path / 'sub' / 'img.jpg'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(sample_image.read_bytes())

    resolved = svc.resolve_safe_path('sub/img.jpg', tmp_path)
    assert resolved == target.resolve()


def test_resolve_safe_path_handles_backslashes(tmp_path: Path, sample_image: Path) -> None:
    target = tmp_path / 'sub' / 'img.jpg'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(sample_image.read_bytes())

    resolved = svc.resolve_safe_path('sub\\img.jpg', tmp_path)
    assert resolved == target.resolve()


def test_resolve_safe_path_absolute_allowed_under_configured_root(
    tmp_path: Path, sample_image: Path
) -> None:
    root = tmp_path / 'root'
    root.mkdir()
    target = root / 'img.jpg'
    target.write_bytes(sample_image.read_bytes())

    cfg = CurationConfig(source_root=root)
    resolved = svc.resolve_safe_path(
        str(target), tmp_path / 'unused', allowed_roots=svc._configured_roots(cfg)
    )
    assert resolved == target.resolve()


def test_resolve_safe_path_absolute_outside_configured_roots_rejected(tmp_path: Path) -> None:
    cfg = CurationConfig(source_root=tmp_path / 'root')
    with pytest.raises(HTTPException) as exc:
        svc.resolve_safe_path(
            '/etc/passwd', tmp_path / 'unused', allowed_roots=svc._configured_roots(cfg)
        )
    assert exc.value.status_code == 400


# =============================================================================
# resolve_crop_root
# =============================================================================


def test_resolve_crop_root_matches_alias_prefix(tmp_path: Path) -> None:
    archive_root = tmp_path / 'archive'
    nightly_root = tmp_path / 'nightly'
    fallback_root = tmp_path / 'fallback'
    cfg = CurationConfig(
        source_root=fallback_root,
        source_path_aliases={'archive': archive_root, 'nightly': nightly_root},
    )

    assert svc.resolve_crop_root('archive/foo/bar.jpg', cfg) == archive_root
    # Prefix match: leading segment merely starts with the alias name.
    assert svc.resolve_crop_root('archive_2024/x.jpg', cfg) == archive_root
    assert svc.resolve_crop_root('nightly/x/y.jpg', cfg) == nightly_root


def test_resolve_crop_root_falls_back_to_source_root(tmp_path: Path) -> None:
    cfg = CurationConfig(
        source_root=tmp_path / 'fallback',
        source_path_aliases={'archive': tmp_path / 'archive'},
    )
    assert svc.resolve_crop_root('train/images/00001.jpg', cfg) == cfg.source_root
    assert svc.resolve_crop_root('val/images/00001.jpg', cfg) == cfg.source_root


def test_resolve_crop_root_absolute_path_longest_match_wins(tmp_path: Path) -> None:
    outer = tmp_path / 'outer'
    inner = outer / 'inner'
    inner.mkdir(parents=True)
    cfg = CurationConfig(
        source_root=outer,
        source_path_aliases={'inner_alias': inner},
    )
    assert svc.resolve_crop_root(str(inner / 'x.jpg'), cfg) == inner


def test_resolve_crop_root_uses_module_default_when_no_config_passed() -> None:
    default_cfg = get_curation_config()
    assert svc.resolve_crop_root('anything/x.jpg') == default_cfg.source_root


# =============================================================================
# Thumbnail cache hit/miss
# =============================================================================


def test_thumbnail_cache_miss_then_hit(sample_image: Path, fresh_cache: svc.ThumbnailCache) -> None:
    bbox = (0.25, 0.25, 0.5, 0.5)
    info = fresh_cache.cache_info()
    assert info == {'hits': 0, 'misses': 0, 'current_size': 0, 'max_size': 8}

    first = fresh_cache.get_or_compute(sample_image, bbox, size=64)
    assert first.startswith(b'\xff\xd8'), 'must be JPEG magic bytes'
    info1 = fresh_cache.cache_info()
    assert info1['misses'] == 1
    assert info1['hits'] == 0
    assert info1['current_size'] == 1

    second = fresh_cache.get_or_compute(sample_image, bbox, size=64)
    assert second is first  # exact bytes returned from cache
    info2 = fresh_cache.cache_info()
    assert info2['hits'] == 1
    assert info2['misses'] == 1


def test_thumbnail_cache_distinguishes_size(
    sample_image: Path, fresh_cache: svc.ThumbnailCache
) -> None:
    bbox = (0.25, 0.25, 0.5, 0.5)
    fresh_cache.get_or_compute(sample_image, bbox, size=64)
    fresh_cache.get_or_compute(sample_image, bbox, size=128)
    info = fresh_cache.cache_info()
    assert info['misses'] == 2
    assert info['current_size'] == 2


def test_thumbnail_cache_distinguishes_bbox(
    sample_image: Path, fresh_cache: svc.ThumbnailCache
) -> None:
    fresh_cache.get_or_compute(sample_image, (0.0, 0.0, 0.5, 0.5), size=64)
    fresh_cache.get_or_compute(sample_image, (0.5, 0.5, 1.0, 1.0), size=64)
    info = fresh_cache.cache_info()
    assert info['misses'] == 2
    assert info['current_size'] == 2


def test_thumbnail_cache_lru_eviction(sample_image: Path) -> None:
    cache = svc.ThumbnailCache(maxsize=3)
    for i in range(5):
        cache.get_or_compute(sample_image, (0.0, 0.0, 0.1 + 0.05 * i, 0.5))
    info = cache.cache_info()
    assert info['current_size'] == 3
    assert info['misses'] == 5


def test_thumbnail_cache_rejects_bad_bbox_length(
    sample_image: Path, fresh_cache: svc.ThumbnailCache
) -> None:
    with pytest.raises(ValueError, match='4 elements'):
        fresh_cache.get_or_compute(sample_image, (0.0, 0.0, 1.0))  # type: ignore[arg-type]


def test_thumbnail_cache_rejects_zero_area_bbox(
    sample_image: Path, fresh_cache: svc.ThumbnailCache
) -> None:
    with pytest.raises(ValueError, match=r'.'):
        fresh_cache.get_or_compute(sample_image, (0.5, 0.5, 0.5, 0.5))


def test_thumbnail_cache_uses_pil_correctly(
    sample_image: Path, fresh_cache: svc.ThumbnailCache
) -> None:
    """Mock-based check that the PIL pipeline is invoked once per miss."""
    real_open = Image.open
    call_count = {'n': 0}

    def counting_open(*args: object, **kwargs: object) -> Image.Image:
        call_count['n'] += 1
        return real_open(*args, **kwargs)

    with patch.object(Image, 'open', side_effect=counting_open):
        fresh_cache.get_or_compute(sample_image, (0.1, 0.1, 0.6, 0.6))
        fresh_cache.get_or_compute(sample_image, (0.1, 0.1, 0.6, 0.6))  # hit
        fresh_cache.get_or_compute(sample_image, (0.2, 0.2, 0.7, 0.7))  # miss

    assert call_count['n'] == 2  # one per unique bbox


# =============================================================================
# K6: served images are the clean source render, never an overlay
# =============================================================================


def test_render_source_image_returns_jpeg(sample_image: Path) -> None:
    jpeg = asyncio.run(svc.render_source_image(sample_image))
    assert jpeg.startswith(b'\xff\xd8'), 'must be JPEG magic bytes'
    assert jpeg.endswith(b'\xff\xd9'), 'must be JPEG end-of-image marker'


def test_render_source_image_is_pixel_identical_to_a_clean_manual_render(
    sample_image: Path,
) -> None:
    """K6: `render_source_image` must never draw a box/label — its output
    must be pixel-identical to the same EXIF-transpose+RGB-convert+encode
    pipeline applied with no drawing step at all."""
    from PIL import Image, ImageOps

    jpeg = asyncio.run(svc.render_source_image(sample_image))
    got = Image.open(io.BytesIO(jpeg))

    with Image.open(sample_image) as src:
        expected_img = ImageOps.exif_transpose(src)
        if expected_img.mode != 'RGB':
            expected_img = expected_img.convert('RGB')
        buf = io.BytesIO()
        expected_img.save(buf, format='JPEG', quality=88)
        expected = Image.open(io.BytesIO(buf.getvalue()))

    assert got.size == expected.size
    assert list(got.getdata()) == list(expected.getdata())


def test_render_source_image_has_no_overlay_drawing_helpers_left() -> None:
    """K6: the drawing helpers themselves must be gone, not just unused —
    guards against a future caller quietly re-wiring an overlay back in."""
    for name in (
        'render_image_with_bbox',
        'render_image_with_multiple_bboxes',
        '_load_label_font',
        '_DEFAULT_BBOX_COLOR',
        '_BBOX_LINE_WIDTH',
        '_LABEL_BG_COLOR',
        '_LABEL_TEXT_COLOR',
    ):
        assert not hasattr(svc, name), f'{name} should have been removed (K6)'


def test_render_source_image_downscales_with_max_dim(sample_image: Path) -> None:
    jpeg = asyncio.run(svc.render_source_image(sample_image, max_dim=8))
    from PIL import Image

    img = Image.open(io.BytesIO(jpeg))
    assert max(img.size) <= 8


# =============================================================================
# _fetch_crop with mocked OpenSearch
# =============================================================================


class _FakeOSClient:
    def __init__(self, source: dict | None = None, raise_not_found: bool = False) -> None:
        self._source = source
        self._raise_not_found = raise_not_found
        self.last_call: dict | None = None

    async def get(
        self,
        index: str,
        id: str,  # noqa: A002 - test stub mirrors OpenSearch signature
        _source_includes: list[str] | None = None,
    ) -> dict:
        self.last_call = {'index': index, 'id': id, '_source_includes': _source_includes}
        if self._raise_not_found:
            err = Exception(f'NotFoundError: no such crop {id}')
            raise err
        return {'_source': self._source}


def test_fetch_crop_returns_source() -> None:
    src = {'image_path': 'archive/foo.jpg', 'bbox_norm': [0.1, 0.1, 0.5, 0.5]}
    client = _FakeOSClient(source=src)
    result = asyncio.run(svc._fetch_crop('crop_1', client))
    assert result == src


def test_fetch_crop_404_when_missing() -> None:
    client = _FakeOSClient(raise_not_found=True)
    with pytest.raises(HTTPException) as exc:
        asyncio.run(svc._fetch_crop('crop_404', client))
    assert exc.value.status_code == 404


def test_fetch_crop_500_when_no_source() -> None:
    client = _FakeOSClient(source=None)
    with pytest.raises(HTTPException) as exc:
        asyncio.run(svc._fetch_crop('crop_x', client))
    assert exc.value.status_code == 404


def test_fetch_crop_uses_configured_items_index() -> None:
    """The literal 'legacy_vehicle_crops' fallback (reference lines 542-544)
    must not exist here — the index name always comes from CurationConfig."""
    seen: dict[str, str] = {}

    class _RecordingOS:
        async def get(self, index: str, id: str, **kwargs: object) -> dict:  # noqa: A002, ARG002
            seen['index'] = index
            return {'_source': {'image_path': 'x.jpg'}}

    cfg = CurationConfig(items_index='custom_items_index')
    asyncio.run(svc._fetch_crop('crop_1', _RecordingOS(), config=cfg))
    assert seen['index'] == 'custom_items_index'


def test_fetch_crop_uses_source_includes_covering_every_caller_field() -> None:
    """F-14: no embeddings/history round-trip — only the fields any
    ``crops_router`` route actually reads off the returned doc."""
    src = {'image_path': 'archive/foo.jpg', 'bbox_norm': [0.1, 0.1, 0.5, 0.5]}
    client = _FakeOSClient(source=src)
    asyncio.run(svc._fetch_crop('crop_1', client))
    assert client.last_call is not None
    includes = client.last_call['_source_includes']
    assert includes is not None
    # image_path: all three crops_router routes.
    # bbox_norm: crop_thumbnail.
    # region bbox field + candidate bbox field: crop_region_thumbnail.
    # K6: crop_full_image no longer reads bbox_norm/class_name/region bbox
    # (it only resolves image_path and serves the clean source) so
    # class_name is deliberately NOT in this list any more.
    for field in (
        'image_path',
        'bbox_norm',
        get_region_fields().bbox_norm,
        get_region_fields().candidate_bbox_norm,
    ):
        assert field in includes, f'{field!r} missing from _source_includes: {includes}'
    assert 'class_name' not in includes


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
