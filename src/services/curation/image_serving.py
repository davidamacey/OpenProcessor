"""Image-serving primitives for the curation labeling pipeline.

The labeler frontend needs two things:

1. Stream a source JPEG from disk (any mounted volume — NAS, NVM, object
   store passthrough, etc.), optionally downscaled.
2. Crop+resize a 128px thumbnail of a single item bbox, cached LRU
   (2000 entries ~50MB). Used by cluster grids.

K6: this module used to also burn bbox overlays into the full-resolution
source image server-side (the "expand to source" view). Every served
image is now the clean source render (resize + EXIF transpose + crop,
never a drawn box or label) — the frontend draws its own boxes from the
geometry ``GET {prefix}/crops/{id}/context`` already serves in
source-image-normalized coordinates. There is no overlay flag to opt
back into: the drawing code was removed, not hidden behind a query
param.

Why the API service and not a static file server: it is the only service
with both OpenSearch metadata and the source files mounted, so the crop
endpoint can derive the thumbnail from ``bbox_norm + JPEG`` without a
second hop.

The OpenSearch item lookup is intentionally factored into a private
``_fetch_crop()`` helper so unit tests can mock it easily.

Source-root configuration is data, not code — see
``src.config.curation.CurationConfig.source_root`` (the primary root)
and ``source_path_aliases`` (additional named roots, e.g. a secondary
archive or a training-corpus mount).
"""

from __future__ import annotations

import io
import os
from collections import OrderedDict
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, Any

from fastapi import HTTPException
from fastapi.responses import FileResponse
from PIL import Image, ImageOps

from src.config import CurationConfig, get_curation_config


if TYPE_CHECKING:
    from collections.abc import Sequence


# Use the structured logger when available; fall back to a structlog-
# compatible stdlib wrapper in slim test environments where the full
# ``src.core`` import chain (tritonclient, onnxruntime) isn't installed.
# Production always takes the first branch.
try:
    from src.core.logging import get_logger as _get_logger
except Exception:
    try:
        import structlog as _structlog

        def _get_logger(name: str) -> Any:  # type: ignore[misc]
            return _structlog.get_logger(name)
    except Exception:
        import logging as _stdlib_logging

        class _KwargLogger:
            """Stdlib logger that swallows structlog-style kwargs."""

            def __init__(self, name: str) -> None:
                self._inner = _stdlib_logging.getLogger(name)

            def _format(self, msg: str, **kwargs: Any) -> str:
                if not kwargs:
                    return msg
                extras = ' '.join(f'{k}={v!r}' for k, v in kwargs.items())
                return f'{msg} {extras}'

            def debug(self, msg: str, **kwargs: Any) -> None:
                self._inner.debug(self._format(msg, **kwargs))

            def info(self, msg: str, **kwargs: Any) -> None:
                self._inner.info(self._format(msg, **kwargs))

            def warning(self, msg: str, **kwargs: Any) -> None:
                self._inner.warning(self._format(msg, **kwargs))

            def error(self, msg: str, **kwargs: Any) -> None:
                self._inner.error(self._format(msg, **kwargs))

        def _get_logger(name: str) -> Any:  # type: ignore[misc]
            return _KwargLogger(name)


def get_logger(name: str) -> Any:
    """Module-local re-export so callers can patch this in tests."""
    return _get_logger(name)


logger = get_logger(__name__)


# =============================================================================
# Path Roots
# =============================================================================


def _configured_roots(config: CurationConfig | None = None) -> tuple[Path, ...]:
    """All configured source roots for a deployment.

    The primary ``source_root`` plus every ``source_path_aliases`` value.
    Consulted by :func:`resolve_safe_path`'s absolute-path branch and by
    :func:`resolve_crop_root` — anything outside this list is rejected /
    unmatched.
    """
    cfg = config or get_curation_config()
    # BA-1: the upload root is a server-managed root too -- an uploaded
    # item's persisted image_path must pass the same servability guard
    # as any mounted source root.
    return (cfg.source_root, cfg.upload_root, *cfg.source_path_aliases.values())


def is_servable_image_path(path: str) -> bool:
    """Whether the image server can later serve ``path`` as an item's
    ``image_path``: an absolute path under one of the configured source
    roots (:func:`_configured_roots`). Ingest refuses anything else so a
    stored item never points at an image no route can return."""
    if not path.startswith('/'):
        return False
    try:
        candidate = Path(path).resolve()
    except OSError:
        return False
    for root in _configured_roots():
        try:
            candidate.relative_to(root.resolve())
        except ValueError:
            continue
        return True
    return False


UNSERVABLE_PATH_ERROR = (
    'image path is not under a configured source root (OP_SOURCE_ROOT / '
    'OP_SOURCE_PATH_ALIASES), so its images could not be served'
)


def persist_uploaded_bytes(
    payload: bytes,
    *,
    imohash: str,
    extension: str,
    config: CurationConfig | None = None,
) -> Path:
    """Persist uploaded image bytes under ``CurationConfig.upload_root``,
    content-addressed (BA-1).

    Path shape: ``<upload_root>/<imohash[:2]>/<imohash><extension>`` —
    the same bytes always resolve to the same path, so re-uploading the
    same image is a no-op write (dedup by content hash) and the file is
    written atomically (temp file + ``os.replace``) so a concurrent
    reader never observes a partial file. Returns the absolute
    destination path; the caller stores this as the item's ``image_path``
    (it is guaranteed servable -- ``upload_root`` is one of
    :func:`_configured_roots`).
    """
    cfg = config or get_curation_config()
    root = cfg.upload_root
    shard = imohash[:2] if len(imohash) >= 2 else '00'
    dest_dir = root / shard
    dest = dest_dir / f'{imohash}{extension}'
    if dest.exists():
        return dest
    dest_dir.mkdir(parents=True, exist_ok=True)
    tmp = dest_dir / f'.{imohash}{extension}.{os.getpid()}.tmp'
    tmp.write_bytes(payload)
    tmp.replace(dest)
    return dest


# =============================================================================
# Path Resolution / Traversal Guard
# =============================================================================


def resolve_safe_path(
    rel_path: str,
    root: Path,
    *,
    allowed_roots: Sequence[Path] | None = None,
) -> Path:
    """Resolve a relative path under ``root`` and reject traversal attempts.

    Args:
        rel_path: Relative path under root. Forward or back slashes, but
            must NOT be absolute and must NOT contain ``..`` segments.
        root: Whitelisted root directory (must already be a real path).
        allowed_roots: Roots an *absolute* ``rel_path`` is permitted to
            resolve under. Defaults to the caller's configured
            ``CurationConfig`` roots (:func:`_configured_roots`).

    Returns:
        Absolute resolved path that is guaranteed to be within ``root``
        (relative input) or one of ``allowed_roots`` (absolute input).

    Raises:
        HTTPException(400): If the path is absolute, contains traversal
            segments, or resolves outside the whitelisted roots.
        HTTPException(404): If the resolved path does not exist.
    """
    if not rel_path:
        raise HTTPException(status_code=400, detail='path is required')

    normalized = rel_path.replace('\\', '/')

    # Absolute paths are accepted only when they fall under one of the
    # whitelisted roots (the configured source_root + source_path_aliases).
    # This is the path shape used by callers that store full absolute
    # paths in the item document's ``image_path``.
    if normalized.startswith('/'):
        allowed = tuple(allowed_roots) if allowed_roots is not None else _configured_roots()
        try:
            abs_candidate = Path(normalized).resolve()
        except OSError as exc:  # malformed input
            raise HTTPException(status_code=400, detail=f'invalid path: {exc}') from exc
        for whitelisted in allowed:
            try:
                abs_candidate.relative_to(whitelisted.resolve())
            except ValueError:
                continue
            if not abs_candidate.exists():
                raise HTTPException(status_code=404, detail=f'image not found: {rel_path}')
            return abs_candidate
        logger.warning('path_traversal_absolute', path=rel_path)
        raise HTTPException(status_code=400, detail='absolute paths not allowed')

    if normalized.startswith('~'):
        logger.warning('path_traversal_absolute', path=rel_path)
        raise HTTPException(status_code=400, detail='absolute paths not allowed')

    parts = [p for p in normalized.split('/') if p not in ('', '.')]
    if any(p == '..' for p in parts):
        logger.warning('path_traversal_dotdot', path=rel_path)
        raise HTTPException(status_code=400, detail='path traversal not allowed')

    # Resolve and confirm the result is still under root.
    root_resolved = root.resolve()
    candidate = (root_resolved / Path(*parts)).resolve()
    try:
        candidate.relative_to(root_resolved)
    except ValueError as exc:
        logger.warning(
            'path_traversal_escape', path=rel_path, resolved=str(candidate), root=str(root_resolved)
        )
        raise HTTPException(status_code=400, detail='path escapes root') from exc

    if not candidate.exists():
        raise HTTPException(status_code=404, detail=f'image not found: {rel_path}')

    return candidate


# =============================================================================
# Source-image streaming
# =============================================================================


async def serve_source_image(path: str, root: Path) -> FileResponse:
    """Stream a source JPEG via FastAPI ``FileResponse`` (zero-copy).

    Args:
        path: Relative path under ``root``.
        root: Whitelisted root directory.

    Returns:
        ``FileResponse`` with proper media-type and 1h Cache-Control.
    """
    safe_path = resolve_safe_path(path, root)

    # Best-effort media-type detection. Default JPEG.
    suffix = safe_path.suffix.lower()
    media_type = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.webp': 'image/webp',
        '.heic': 'image/heic',
        '.heif': 'image/heif',
    }.get(suffix, 'application/octet-stream')

    logger.debug('serve_source_image', path=path, resolved=str(safe_path), media_type=media_type)
    return FileResponse(
        path=str(safe_path),
        media_type=media_type,
        headers={'Cache-Control': 'public, max-age=3600'},
    )


# =============================================================================
# Thumbnail Cache (LRU 2000 entries)
# =============================================================================


# Try cachetools (preferred), fall back to a small thread-safe
# ``OrderedDict`` LRU. The fallback is enough for unit tests when the
# package is not installed in a slim test env.
try:
    from cachetools import LRUCache as _CachetoolsLRU  # type: ignore[import-untyped]

    _HAS_CACHETOOLS = True
except ImportError:  # pragma: no cover - fallback path
    _HAS_CACHETOOLS = False

    class _CachetoolsLRU(OrderedDict):  # type: ignore[no-redef]
        """Tiny stand-in mimicking ``cachetools.LRUCache`` semantics."""

        def __init__(self, maxsize: int) -> None:
            super().__init__()
            self.maxsize = maxsize

        def __getitem__(self, key: Any) -> Any:
            value = super().__getitem__(key)
            self.move_to_end(key)
            return value

        def __setitem__(self, key: Any, value: Any) -> None:
            super().__setitem__(key, value)
            self.move_to_end(key)
            while len(self) > self.maxsize:
                self.popitem(last=False)


THUMBNAIL_CACHE_MAX_ENTRIES = 2000  # ~50MB at 128x128 JPEG q=85.


class ThumbnailCache:
    """LRU cache of item-crop thumbnails (JPEG bytes).

    Key: ``(image_path, bbox_norm_tuple, size)``. Value: encoded JPEG
    bytes ready to stream. Hits/misses are tracked for the cache-stats
    diagnostics endpoint.
    """

    def __init__(self, maxsize: int = THUMBNAIL_CACHE_MAX_ENTRIES) -> None:
        self._cache: Any = _CachetoolsLRU(maxsize=maxsize)
        self._lock = Lock()
        self._hits = 0
        self._misses = 0
        self._maxsize = maxsize

    @staticmethod
    def _normalize_bbox(
        bbox_norm: tuple[float, ...] | list[float],
    ) -> tuple[float, float, float, float]:
        if len(bbox_norm) != 4:
            raise ValueError(f'bbox_norm must have 4 elements, got {len(bbox_norm)}')
        x1, y1, x2, y2 = (float(v) for v in bbox_norm)
        # Round to 6 decimals so floating-point noise doesn't fragment
        # the cache for what is morally the same bbox.
        return (round(x1, 6), round(y1, 6), round(x2, 6), round(y2, 6))

    def get_or_compute(
        self,
        image_path: Path,
        bbox_norm: tuple[float, ...] | list[float],
        size: int = 128,
    ) -> bytes:
        """Return cached or freshly-computed thumbnail for ``image_path``.

        Args:
            image_path: Absolute path to the source JPEG (must already be
                resolved+validated by ``resolve_safe_path``).
            bbox_norm: ``(x1, y1, x2, y2)`` normalized 0-1 coords.
            size: Square thumbnail size in pixels (default 128).

        Returns:
            JPEG-encoded bytes (q=85).
        """
        key = (str(image_path), self._normalize_bbox(bbox_norm), int(size))
        with self._lock:
            cached = self._cache.get(key) if _HAS_CACHETOOLS else None
            if cached is None and not _HAS_CACHETOOLS and key in self._cache:
                # OrderedDict fallback path — use ``in`` instead of get.
                cached = self._cache[key]
            if cached is not None:
                self._hits += 1
                return cached

        # Miss — compute outside the lock so concurrent requests for
        # different keys can run in parallel.
        thumb_bytes = self._render_thumbnail(image_path, bbox_norm, size)

        with self._lock:
            self._cache[key] = thumb_bytes
            self._misses += 1
        return thumb_bytes

    @staticmethod
    def _render_thumbnail(
        image_path: Path,
        bbox_norm: tuple[float, ...] | list[float],
        size: int,
    ) -> bytes:
        """Open, exif-rotate, crop, resize, JPEG-encode."""
        with Image.open(image_path) as src:
            img = ImageOps.exif_transpose(src)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            w, h = img.size
            x1 = max(0, int(bbox_norm[0] * w))
            y1 = max(0, int(bbox_norm[1] * h))
            x2 = min(w, int(bbox_norm[2] * w))
            y2 = min(h, int(bbox_norm[3] * h))
            if x2 <= x1 or y2 <= y1:
                raise ValueError(
                    f'invalid bbox after pixel mapping: ({x1},{y1},{x2},{y2}) for {image_path}'
                )
            # PIL.Image.thumbnail() preserves aspect ratio — fits the largest
            # dimension to ``size`` and scales the other proportionally. The
            # frontend pairs this with object-contain + neutral letterbox so a
            # 200x600 tall crop displays at 43x128 inside a 128x128 cell
            # instead of being squashed to a 128x128 square.
            crop = img.crop((x1, y1, x2, y2))
            crop.thumbnail((size, size), Image.LANCZOS)
            buf = io.BytesIO()
            crop.save(buf, format='JPEG', quality=85)
            return buf.getvalue()

    def cache_info(self) -> dict[str, int]:
        """Return cache stats for the diagnostics endpoint."""
        with self._lock:
            return {
                'hits': self._hits,
                'misses': self._misses,
                'current_size': len(self._cache),
                'max_size': self._maxsize,
            }

    def clear(self) -> None:
        """Drop all cached thumbnails (test helper)."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0


# Module-level singleton used by the router. The router imports this
# directly so the cache survives across requests.
THUMBNAIL_CACHE = ThumbnailCache()


# =============================================================================
# Clean source-image rendering (K6: no server-side overlays)
# =============================================================================


def _maybe_downscale(img: Image.Image, max_dim: int | None) -> Image.Image:
    """Resize ``img`` so its longest edge is <= ``max_dim`` (Lanczos).

    For full-res sources (e.g. 5472x3648) Lanczos on the decoded buffer
    costs ~600 ms by itself. Callers that open the image with
    ``Image.draft('RGB', (max_dim, max_dim))`` before calling here pay
    a fraction of that because the JPEG decoder hands back a 4x or 8x
    pre-scaled buffer; this function then tightens it the rest of the
    way.
    """
    if max_dim is None or max_dim <= 0:
        return img
    w, h = img.size
    longest = max(w, h)
    if longest <= max_dim:
        return img
    scale = max_dim / longest
    new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
    return img.resize(new_size, Image.LANCZOS)


async def render_source_image(
    image_path: Path,
    max_dim: int | None = None,
) -> bytes:
    """Render the clean source image as JPEG bytes: EXIF-transpose,
    RGB-convert, optionally downscale, re-encode. No box, label or other
    overlay is ever drawn (K6) — the frontend draws every box itself from
    the geometry ``GET {prefix}/crops/{id}/context`` serves.

    Pass ``max_dim`` to cap the longest side so the labeler can pull a
    review-friendly preview (~1024px) instead of the full-resolution
    source — the JPEG drops from ~2.2 MB to ~200 KB.
    """
    with Image.open(image_path) as src:
        # JPEG draft mode lets the decoder return a pre-scaled buffer
        # directly — for a 5472x3648 source asked to fit in 1280px the
        # decoder hands back a 1368x912 image without a separate resize
        # pass. ``_maybe_downscale`` then tightens it to the exact cap.
        if max_dim and src.format == 'JPEG':
            src.draft('RGB', (max_dim, max_dim))
        img = ImageOps.exif_transpose(src)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = _maybe_downscale(img, max_dim)
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=88)
        return buf.getvalue()


# =============================================================================
# OpenSearch crop fetcher (stub-friendly)
# =============================================================================


def _crop_source_includes() -> list[str]:
    """``_source_includes`` for :func:`_fetch_crop`.

    Every field any ``crops_router`` route actually reads off the
    returned doc: ``image_path`` (all three routes), ``bbox_norm``
    (``crop_thumbnail``), and the region-of-interest bbox field plus the
    verifier-rejected candidate bbox (both ``crop_region_thumbnail`` --
    the candidate bbox is its fallback for a ``verify_rejected`` item,
    which never has the region bbox field). K6: ``crop_full_image`` used
    to also read ``bbox_norm``/``class_name``/the region bbox to draw a
    server-side overlay; it now only resolves ``image_path`` and serves
    the clean source, so ``class_name`` is no longer read by anything
    here. Without this narrowed list, a bare ``.get()`` also decompresses
    the item's embedding vectors + nested history JSON, none of which any
    caller reads (see F-14).
    """
    from src.config import get_region_fields

    fields = get_region_fields()
    return ['image_path', 'bbox_norm', fields.bbox_norm, fields.candidate_bbox_norm]


async def _fetch_crop(
    crop_id: str,
    opensearch_client: Any,
    config: CurationConfig | None = None,
) -> dict[str, Any]:
    """Fetch a single item document from the configured items index.

    We expect at least these fields:

    - ``image_path``: relative (or whitelisted-absolute) path under one
      of the configured source roots
    - ``bbox_norm``: ``[x1, y1, x2, y2]`` normalized 0-1
    - region-of-interest bbox (optional): see
      ``src.config.region_fields.RegionFields.bbox_norm``

    Fetches only :func:`_crop_source_includes` via ``_source_includes``
    — embedding vectors and history arrays are never read by any
    ``crops_router`` route, so there's no reason to decompress them on
    every thumbnail/image request (F-14).

    Tests mock this function; production wires the real OpenSearch
    client through the router dependency.

    Raises:
        HTTPException(404) if the crop does not exist.
    """
    cfg = config or get_curation_config()
    try:
        # ``AsyncOpenSearch.get`` raises ``opensearchpy.NotFoundError`` if
        # the doc is missing. We avoid a hard import on opensearchpy here
        # so unit tests can pass a plain mock with ``.get``.
        result = await opensearch_client.get(
            index=cfg.items_index,
            id=crop_id,
            _source_includes=_crop_source_includes(),
        )
    except Exception as exc:
        msg = str(exc).lower()
        if 'notfound' in msg or 'not found' in msg or '404' in msg:
            raise HTTPException(status_code=404, detail=f'crop not found: {crop_id}') from exc
        logger.error('fetch_crop_failed', crop_id=crop_id, error=str(exc))
        raise HTTPException(status_code=500, detail=f'crop lookup failed: {exc}') from exc

    source = result.get('_source') if isinstance(result, dict) else None
    if not source:
        raise HTTPException(status_code=404, detail=f'crop has no source: {crop_id}')
    return source


def resolve_crop_root(image_path: str, config: CurationConfig | None = None) -> Path:
    """Pick the right filesystem root for a stored ``image_path``.

    For absolute paths, the longest matching configured root wins (the
    primary ``source_root`` or a ``source_path_aliases`` entry). For
    relative paths, the leading path segment is matched (exact or as a
    prefix) against ``source_path_aliases`` keys; anything unmatched
    falls back to ``source_root``.
    """
    cfg = config or get_curation_config()
    normalized = image_path.replace('\\', '/')

    if normalized.startswith('/'):
        try:
            abs_candidate = Path(normalized).resolve()
        except OSError:
            return cfg.source_root
        # Pick the most specific (longest) matching root.
        best: Path | None = None
        for root in _configured_roots(cfg):
            try:
                abs_candidate.relative_to(root.resolve())
            except ValueError:
                continue
            if best is None or len(str(root)) > len(str(best)):
                best = root
        return best if best is not None else cfg.source_root

    head = normalized.lstrip('/').split('/', 1)[0].lower()
    for alias, alias_root in cfg.source_path_aliases.items():
        alias_lower = alias.lower()
        if head == alias_lower or head.startswith(alias_lower):
            return alias_root
    return cfg.source_root
