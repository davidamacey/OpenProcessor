"""Ingest helpers: the image fingerprint, the upright decode and the item id.

Split out of :mod:`src.services.curation.ingest` so the service module holds the
orchestration only.
"""

from __future__ import annotations

import hashlib
import io

from src.config.env import env_int
from src.services.curation.image_decode import DecodedImage, decode_image
from src.services.detection.geometry import crop_id as _crop_id_fn
from src.utils.stage_timing import stage_timer


# Per-batch in-flight `ingest_one` calls. Triton's own dynamic batching
# coalesces the actual GPU work; this cap just keeps the request
# pipeline from swamping Triton's queue. Override with
# OP_MAX_INGEST_CONCURRENCY (no rebuild required).
MAX_INGEST_CONCURRENCY = env_int('OP_MAX_INGEST_CONCURRENCY', 16)


def _imohash_bytes(data: bytes) -> str:
    """Compute a fingerprint for raw image bytes, used for dedup.

    Uses ``imohash`` if available; otherwise falls back to a sha256
    prefix (slim test environments only — production installs imohash).
    """
    with stage_timer('hash', nbytes=len(data)):
        try:
            import imohash

            return imohash.hashfileobject(io.BytesIO(data)).hex()
        except Exception:
            return hashlib.sha256(data).hexdigest()[:32]


def _decode_image(image_bytes: bytes) -> DecodedImage:
    """Decode bytes into the upright RGB frame (:func:`~src.services.curation.image_decode.decode_image`)."""
    with stage_timer('decode', nbytes=len(image_bytes)):
        return decode_image(image_bytes)


def _crop_id(image_id: str, bbox_norm: list[float]) -> str:
    """Stable item id = sha256(image_id + bbox)[:32].

    Delegates to :func:`src.services.detection.geometry.crop_id` — the
    same function :mod:`src.services.curation.label_import` uses, so
    detector-created and label-created items for the same (image, bbox)
    always collide onto the same document.
    """
    return _crop_id_fn(image_id, bbox_norm)
