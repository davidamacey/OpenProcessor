"""Exact and perceptual hashes with their timings (nd:P2).

* SHA-256 of the file bytes (``hashlib``), ``imohash`` (the package the product uses for its exact key),
* PDQ (``pdqhash``, 256 bits), pHash and dHash (``ImageHash``, 64 bits each), packed into ``uint64`` words
  so a Hamming distance is ``popcount(a ^ b)``.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image, ImageOps


if TYPE_CHECKING:
    from collections.abc import Callable


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def imohash_file(path: str | Path) -> str:
    import imohash

    return str(imohash.hashfile(str(path), hexdigest=True))


def decode_rgb(path: str | Path) -> Image.Image:
    with Image.open(path) as img:
        return ImageOps.exif_transpose(img).convert('RGB')


def _pack(bits: np.ndarray) -> np.ndarray:
    flat = np.asarray(bits, dtype=np.uint8).reshape(-1)
    return np.packbits(flat).view(np.uint64)


def pdq_bits(rgb: np.ndarray) -> tuple[np.ndarray, int]:
    """``(4 x uint64, quality)`` of an HWC uint8 RGB array."""
    import pdqhash

    vec, quality = pdqhash.compute(rgb)
    return _pack(vec), int(quality)


def phash_bits(img: Image.Image) -> np.ndarray:
    import imagehash

    return _pack(imagehash.phash(img).hash)


def dhash_bits(img: Image.Image) -> np.ndarray:
    import imagehash

    return _pack(imagehash.dhash(img).hash)


def all_hashes(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """PDQ (4 words), pHash (1), dHash (1) and the PDQ quality of one file."""
    img = decode_rgb(path)
    pdq, quality = pdq_bits(np.asarray(img))
    return pdq, phash_bits(img), dhash_bits(img), quality


def hamming_pairs(
    words: np.ndarray, max_dist: int, block: int = 256
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """All pairs ``i < j`` with ``popcount(w_i ^ w_j) <= max_dist``; ``words`` is ``(n, k)`` uint64."""
    n = words.shape[0]
    xi, xj, xd = [], [], []
    for r0 in range(0, n, block):
        part = words[r0 : r0 + block]
        dist = np.bitwise_count(part[:, None, :] ^ words[None, :, :]).sum(axis=2)
        rows, cols = np.nonzero(dist <= max_dist)
        keep = cols > rows + r0
        xi.append(rows[keep] + r0)
        xj.append(cols[keep])
        xd.append(dist[rows[keep], cols[keep]])
    if not xi:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty, empty
    return np.concatenate(xi), np.concatenate(xj), np.concatenate(xd)


def timed(fn: Callable[..., object], *args: object, repeat: int = 3) -> tuple[float, float, float]:
    """``(median_ms, min_ms, max_ms)`` of ``repeat`` calls of ``fn(*args)``."""
    t = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn(*args)
        t.append((time.perf_counter() - t0) * 1000.0)
    t.sort()
    return t[len(t) // 2], t[0], t[-1]
