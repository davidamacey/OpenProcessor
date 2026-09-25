"""Shared helpers for the public sample-dataset fetchers (G-05).

Both ``fetch_coco_subset.py`` and ``fetch_openimages_plates.py`` need the
same primitives: a resumable/verified HTTP download, SHA-256 checking, a
deterministic (seeded) per-class sample, and CSV/JSON manifest writers.
Kept here so neither fetcher duplicates network or hashing code.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import random
import time
import urllib.error
import urllib.request
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from pathlib import Path

logger = logging.getLogger('datasets.fetch')

USER_AGENT = 'OpenProcessor-sample-fetcher/1 (+https://github.com/davidamacey/OpenProcessor)'


class FetchError(RuntimeError):
    """Raised for anything that should abort the fetch with a clear message."""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path, *, expected_sha256: str | None = None, retries: int = 3) -> Path:
    """Download ``url`` to ``dest`` (atomically, via a ``.part`` temp file).

    Idempotent: if ``dest`` already exists and (when given) matches
    ``expected_sha256``, the download is skipped entirely -- repeated runs
    of the fetchers don't re-pull hundreds of images that are already on
    disk. A checksum mismatch on an existing file re-downloads once rather
    than trusting a partial/corrupt local copy.
    """
    if dest.is_file() and dest.stat().st_size > 0:
        if expected_sha256 is None or sha256_file(dest) == expected_sha256:
            return dest
        logger.warning('sha256 mismatch for existing %s; re-downloading', dest.name)
        dest.unlink()

    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + '.part')
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': USER_AGENT})
            with urllib.request.urlopen(req, timeout=60) as resp, part.open('wb') as out:
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    out.write(chunk)
            if expected_sha256 is not None:
                got = sha256_file(part)
                if got != expected_sha256:
                    raise FetchError(
                        f'{url}: sha256 mismatch (expected {expected_sha256}, got {got})'
                    )
            part.replace(dest)
            return dest
        except (urllib.error.URLError, OSError, FetchError) as exc:
            last_exc = exc
            part.unlink(missing_ok=True)
            if attempt < retries:
                logger.warning('download failed (%s/%s) for %s: %s', attempt, retries, url, exc)
                time.sleep(min(2**attempt, 10))
    raise FetchError(f'failed to download {url} after {retries} attempts') from last_exc


def seeded_sample(pool: Sequence[Any], n: int, seed: int) -> list[Any]:
    """A deterministic sample of up to ``n`` items from ``pool``.

    ``pool`` must already be in a stable order (callers sort by id first) --
    ``random.Random(seed).sample`` is order-sensitive, so the same pool
    order + seed always yields the same subset, but a differently-ordered
    equal pool would not. Returns the whole pool (in its given order) when
    it has fewer than ``n`` items.
    """
    if len(pool) <= n:
        return list(pool)
    return random.Random(seed).sample(list(pool), n)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def load_pinned_manifest(path: Path) -> list[dict[str, Any]] | None:
    """The committed manifest at ``path``, or ``None`` if it doesn't exist
    yet (first run: the caller computes selection and writes it as the
    new pin)."""
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(data, list):
        raise FetchError(f'{path}: expected a JSON list of manifest entries')
    return data
