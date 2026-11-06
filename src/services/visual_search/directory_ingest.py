"""Bulk ingest of a server-side directory: the body of the ``directory_ingest`` job.

The job takes the images of one directory, reads them a batch at a time and hands each batch to
``VisualSearchService.ingest_batch``. A re-run of the same directory stores nothing twice: every
global document is keyed by the image's path relative to the directory (a repeat overwrites it),
and with ``skip_duplicates`` the batch drops images whose content hash is already indexed
(``ingest_batch.py``, ``check_duplicates_by_hash_batch``).
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from opensearchpy.exceptions import OpenSearchException
from tritonclient.utils import InferenceServerException

from src.domain.job_perf import RESULT_KEY, JobPerfSummary


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Mapping

logger = logging.getLogger(__name__)

# What a batch raises when Triton or OpenSearch fails; anything else is a bug and fails the job.
_BATCH_ERRORS = (OpenSearchException, InferenceServerException, OSError, TimeoutError)

IMAGE_EXTENSIONS = frozenset({'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff', '.tif'})
INDEXED_KEYS = ('global', 'vehicles', 'people', 'faces', 'ocr')


class DirectoryIngestError(ValueError):
    """The directory cannot be ingested; the message says why (answered 400 by the route)."""


@dataclass(frozen=True)
class DirectoryIngestOptions:
    directory: str
    recursive: bool = True
    max_images: int | None = None
    batch_size: int = 64
    skip_duplicates: bool = True
    detect_near_duplicates: bool = True
    near_duplicate_threshold: float = 0.99
    enable_ocr: bool = True
    enable_detection: bool = True
    enable_faces: bool = True
    enable_clip: bool = True
    defer_heavy_ops: bool = False

    def to_params(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_params(cls, params: Mapping[str, Any]) -> DirectoryIngestOptions:
        return cls(**dict(params))


def check_directory(directory: str) -> Path:
    """The directory as a ``Path``, or :class:`DirectoryIngestError` naming what is wrong."""
    path = Path(directory)
    if not path.exists():
        raise DirectoryIngestError(f'Directory not found: {directory}')
    if not path.is_dir():
        raise DirectoryIngestError(f'Path is not a directory: {directory}')
    return path


def collect_images(
    root: Path, *, recursive: bool, max_images: int | None
) -> tuple[list[Path], list[str]]:
    """The image files in a stable order (so a resumed job sees the same list), and the
    extensions of the files that were not images. Blocking: call it off the event loop."""
    found = root.rglob('*') if recursive else root.glob('*')
    files = sorted(f for f in found if f.is_file())
    images = [f for f in files if f.suffix.lower() in IMAGE_EXTENSIONS]
    skipped = sorted({f.suffix.lower() for f in files if f.suffix.lower() not in IMAGE_EXTENSIONS})
    if max_images and len(images) > max_images:
        images = images[:max_images]
    return images, skipped


def _read_batch(root: Path, files: list[Path]) -> tuple[list[tuple[bytes, str, str]], int]:
    """``(image bytes, image id, path)`` per readable file, and how many could not be read."""
    data: list[tuple[bytes, str, str]] = []
    errors = 0
    for file_path in files:
        try:
            image_bytes = file_path.read_bytes()
        except OSError as exc:
            logger.warning('Failed to read %s: %s', file_path, exc)
            errors += 1
            continue
        if image_bytes:
            image_id = str(file_path.relative_to(root)).replace(os.sep, '/')
            data.append((image_bytes, image_id, str(file_path)))
    return data, errors


def _status(processed: int, errors: int) -> Literal['success', 'partial', 'error']:
    if not errors:
        return 'success'
    return 'partial' if processed else 'error'


async def run_directory_ingest(
    search_service: Any,
    options: DirectoryIngestOptions,
    *,
    resume_batch: int = 0,
    should_stop: Callable[[], bool] = lambda: False,
    on_batch: Callable[[int, int, int], Awaitable[None]] | None = None,
    servers: list[str] | None = None,
) -> dict[str, Any]:
    """Ingest the directory and return the ``DirectoryIngestResponse`` fields plus the job's
    ``perf_summary``. ``resume_batch`` skips the batches an earlier attempt finished (counts then
    cover this attempt only); ``on_batch(batch_index, files_done, total_files)`` runs after each."""
    started = time.perf_counter()
    root = check_directory(options.directory)
    images, skipped = await asyncio.to_thread(
        collect_images, root, recursive=options.recursive, max_images=options.max_images
    )
    totals = {'processed': 0, 'duplicates': 0, 'errors': 0, 'near_duplicates': 0}
    indexed = dict.fromkeys(INDEXED_KEYS, 0)
    size = options.batch_size
    for index, start in enumerate(range(0, len(images), size)):
        if index < resume_batch:
            continue
        if should_stop():
            break
        data, unreadable = await asyncio.to_thread(_read_batch, root, images[start : start + size])
        totals['errors'] += unreadable
        if data:
            try:
                result = await search_service.ingest_batch(
                    images_data=data,
                    skip_duplicates=options.skip_duplicates,
                    detect_near_duplicates=options.detect_near_duplicates,
                    near_duplicate_threshold=options.near_duplicate_threshold,
                    enable_ocr=options.enable_ocr,
                    enable_detection=options.enable_detection,
                    enable_faces=options.enable_faces,
                    enable_clip=options.enable_clip,
                    defer_heavy_ops=options.defer_heavy_ops,
                )
            except _BATCH_ERRORS as exc:  # one failed batch is counted, the directory goes on
                logger.error('Batch processing failed: %s', exc)
                totals['errors'] += len(data)
            else:
                totals['processed'] += result.get('processed', 0)
                totals['duplicates'] += result.get('duplicates', 0)
                totals['errors'] += result.get('errors_count', 0)
                totals['near_duplicates'] += result.get('near_duplicates', 0)
                for key in INDEXED_KEYS:
                    indexed[key] += (result.get('indexed') or {}).get(key, 0)
        if on_batch is not None:
            await on_batch(index + 1, min(start + size, len(images)), len(images))
    elapsed = time.perf_counter() - started
    return {
        'status': _status(totals['processed'], totals['errors']),
        'directory': options.directory,
        'total_files': len(images),
        'total': len(images),
        'processed': totals['processed'],
        'duplicates': totals['duplicates'],
        'errors_count': totals['errors'],
        'indexed': indexed,
        'near_duplicates': totals['near_duplicates'],
        'skipped_extensions': skipped or None,
        'total_time_ms': round(elapsed * 1000, 2),
        RESULT_KEY: JobPerfSummary(
            images=totals['processed'] + totals['duplicates'],
            wall_s=round(elapsed, 3),
            images_per_s=round((totals['processed'] + totals['duplicates']) / elapsed, 3)
            if elapsed
            else 0.0,
            decode_routes={},
            stage_seconds={},
            pe_calls_per_image=0.0,
            servers=list(servers or []),
        ).model_dump(),
    }
