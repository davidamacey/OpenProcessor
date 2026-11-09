"""Read the capture metadata of an ingested file through the worker's ExifTool process.

One thread reads at a time, which is what the single ``-stay_open`` pipe allows, so the reads of
many concurrent images queue on a dedicated one-thread pool instead of occupying the shared
executor. An ExifTool outage is logged by the client and answers :func:`unknown_capture`: the
image is ingested without capture, never with an invented position, and the capture backfill
reads it again (it carries no parser stamp).
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor

from src.clients.exiftool_client import ExifToolError, get_exiftool_client
from src.config.project_context import run_in_executor_bound
from src.domain.capture_metadata import CaptureMetadata, parse_capture, unknown_capture


_POOL = ThreadPoolExecutor(max_workers=1, thread_name_prefix='exiftool')


def _read(data: bytes) -> CaptureMetadata:
    try:
        return parse_capture(get_exiftool_client().read_bytes(data))
    except ExifToolError:
        return unknown_capture()


async def read_capture(data: bytes) -> CaptureMetadata:
    """The capture metadata of the file whose bytes are ``data``."""
    return await run_in_executor_bound(asyncio.get_running_loop(), _POOL, _read, data)


__all__ = ['read_capture']
