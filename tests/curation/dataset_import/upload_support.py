"""Receive and extract an archive in one call, as the route and the task do in two steps."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from src.services.curation.dataset_import import upload


if TYPE_CHECKING:
    from collections.abc import AsyncIterator


async def receive_archive(
    stream: AsyncIterator[bytes], *, upload_root: Path
) -> upload.UploadResult:
    with tempfile.TemporaryDirectory() as scratch:
        archive = Path(scratch) / 'received.archive'
        size, digest = await upload.receive_to_file(stream, archive)
        return upload.extract_received(archive, sha256=digest, size=size, upload_root=upload_root)
