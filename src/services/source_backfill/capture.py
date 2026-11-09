"""The capture step: each stored image's capture metadata into the images and items indexes.

Images the 0.6.0 upgrade carried over, and images whose ingest-time read failed (an ExifTool
outage leaves no parser stamp), have no `capture.parser`. The step reads each such file's header
once through ExifTool (a missing file is counted, not failed), stores the `capture` block under
the project's location policy, and copies `capture_taken_local` / `capture_taken_utc` onto the
image's items in one `update_by_query` per page. It writes nothing for an image that already has
a parser stamp, so a re-run is a no-op. An ExifTool outage ends the step with an error rather
than counting every file unknown.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from src.clients.curation_opensearch.capture_fields import capture_image_block, capture_item_fields
from src.clients.exiftool_client import ExifToolError
from src.config.curation_accessors import images_index, items_index
from src.domain.location_policy import apply_location_policy
from src.services.curation.capture_read import read_capture_file


if TYPE_CHECKING:
    from collections.abc import Sequence

    from opensearchpy import AsyncOpenSearch

    from src.domain.capture_metadata import CaptureMetadata

#: Consecutive ExifTool failures after which the step stops: the reader is down, not the files.
MAX_CONSECUTIVE_OUTAGES = 3

_FILL_IMAGE = (
    'if (ctx._source.capture == null || ctx._source.capture.parser == null) '
    "{ ctx._source.capture = params.capture } else { ctx.op = 'noop' }"
)
_COPY_TO_ITEMS = (
    'def f = params.by_image.get(ctx._source.image_id); '
    "if (f == null) { ctx.op = 'noop' } "
    'else { for (e in f.entrySet()) { ctx._source[e.getKey()] = e.getValue() } }'
)


class CaptureWriteError(RuntimeError):
    """The images index refused part of a capture write (a deleted image, 404, is not one)."""


@dataclass
class CaptureCounts:
    recorded: int = 0
    already: int = 0
    missing: int = 0
    unreadable: int = 0

    def add(self, other: CaptureCounts) -> None:
        self.recorded += other.recorded
        self.already += other.already
        self.missing += other.missing
        self.unreadable += other.unreadable

    def as_dict(self) -> dict[str, int]:
        return {f'capture_{k}': v for k, v in self.__dict__.items()}


async def _write_images(client: AsyncOpenSearch, blocks: dict[str, dict[str, Any]]) -> None:
    body: list[dict[str, Any]] = []
    for image_id, block in blocks.items():
        body.append({'update': {'_index': images_index(), '_id': image_id}})
        body.append(
            {
                'script': {
                    'lang': 'painless',
                    'source': _FILL_IMAGE,
                    'params': {'capture': block},
                }
            }
        )
    resp = await client.bulk(body=body, refresh=False)
    failed = [
        row['update']
        for row in resp.get('items', [])
        if row['update'].get('status', 200) >= 300 and row['update'].get('status') != 404
    ]
    if failed:
        msg = f'capture write refused for {len(failed)} images'
        raise CaptureWriteError(msg)


async def _copy_to_items(client: AsyncOpenSearch, by_image: dict[str, dict[str, Any]]) -> None:
    await client.update_by_query(
        index=items_index(),
        body={
            'query': {'terms': {'image_id': list(by_image)}},
            'script': {
                'lang': 'painless',
                'source': _COPY_TO_ITEMS,
                'params': {'by_image': by_image},
            },
        },
        conflicts='proceed',
        refresh=False,
    )


async def record_captures(
    client: AsyncOpenSearch, policy: str, images: Sequence[tuple[str, str, bool]]
) -> CaptureCounts:
    """Record the capture of ``images`` (``(image_id, image_path, has_parser_stamp)``) of one page."""
    counts = CaptureCounts()
    blocks: dict[str, dict[str, Any]] = {}
    item_fields: dict[str, dict[str, Any]] = {}
    outages = 0
    for image_id, path, stamped in images:
        if stamped:
            counts.already += 1
            continue
        try:
            meta: CaptureMetadata | None = await read_capture_file(path)
        except ExifToolError:
            counts.unreadable += 1
            outages += 1
            if outages >= MAX_CONSECUTIVE_OUTAGES:
                raise
            continue
        outages = 0
        if meta is None:
            counts.missing += 1
            continue
        blocks[image_id] = capture_image_block(apply_location_policy(meta, policy))
        if fields := capture_item_fields(meta):
            item_fields[image_id] = fields
    if blocks:
        await _write_images(client, blocks)
        counts.recorded += len(blocks)
    if item_fields:
        await _copy_to_items(client, item_fields)
    return counts


__all__ = ['MAX_CONSECUTIVE_OUTAGES', 'CaptureCounts', 'CaptureWriteError', 'record_captures']
