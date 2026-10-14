"""Stage image bytes and run them through the ingest pipeline.

The one implementation behind both byte-carrying ingest routes: ``/ingest/upload``
(the client sends the bytes) and ``/ingest/objects`` (the API fetches them from an
object store). Each entry is checked (non-empty, accepted extension), fingerprinted,
written content-addressed under the project's upload root (so the stored
``image_path`` stays openable by thumbnails, crops, the region worker and the VLM)
and handed to :meth:`CurationIngestService.ingest_batch` with the caller's identifier
recorded as ``source_identifier``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from src.services.curation.image_serving import persist_uploaded_bytes
from src.services.curation.ingest import _imohash_bytes
from src.services.curation.ingest_models import ERROR_KIND_EMPTY, ERROR_KIND_UNSUPPORTED_TYPE
from src.services.curation.open_vocab_ingest import schedule_open_vocab_after_ingest
from src.utils.stage_timing import stage_timer


if TYPE_CHECKING:
    from opensearchpy import AsyncOpenSearch

    from src.config import CurationConfig
    from src.services.curation.ingest import CurationIngestService
    from src.services.curation.ingest_models import BatchIngestResult


class PayloadTooLargeError(Exception):
    """The accepted entries add up to more than the caller allows."""


@dataclass(frozen=True)
class StagedEntry:
    identifier: str
    payload: bytes
    extension: str


@dataclass(frozen=True)
class EarlyFailure:
    """An entry refused before the pipeline saw it."""

    identifier: str
    error: str
    error_kind: str


async def stage_and_ingest(
    opensearch: AsyncOpenSearch,
    service: CurationIngestService,
    entries: list[StagedEntry],
    *,
    config: CurationConfig,
    source: str,
    run_id: str | None,
    max_total_bytes: int | None = None,
) -> tuple[BatchIngestResult | None, list[EarlyFailure]]:
    """Returns the pipeline's batch result (``None`` when nothing was admitted) and
    the entries refused up front. Raises :class:`PayloadTooLargeError` when the admitted
    entries exceed ``max_total_bytes``."""
    data: list[bytes] = []
    kept_paths: list[str] = []
    identifiers: list[str | None] = []
    early: list[EarlyFailure] = []
    total = 0
    for entry in entries:
        if not entry.payload:
            early.append(EarlyFailure(entry.identifier, 'empty upload', ERROR_KIND_EMPTY))
            continue
        if entry.extension not in config.upload_accepted_extensions:
            early.append(
                EarlyFailure(
                    entry.identifier,
                    f'{entry.extension!r} is not an accepted extension '
                    f'({", ".join(config.upload_accepted_extensions)})',
                    ERROR_KIND_UNSUPPORTED_TYPE,
                )
            )
            continue
        total += len(entry.payload)
        if max_total_bytes is not None and total > max_total_bytes:
            raise PayloadTooLargeError(total)
        image_hash = _imohash_bytes(entry.payload)
        with stage_timer('persist', nbytes=len(entry.payload)):
            persisted = persist_uploaded_bytes(
                entry.payload, imohash=image_hash, extension=entry.extension, config=config
            )
        data.append(entry.payload)
        kept_paths.append(str(persisted))
        identifiers.append(entry.identifier)

    result: Any = None
    if data:
        result = await service.ingest_batch(
            data,
            kept_paths,
            source=source,
            source_identifiers=identifiers,
            ingest_run_id=run_id,
        )
        await schedule_open_vocab_after_ingest(
            opensearch,
            service,
            [r.image_id for r in result.results if r.status == 'success' and r.image_id],
        )
    return result, early
