"""``POST /ingest/bulk/preview``: what a bulk ingest would touch and cost, with no ingestion.

A read-only look at a folder, manifest, S3 prefix or registered source, bounded by the
served caps and cancelled when the client disconnects. Nothing is written and no
datastore is read; the source must obey the same rules a start does (inside a source
root, readable, inside ``OP_S3_ALLOWED``).
"""

from __future__ import annotations

import asyncio
import threading
from dataclasses import asdict
from typing import TYPE_CHECKING

from fastapi import HTTPException, Request

from src.config import get_curation_config
from src.routers.curation._api_error import api_error
from src.routers.curation._common import router
from src.routers.curation._ingest_bulk_preview_models import (
    BulkPreview,
    BulkPreviewConfig,
    BulkPreviewRequest,
    PreviewEstimateWire,
    PreviewExtensionWire,
    PreviewMeasuredWire,
    PreviewMegapixelsWire,
    PreviewRateWire,
)
from src.routers.curation._ingest_bulk_wire import slug
from src.services.curation.bulk_ingest import preflight, preflight_rates
from src.services.curation.bulk_ingest.object_store import (
    ObjectStoreError,
    S3Config,
    S3NotAllowedError,
    S3NotConfiguredError,
    get_object_store,
)
from src.services.curation.bulk_ingest.source import (
    BulkSource,
    InvalidSourceError,
    normalise_extensions,
)
from src.services.curation.bulk_ingest.sources import load_source


if TYPE_CHECKING:
    from collections.abc import Callable

    from src.services.curation.bulk_ingest.preflight import PreviewResult

PREVIEW_LIMITS = preflight.PreviewLimits()
_POLL_DISCONNECT_S = 0.5


def preview_config() -> BulkPreviewConfig:
    """Served inside ``GET /ingest/bulk/config``."""
    config = get_curation_config()
    return BulkPreviewConfig(
        max_files=PREVIEW_LIMITS.max_files,
        max_seconds=PREVIEW_LIMITS.max_seconds,
        header_sample=PREVIEW_LIMITS.header_sample,
        gpu_decode=config.ingest_gpu_decode,
        gpu_min_megapixels=config.ingest_gpu_min_megapixels,
        rates=[PreviewRateWire(**asdict(r)) for r in preflight_rates.RATES],
        embeddings_per_image=PreviewMeasuredWire(**asdict(preflight_rates.EMBEDDINGS_PER_IMAGE)),
        opensearch_bytes_per_image=PreviewMeasuredWire(
            **asdict(preflight_rates.OPENSEARCH_BYTES_PER_IMAGE)
        ),
        source_document='docs/PERFORMANCE.md',
    )


def _wire(
    result: PreviewResult, kind: str, target: str, extensions: tuple[str, ...]
) -> BulkPreview:
    return BulkPreview(
        project=slug(),
        kind=kind,
        target=target,
        extensions=list(extensions),
        files=result.files,
        total_bytes=result.total_bytes,
        by_extension={e: PreviewExtensionWire(**asdict(t)) for e, t in result.by_extension.items()},
        megapixels=PreviewMegapixelsWire(**asdict(result.megapixels)),
        estimate=PreviewEstimateWire(**asdict(result.estimate)),
        partial=result.partial,
        stop_reason=result.stop_reason,
        unreadable_dirs=result.unreadable_dirs,
        unreadable_samples=result.unreadable_samples,
        symlinks_skipped=result.symlinks_skipped,
        missing=result.missing,
        outside_roots=result.outside_roots,
        unreadable_files=result.unreadable_files,
        gpu_decode=result.gpu_decode,
    )


def _plan(
    body: BulkPreviewRequest,
) -> tuple[str, str, tuple[str, ...], Callable[..., PreviewResult]]:
    """``(kind, target, extensions, run)`` for the request; ``run`` takes ``should_stop``."""
    config = get_curation_config()
    gpu_decode = config.ingest_gpu_decode
    if body.source_id is not None:
        defn = load_source(body.source_id)
        if defn is None:
            raise api_error(
                404,
                'ingest_source_not_found',
                f'no ingest source {body.source_id!r}',
                project=slug(),
            )
        extensions = tuple(defn.extensions)
        if defn.backend == 'folder':
            body = BulkPreviewRequest(folder=defn.path, extensions=list(extensions))
        else:
            from src.routers.curation._ingest_sources_models import S3Location

            body = BulkPreviewRequest(
                s3=S3Location(bucket=defn.bucket, prefix=defn.prefix), extensions=list(extensions)
            )
    extensions = normalise_extensions(body.extensions)
    if body.s3 is not None:
        s3 = body.s3
        prefix = s3.prefix.lstrip('/')

        def run_s3(should_stop: Callable[[], bool]) -> PreviewResult:
            s3_config = S3Config.from_env()
            s3_config.check(s3.bucket, prefix)
            return preflight.preview_s3(
                get_object_store(),
                s3_config,
                s3.bucket,
                prefix,
                extensions,
                limits=PREVIEW_LIMITS,
                should_stop=should_stop,
                gpu_decode=gpu_decode,
            )

        return 's3', f's3://{s3.bucket}/{prefix}', extensions, run_s3
    kind = 'folder' if body.folder is not None else 'manifest'
    path = body.folder or body.manifest or ''
    source = BulkSource(kind, path, extensions)  # type: ignore[arg-type]

    def run_path(should_stop: Callable[[], bool]) -> PreviewResult:
        return preflight.preview_source(
            source,
            limits=PREVIEW_LIMITS,
            should_stop=should_stop,
            gpu_decode=gpu_decode,
            gpu_min_megapixels=config.ingest_gpu_min_megapixels,
        )

    return kind, path, extensions, run_path


@router.post('/ingest/bulk/preview', response_model=BulkPreview)
async def preview_bulk_ingest(body: BulkPreviewRequest, request: Request) -> BulkPreview:
    """Count the files, bytes and extensions of a source, sample image sizes, and estimate
    duration and index footprint from measured figures (``GET /ingest/bulk/config``).
    Capped by file count and seconds (``partial: true`` then); a client disconnect stops
    the walk."""
    kind, target, extensions, run = _plan(body)
    cancelled = threading.Event()

    async def watch_disconnect() -> None:
        while not cancelled.is_set():
            if await request.is_disconnected():
                cancelled.set()
                return
            await asyncio.sleep(_POLL_DISCONNECT_S)

    watcher = asyncio.create_task(watch_disconnect())
    try:
        result = await asyncio.to_thread(run, cancelled.is_set)
    except InvalidSourceError as exc:
        raise api_error(422, 'bulk_ingest_invalid_source', str(exc), project=slug()) from None
    except S3NotConfiguredError as exc:
        raise api_error(503, 's3_not_configured', str(exc), project=slug()) from None
    except S3NotAllowedError as exc:
        raise api_error(422, 's3_not_allowed', str(exc), project=slug()) from None
    except ObjectStoreError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from None
    finally:
        cancelled.set()
        watcher.cancel()
    return _wire(result, kind, target, extensions)
