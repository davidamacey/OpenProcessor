"""``POST /ingest/upload`` — multipart byte-upload ingest.

Split out of ``ingest.py`` to stay under the 700-LOC ratchet: this route's
body (extension sniffing, per-item size/type accounting, content-addressed
persistence) is self-contained and only needs ``_get_ingest_service`` /
``_batch_response`` from the sibling module.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

from fastapi import File, Form, HTTPException, UploadFile

from src.routers.curation._common import (
    BatchIngestResponse as _BatchIngestResponse,
    IngestImageResponse,
    OpenSearchDep,
    RegistryDep,
    _ensure_indexes,
    router,
)
from src.routers.curation.ingest import _batch_response, _get_ingest_service
from src.services.curation.upload_ingest import (
    EarlyFailure,
    PayloadTooLargeError,
    StagedEntry,
    stage_and_ingest,
)


def early_failures_to_wire(early: list[EarlyFailure]) -> list[IngestImageResponse]:
    return [
        IngestImageResponse(
            status='failed',
            image_path=f.identifier,
            error=f.error,
            error_kind=f.error_kind,
            source_identifier=f.identifier,
        )
        for f in early
    ]


def _parse_upload_paths(image_paths: str | None, uploads: list[UploadFile]) -> list[str]:
    if image_paths is None or not image_paths.strip():
        return [u.filename or f'upload_{i}' for i, u in enumerate(uploads)]
    try:
        paths = json.loads(image_paths)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=422, detail=f'image_paths is not JSON: {exc}') from None
    if not isinstance(paths, list) or not all(isinstance(p, str) and p for p in paths):
        raise HTTPException(status_code=422, detail='image_paths must be a JSON list of strings')
    if len(paths) != len(uploads):
        raise HTTPException(
            status_code=422,
            detail=f'image_paths has {len(paths)} entries but {len(uploads)} images were sent',
        )
    return paths


def _upload_extension(path: str, upload: UploadFile) -> str:
    """The file extension to persist an uploaded image under — from the
    client identifier first (it usually looks like a filename), else the
    multipart upload's own filename, else '.jpg'."""
    for candidate in (path, upload.filename or ''):
        suffix = Path(candidate).suffix.lower()
        if suffix:
            return suffix
    return '.jpg'


@router.post(
    '/ingest/upload',
    response_model=_BatchIngestResponse,
    responses={413: {'description': 'Too many images, or total upload bytes, in one request.'}},
)
async def curation_ingest_upload(
    images: Annotated[list[UploadFile], File(description='Encoded image files (JPEG/PNG)')],
    opensearch: OpenSearchDep,
    registry: RegistryDep,
    image_paths: Annotated[
        str | None,
        Form(
            description=(
                'JSON list of stable identifiers, one per image, stored as '
                'source_identifier (default: the upload filenames). Need not exist '
                'on the server -- image_path is now the server-persisted path '
                'the uploaded bytes were written under.'
            )
        ),
    ] = None,
    source: Annotated[str, Form(description='Provenance tag for every image')] = 'upload',
    run_id: Annotated[
        str | None,
        Form(description='Optional tag for this upload call, recorded as ingest_run_id.'),
    ] = None,
) -> _BatchIngestResponse:
    """Ingest a batch of images sent as bytes (multipart), not server-side paths.

    For storage the API container cannot mount (a laptop, a remote NAS, a
    high-latency share): the client reads the files and uploads them.

    The uploaded bytes are now persisted server-side, content-addressed,
    under ``CurationConfig.upload_root`` (see
    :func:`src.services.curation.image_serving.persist_uploaded_bytes`) —
    that path is stored as ``image_path`` (so thumbnails, the region worker
    and the VLM can all re-open it), and the client's own identifier is kept
    verbatim in the new ``source_identifier`` field. ``POST
    /curation/ingest/path_lookup`` matches on either field.

    Resume is server-side content dedup: every image is fingerprinted
    (imohash over the uploaded bytes) and one already in the images index
    comes back as ``duplicate`` without re-running inference — and since
    the persisted path is itself content-addressed by that same hash, the
    same bytes are only ever written to disk once, so a crashed upload run
    can simply be restarted. The whole-frame embedding is computed from the
    uploaded bytes, not by re-opening the path.
    """
    from src.config import get_curation_config

    cfg = get_curation_config()
    if not images:
        raise HTTPException(status_code=422, detail='no images uploaded')
    if len(images) > cfg.upload_max_images_per_request:
        raise HTTPException(
            status_code=413,
            detail=(
                f'{len(images)} images exceeds the per-request limit of '
                f'{cfg.upload_max_images_per_request}'
            ),
        )
    paths = _parse_upload_paths(image_paths, images)
    await _ensure_indexes(opensearch)
    service = await _get_ingest_service(opensearch, registry)

    entries = [
        StagedEntry(path, await upload.read(), _upload_extension(path, upload))
        for upload, path in zip(images, paths, strict=True)
    ]
    try:
        batch_result, early = await stage_and_ingest(
            opensearch,
            service,
            entries,
            config=cfg,
            source=source,
            run_id=run_id,
            max_total_bytes=cfg.upload_max_bytes_per_request,
        )
    except PayloadTooLargeError:
        raise HTTPException(
            status_code=413,
            detail=(
                f'total upload bytes exceeds the per-request limit of '
                f'{cfg.upload_max_bytes_per_request} bytes'
            ),
        ) from None
    return _batch_response(batch_result, early_failures_to_wire(early))
