"""``POST /ingest/objects``: ingest images held in an S3-compatible object store.

The object-store sibling of ``/ingest/upload``: the request carries references, the
API fetches the bytes (bounded in-flight encoded bytes, classified per-object
failures) and both routes then run the one shared staging-and-ingest function. The
stored reference of the original is ``source_identifier``
(``s3://bucket/key#etag=...``); an object whose exact reference is already stored is
answered ``duplicate`` without a download. ``/ingest/objects/presign`` turns a stored
reference into a time-limited download URL of the original.

Which buckets and prefixes may be read is deployment configuration (``OP_S3_ALLOWED``;
empty means off); credentials never travel in a request.
"""

from __future__ import annotations

import asyncio
from pathlib import PurePosixPath

from fastapi import HTTPException

from src.config import get_curation_config
from src.routers.curation._common import (
    OpenSearchDep,
    RegistryDep,
    _ensure_indexes,
    images_index,
    router,
)
from src.routers.curation._common_models import (
    BatchIngestResponse as _BatchIngestResponse,
    IngestImageResponse,
)
from src.routers.curation._config_common_models import api_error
from src.routers.curation._ingest_objects_models import (
    IngestObjectsRequest,
    ObjectPresignRequest,
    ObjectPresignResponse,
)
from src.routers.curation.ingest import _batch_response, _get_ingest_service
from src.routers.curation.ingest_upload import early_failures_to_wire
from src.services.curation.bulk_ingest.object_fetch import held_fetch
from src.services.curation.bulk_ingest.object_store import (
    ObjectRef,
    ObjectStoreError,
    S3Config,
    S3NotAllowedError,
    S3NotConfiguredError,
    get_object_store,
)
from src.services.curation.upload_ingest import EarlyFailure, StagedEntry, stage_and_ingest


def _s3_config() -> S3Config:
    config = S3Config.from_env()
    if not config.configured:
        raise api_error(
            503,
            's3_not_configured',
            'S3 ingest is off: set OP_S3_ALLOWED to the bucket[/prefix] entries it may read',
        )
    return config


def _check_allowed(config: S3Config, bucket: str, key: str) -> None:
    try:
        config.check(bucket, key)
    except S3NotAllowedError as exc:
        raise api_error(422, 's3_not_allowed', str(exc)) from None
    except S3NotConfiguredError as exc:
        raise api_error(503, 's3_not_configured', str(exc)) from None


def _extension(key: str) -> str:
    return PurePosixPath(key).suffix.lower()


async def _already_stored(opensearch: OpenSearchDep, uris: list[str]) -> dict[str, str]:
    """``{source_identifier: image_id}`` for the exact references already stored."""
    if not uris:
        return {}
    resp = await opensearch.search(
        index=images_index(),
        body={
            'size': len(uris),
            '_source': ['image_id', 'source_identifier'],
            'query': {'terms': {'source_identifier': uris}},
        },
    )
    found: dict[str, str] = {}
    for hit in resp.get('hits', {}).get('hits', []):
        src = hit.get('_source') or {}
        if src.get('source_identifier') and src.get('image_id'):
            found[src['source_identifier']] = src['image_id']
    return found


@router.post('/ingest/objects', response_model=_BatchIngestResponse)
async def curation_ingest_objects(
    body: IngestObjectsRequest,
    opensearch: OpenSearchDep,
    registry: RegistryDep,
) -> _BatchIngestResponse:
    """Fetch and ingest a batch of objects. A per-object failure (gone, forbidden,
    too large, unreachable, not an image) is reported as a ``failed`` result with its
    ``error_kind``; it never fails the batch."""
    cfg = get_curation_config()
    s3 = _s3_config()
    if len(body.items) > cfg.upload_max_images_per_request:
        raise HTTPException(
            status_code=413,
            detail=(
                f'{len(body.items)} objects exceeds the per-request limit of '
                f'{cfg.upload_max_images_per_request}'
            ),
        )
    for item in body.items:
        _check_allowed(s3, item.bucket, item.key)
    await _ensure_indexes(opensearch)
    service = await _get_ingest_service(opensearch, registry)

    refs = [ObjectRef(i.bucket, i.key, i.etag.strip('"'), i.version_id, i.size) for i in body.items]
    stored = await _already_stored(opensearch, [r.to_uri() for r in refs if r.etag])
    results: list[IngestImageResponse] = [
        IngestImageResponse(
            status='duplicate',
            image_id=stored[r.to_uri()],
            image_path=r.to_uri(),
            source_identifier=r.to_uri(),
        )
        for r in refs
        if r.etag and r.to_uri() in stored
    ]
    todo = [r for r in refs if not (r.etag and r.to_uri() in stored)]

    failed: list[IngestImageResponse] = []
    batch_result = None
    early: list[EarlyFailure] = []
    if todo:
        async with held_fetch(get_object_store(), todo, s3) as outcomes:
            entries = []
            for outcome in outcomes:
                if outcome.data is None or outcome.actual is None:
                    uri = outcome.requested.to_uri()
                    failed.append(
                        IngestImageResponse(
                            status='failed',
                            image_path=uri,
                            error=outcome.error or 'fetch failed',
                            error_kind=outcome.error_kind or 'fetch_error',
                            source_identifier=uri,
                        )
                    )
                    continue
                entries.append(
                    StagedEntry(
                        outcome.actual.to_uri(), outcome.data, _extension(outcome.actual.key)
                    )
                )
            batch_result, early = await stage_and_ingest(
                opensearch,
                service,
                entries,
                config=cfg,
                source=body.source,
                run_id=body.ingest_run_id,
            )
    response = _batch_response(batch_result, [*failed, *early_failures_to_wire(early)])
    response.results = [*results, *response.results]
    response.summary.duplicates += len(results)
    return response


@router.post('/ingest/objects/presign', response_model=ObjectPresignResponse)
async def curation_presign_object(body: ObjectPresignRequest) -> ObjectPresignResponse:
    """A time-limited URL of the original behind a stored object reference. The
    reference must lie inside ``OP_S3_ALLOWED``; the API never proxies the bytes."""
    s3 = _s3_config()
    try:
        ref = ObjectRef.parse(body.source_identifier)
    except ValueError as exc:
        raise api_error(422, 's3_not_allowed', str(exc)) from None
    _check_allowed(s3, ref.bucket, ref.key)
    try:
        url = await asyncio.to_thread(get_object_store().presign, ref, body.expires_in)
    except ObjectStoreError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from None
    return ObjectPresignResponse(url=url, expires_in=body.expires_in)
