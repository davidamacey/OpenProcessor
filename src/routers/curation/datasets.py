"""``/datasets/*``: import an already-labeled dataset (W10).

Thin routes over :mod:`src.services.curation.dataset_import`: every
decision (scan, mapping, identity, claiming, chunking, undo) lives there;
this module maps its results and errors onto the wire. Every 4xx goes
through :func:`api_error`.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

from fastapi import File, Query, Response, UploadFile

from src.config import get_curation_config
from src.config.ingest_profiles import ingest_primary_profile
from src.routers.curation._common import OpenSearchDep, RegistryDep, _ensure_indexes, router
from src.routers.curation._config_common_models import api_error
from src.routers.curation._dataset_import_models import (
    DatasetFormatsResponse,
    DatasetImportEntry,
    DatasetImportEntryPage,
    DatasetImportJob,
    DatasetImportList,
    DatasetIssuePage,
    DatasetPreview,
    DatasetUndoReportWire,
    DatasetUndoRequest,
    DatasetUploadResponse,
)
from src.routers.curation._dataset_import_views import job_wire, preview_wire
from src.routers.curation._dataset_issue_models import issue_to_wire
from src.services.curation.dataset_import import runner
from src.services.curation.dataset_import.context import ImportContext
from src.services.curation.dataset_import.options import DatasetImportRequest, DatasetPreviewRequest
from src.services.curation.dataset_import.paths import (
    DatasetPathNotAllowedError,
    dataset_path_guard,
)
from src.services.curation.dataset_import.prepare import (
    PreparedImport,
    materialize_created_classes,
    pin_project_view,
    prepare_import,
)
from src.services.curation.dataset_import.scan import FormatUndetectedError
from src.services.curation.dataset_import.store import (
    ACTIVE_STATUSES,
    COMPLETED_STATUSES,
    RESUMABLE_STATUSES,
    list_stores,
    now_iso,
    open_store,
)
from src.services.curation.dataset_import.undo import UndoContext, undo_import
from src.services.curation.dataset_import.upload import (
    ArchiveInvalidError,
    UploadTooLargeError,
    receive_archive,
)
from src.services.curation.ingest import CurationIngestService


if TYPE_CHECKING:
    from src.clients.curation_opensearch import ClassRegistry
    from src.services.curation.dataset_import.store import ImportStore

_UNDOABLE = frozenset(
    {'completed', 'completed_with_errors', 'failed', 'cancelled', 'interrupted', 'undone'}
)


def _slug() -> str:
    return get_curation_config().project_slug


def _guard() -> Any:
    return dataset_path_guard(get_curation_config().export_root)


def _open(import_id: str) -> ImportStore:
    store = open_store(import_id)
    if store is None:
        raise api_error(404, 'import_not_found', f'no import {import_id!r}', project=_slug())
    return store


async def _prepare(request: DatasetPreviewRequest, registry: ClassRegistry) -> PreparedImport:
    view = pin_project_view(registry)
    try:
        return await asyncio.to_thread(prepare_import, request, view, path_guard=_guard())
    except DatasetPathNotAllowedError as exc:
        raise api_error(
            422,
            'dataset_path_not_allowed',
            f'{exc} is outside the configured source roots',
            project=_slug(),
        ) from None
    except FormatUndetectedError:
        raise api_error(
            422,
            'format_undetected',
            'no YOLO, COCO or OpenProcessor-export layout was found at that path',
            project=_slug(),
        ) from None


async def _already_indexed(opensearch: Any, prepared: PreparedImport) -> int:
    """Entries whose resolved path an images doc already carries (a re-import
    of the same files). Content duplicates under other paths are found at
    import time, where the bytes are hashed."""
    paths = [str(e.abs_image_path.resolve()) for e in prepared.scan.entries]
    if not paths:
        return 0
    cfg = get_curation_config()
    found: set[str] = set()
    try:
        for i in range(0, len(paths), 1000):
            resp = await opensearch.search(
                index=cfg.images_index,
                body={
                    'size': 1000,
                    'query': {'terms': {'image_path': paths[i : i + 1000]}},
                    '_source': ['image_path'],
                },
            )
            found.update(
                h['_source'].get('image_path') for h in (resp.get('hits') or {}).get('hits') or []
            )
    except Exception:
        return 0
    return len(found & set(paths))


def _import_service(
    opensearch: Any, registry: Any, *, need_detector: bool
) -> CurationIngestService:
    from src.main import app, get_async_triton_pool

    pe_encoder = getattr(app.state, 'pe_encoder', None)
    if pe_encoder is None:
        raise api_error(503, 'config_store_unavailable', 'PE encoder not initialized')
    try:
        profile = ingest_primary_profile()
    except ValueError as exc:
        raise api_error(503, 'config_store_unavailable', f'ingest misconfigured: {exc}') from exc
    if need_detector and not profile.detector_model:
        raise api_error(
            503, 'config_store_unavailable', 'this import runs the detector; none is configured'
        )
    return CurationIngestService(
        opensearch=opensearch,
        triton_pool=get_async_triton_pool() if need_detector else None,
        registry=registry,
        profile=profile,
        pe_encoder=pe_encoder,
    )


def _context(
    store: ImportStore,
    request: DatasetImportRequest,
    prepared_like: Any,
    opensearch: Any,
    registry: Any,
) -> ImportContext:
    """The run context from persisted/pinned inputs only."""
    resolved, profile, parents = prepared_like
    cfg = get_curation_config()
    state = store.job.read()
    has_region = any(t.kind == 'region' for t in resolved.targets.values())
    need_detector = request.options.processing == 'propose' or (has_region and parents == 'detect')
    service = _import_service(opensearch, registry, need_detector=need_detector)
    return ImportContext(
        import_id=store.import_id,
        options=request.options,
        resolved=resolved,
        profile=profile,
        parents=parents,
        source_sha=state['source_sha'],
        source_format=state.get('source_format', ''),
        source_root=Path(state.get('source_root', '')),
        opensearch=service.opensearch,
        service=service,
        images_index=cfg.images_index,
        items_index=cfg.items_index,
        crop_cache_dir=cfg.crop_cache_dir,
        upload_root=Path(cfg.upload_root),
        export_root=Path(cfg.export_root) if cfg.export_root else None,
        freeze_test=bool(state.get('freeze_test')),
    )


# ------------------------------------------------------------------- upload


@router.post('/datasets/uploads', response_model=DatasetUploadResponse, status_code=201)
async def upload_dataset(file: Annotated[UploadFile, File()]) -> DatasetUploadResponse:
    cfg = get_curation_config()

    async def stream() -> Any:
        while chunk := await file.read(1024 * 1024):
            yield chunk

    try:
        result = await receive_archive(stream(), upload_root=Path(cfg.upload_root))
    except UploadTooLargeError as exc:
        raise api_error(
            413, 'upload_too_large', 'the archive is over the upload limit', limit=exc.limit
        ) from None
    except ArchiveInvalidError as exc:
        raise api_error(422, 'archive_invalid', str(exc)) from None
    return DatasetUploadResponse(
        upload_id=result.upload_id,
        dataset_path=str(result.dataset_path),
        bytes=result.bytes,
        files=result.files,
    )


# ------------------------------------------------------------------ preview


@router.post('/datasets/preview', response_model=DatasetPreview)
async def preview_dataset(
    body: DatasetPreviewRequest, opensearch: OpenSearchDep, registry: RegistryDep
) -> DatasetPreview:
    prepared = await _prepare(body, registry)
    return preview_wire(prepared, already_indexed=await _already_indexed(opensearch, prepared))


# ------------------------------------------------------------------- start


@router.post('/datasets/imports', response_model=DatasetImportJob, status_code=202)
async def start_dataset_import(
    body: DatasetImportRequest,
    response: Response,
    opensearch: OpenSearchDep,
    registry: RegistryDep,
) -> DatasetImportJob:
    await _ensure_indexes(opensearch)
    prepared = await _prepare(body, registry)
    if body.expected_import_key and body.expected_import_key != prepared.import_key:
        raise api_error(
            409,
            'dataset_changed',
            'the dataset or mapping changed since the preview',
            project=_slug(),
        )
    prior = runner.reusable_prior(prepared)
    if prior is not None and prior[1] in COMPLETED_STATUSES:
        response.status_code = 200
        return job_wire(prior[0], project=_slug(), reused=True)
    if prior is not None and prior[1] in RESUMABLE_STATUSES:
        raise api_error(
            409,
            'import_resumable',
            'an interrupted import of this dataset can be resumed',
            import_id=prior[0].import_id,
            project=_slug(),
        )
    _refuse_unmapped_or_blocked(prepared, body)
    try:
        store, reused = runner.claim_import(body, prepared)
    except runner.ImportBusyError as exc:
        raise api_error(
            409, 'import_busy', str(exc), import_id=exc.import_id, project=_slug()
        ) from None
    except runner.ImportResumableError as exc:
        raise api_error(
            409,
            'import_resumable',
            'an interrupted import of this dataset can be resumed',
            import_id=exc.import_id,
            project=_slug(),
        ) from None
    if reused:
        response.status_code = 200
        return job_wire(store, project=_slug(), reused=True)

    new_groups = {m.dataset_class: m.new_class_group for m in body.mapping}
    try:
        materialize_created_classes(prepared.resolved, registry, new_groups=new_groups)
    except Exception as exc:
        store.job.update(status='failed', error=str(exc)[:300], finished_at=now_iso())
        raise api_error(422, 'class_mapping_invalid', f'could not create a class: {exc}') from None
    runner.persist_pinned(store, prepared)
    runner.persist_scan_summary(store, prepared)
    freeze = runner.freeze_default(prepared, body)
    store.job.update(freeze_test=freeze)
    ctx = _context(
        store,
        body,
        (prepared.resolved, prepared.view.profile, prepared.parents),
        opensearch,
        registry,
    )
    runner.spawn(ctx, store, runner.ordered_entries(prepared.scan.entries))
    return job_wire(store, project=_slug())


def _refuse_unmapped_or_blocked(prepared: PreparedImport, body: DatasetImportRequest) -> None:
    errors = prepared.resolved.errors
    unmapped = [e.dataset_class or '' for e in errors if e.code == 'class_mapping_incomplete']
    if unmapped:
        raise api_error(
            422,
            'class_mapping_incomplete',
            'every dataset class with boxes needs a mapping',
            unmapped=unmapped,
            project=_slug(),
        )
    if errors:
        raise api_error(
            422,
            'class_mapping_invalid',
            '; '.join(f'{e.dataset_class}: {e.code}' for e in errors)[:300],
            issues=[issue_to_wire(i) for i in prepared.issues if i.blocking],
            project=_slug(),
        )
    if prepared.blocking and not (body.options.force and prepared.force_allowed()):
        raise api_error(
            422,
            'import_blocked',
            'the dataset has blocking issues',
            issues=[issue_to_wire(i) for i in prepared.issues if i.blocking],
            project=_slug(),
        )


# --------------------------------------------------------------------- reads


@router.get('/datasets/formats', response_model=DatasetFormatsResponse)
async def dataset_formats() -> DatasetFormatsResponse:
    from src.routers.curation._dataset_formats import formats_response

    return formats_response()


@router.get('/datasets/imports', response_model=DatasetImportList)
async def list_dataset_imports(
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=200)] = 20,
    status: str | None = None,
) -> DatasetImportList:
    stores = [s for s in list_stores() if status is None or s.job.read().get('status') == status]
    chunk = stores[(page - 1) * page_size : page * page_size]
    return DatasetImportList(
        items=[job_wire(s, project=_slug()) for s in chunk],
        total=len(stores),
        page=page,
        page_size=page_size,
    )


@router.get('/datasets/imports/{import_id}', response_model=DatasetImportJob)
async def get_dataset_import(import_id: str) -> DatasetImportJob:
    return job_wire(_open(import_id), project=_slug())


@router.get('/datasets/imports/{import_id}/issues', response_model=DatasetIssuePage)
async def dataset_import_issues(
    import_id: str,
    code: str | None = None,
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=500)] = 50,
) -> DatasetIssuePage:
    rows = [r for r in _open(import_id).issues() if code is None or r.get('code') == code]
    return DatasetIssuePage(
        items=rows[(page - 1) * page_size : page * page_size],
        total=len(rows),
        page=page,
        page_size=page_size,
    )


@router.get('/datasets/imports/{import_id}/entries', response_model=DatasetImportEntryPage)
async def dataset_import_entries(
    import_id: str,
    split: str | None = None,
    label_state: str | None = None,
    status: str | None = None,
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=500)] = 50,
) -> DatasetImportEntryPage:
    rows = [
        r
        for r in _open(import_id).ledger_rows()
        if (split is None or r.get('split') == split)
        and (label_state is None or r.get('label_state') == label_state)
        and (status is None or r.get('status') == status)
    ]
    window = rows[(page - 1) * page_size : page * page_size]
    return DatasetImportEntryPage(
        items=[
            DatasetImportEntry(
                **{k: v for k, v in r.items() if k in DatasetImportEntry.model_fields}
            )
            for r in window
        ],
        total=len(rows),
        page=page,
        page_size=page_size,
    )


# ------------------------------------------------------------------ actions


@router.post('/datasets/imports/{import_id}/cancel', response_model=DatasetImportJob)
async def cancel_dataset_import(import_id: str) -> DatasetImportJob:
    store = _open(import_id)
    runner.cancel_import(import_id)
    return job_wire(store, project=_slug())


@router.post(
    '/datasets/imports/{import_id}/resume', response_model=DatasetImportJob, status_code=202
)
async def resume_dataset_import(
    import_id: str, opensearch: OpenSearchDep, registry: RegistryDep
) -> DatasetImportJob:
    store = _open(import_id)
    try:
        runner.check_resumable(store)
    except runner.ImportNotResumableError:
        raise api_error(
            409,
            'import_not_resumable',
            'only an interrupted, failed or cancelled import resumes',
            import_id=import_id,
            project=_slug(),
        ) from None
    except runner.ImportBusyError as exc:
        raise api_error(
            409, 'import_busy', str(exc), import_id=exc.import_id, project=_slug()
        ) from None
    request = DatasetImportRequest(**store.read_request())
    try:
        entries = await asyncio.to_thread(
            runner.rescan_for_resume, store, request, path_guard=_guard()
        )
    except runner.DatasetChangedError:
        raise api_error(
            409,
            'dataset_changed',
            'the dataset changed since this import started',
            import_id=import_id,
            project=_slug(),
        ) from None
    except (DatasetPathNotAllowedError, FormatUndetectedError) as exc:
        raise api_error(422, 'dataset_path_not_allowed', str(exc), project=_slug()) from None
    runner.prepare_resume(store, registry)
    ctx = _context(store, request, runner.load_pinned(store), opensearch, registry)
    store.job.clear_signals()
    store.job.update(status='queued', error=None, finished_at=None)
    store.job.touch_heartbeat()
    runner.spawn(ctx, store, entries)
    return job_wire(store, project=_slug())


@router.post('/datasets/imports/{import_id}/undo', response_model=None)
async def undo_dataset_import(
    import_id: str,
    body: DatasetUndoRequest,
    response: Response,
    opensearch: OpenSearchDep,
    registry: RegistryDep,
) -> DatasetUndoReportWire | DatasetImportJob:
    store = _open(import_id)
    state = store.job.read()
    if state.get('status') not in _UNDOABLE or store.job.is_live(ACTIVE_STATUSES):
        raise api_error(
            409,
            'import_not_undoable',
            'an import that is running cannot be undone',
            import_id=import_id,
            project=_slug(),
        )
    cfg = get_curation_config()
    from src.config.region_fields import get_region_fields

    ctx = UndoContext(
        import_id=import_id,
        opensearch=getattr(opensearch, 'client', opensearch),
        images_index=cfg.images_index,
        items_index=cfg.items_index,
        crop_cache_dir=cfg.crop_cache_dir,
        region_fields=get_region_fields(),
        registry=registry,
    )
    created = store.read_mapping().get('created_classes') or {}
    if body.dry_run:
        report = await undo_import(
            ctx,
            store,
            dry_run=True,
            remove_images=body.remove_images,
            deprecate_created_classes=body.deprecate_created_classes,
            created_classes=created,
        )
        return DatasetUndoReportWire(**report.to_dict())
    runner.spawn_undo(ctx, store, body, created)
    response.status_code = 202
    return job_wire(store, project=_slug())


__all__ = ['router']
