"""``/datasets/*``: import an already-labeled dataset.

Thin routes over :mod:`src.services.curation.dataset_import`: every
decision (scan, mapping, identity, claiming, chunking, undo) lives there;
this module maps its results and errors onto the wire. Every 4xx goes
through :func:`api_error`.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Annotated, Any

from fastapi import File, Query, Response, UploadFile

from src.config import get_curation_config
from src.core.disk_space import DISK_GUARD
from src.db.session import async_session_factory
from src.routers.curation._api_error import api_error
from src.routers.curation._common import OpenSearchDep, RegistryDep, router
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
    DatasetUploadAccepted,
)
from src.routers.curation._dataset_import_views import job_wire, preview_wire
from src.routers.curation._dataset_issue_wire import issues_to_wire
from src.services.curation.dataset_import import control, runner, upload_job
from src.services.curation.dataset_import.options import DatasetImportRequest, DatasetPreviewRequest  # noqa: TC001 - FastAPI body types
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
from src.services.curation.dataset_import.status import COMPLETED_STATUSES, RESUMABLE_STATUSES
from src.services.curation.dataset_import.undo import UndoContext, undo_import
from src.services.curation.dataset_import.upload import UploadTooLargeError


if TYPE_CHECKING:
    from src.services.classes.registry import ClassRegistry
    from src.services.curation.dataset_import.control import ImportView


def _slug() -> str:
    return get_curation_config().project_slug


def _guard() -> Any:
    return dataset_path_guard(get_curation_config().export_root)


async def _open(import_id: str) -> ImportView:
    view = await control.view_of(import_id)
    if view is None:
        raise api_error(404, 'import_not_found', f'no import {import_id!r}', project=_slug())
    return view


async def _prepare(request: DatasetPreviewRequest, registry: ClassRegistry) -> PreparedImport:
    view = pin_project_view(await registry.snapshot())
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
            'no YOLO, COCO, Pascal VOC, Open Images or '
            'OpenProcessor-export layout was found at that path; '
            'see docs/CURATION.md, "Importing other formats", for conversion recipes',
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


# ------------------------------------------------------------------- upload


@router.post(
    '/datasets/uploads',
    dependencies=[DISK_GUARD],
    response_model=DatasetUploadAccepted,
    status_code=202,
)
async def upload_dataset(file: Annotated[UploadFile, File()]) -> DatasetUploadAccepted:
    """Store the archive and queue its extraction (`dataset_upload_extract`); nothing is
    extracted in the request. Poll the job: its result names the extracted `dataset_path`."""

    async def stream() -> Any:
        while chunk := await file.read(1024 * 1024):
            yield chunk

    try:
        async with async_session_factory()() as session:
            upload_id, job = await upload_job.receive_and_queue(
                session, stream(), filename=file.filename or 'dataset'
            )
    except UploadTooLargeError as exc:
        raise api_error(
            413, 'upload_too_large', 'the archive is over the upload limit', limit=exc.limit
        ) from None
    return DatasetUploadAccepted(upload_id=str(upload_id), job_id=str(job.id), status='queued')


# ------------------------------------------------------------------ preview


@router.post('/datasets/preview', response_model=DatasetPreview)
async def preview_dataset(
    body: DatasetPreviewRequest, opensearch: OpenSearchDep, registry: RegistryDep
) -> DatasetPreview:
    prepared = await _prepare(body, registry)
    return preview_wire(prepared, already_indexed=await _already_indexed(opensearch, prepared))


# ------------------------------------------------------------------- start


@router.post(
    '/datasets/imports', dependencies=[DISK_GUARD], response_model=DatasetImportJob, status_code=202
)
async def start_dataset_import(
    body: DatasetImportRequest, response: Response, registry: RegistryDep
) -> DatasetImportJob:
    prepared = await _prepare(body, registry)
    if body.expected_import_key and body.expected_import_key != prepared.import_key:
        raise api_error(
            409,
            'dataset_changed',
            'the dataset or mapping changed since the preview',
            project=_slug(),
        )
    prior = await control.prior_of(prepared)
    if prior is not None and prior.status in COMPLETED_STATUSES:
        response.status_code = 200
        return job_wire(prior, project=_slug(), reused=True)
    if prior is not None and prior.status in RESUMABLE_STATUSES:
        raise api_error(
            409,
            'import_resumable',
            'an interrupted import of this dataset can be resumed',
            import_id=prior.import_id,
            project=_slug(),
        )
    _refuse_unmapped_or_blocked(prepared, body)
    live = await control.live_import_id()
    if live is not None:
        raise api_error(
            409,
            'import_busy',
            f'an import is already running: {live}',
            import_id=live,
            project=_slug(),
        )
    new_groups = {m.dataset_class: m.new_class_group for m in body.mapping}
    try:
        await materialize_created_classes(prepared.resolved, registry, new_groups=new_groups)
    except Exception as exc:
        raise api_error(422, 'class_mapping_invalid', f'could not create a class: {exc}') from None
    try:
        view = await control.start(
            prepared,
            body,
            [i.model_dump() for i in issues_to_wire(prepared.issues)],
            freeze_test=runner.freeze_default(prepared, body),
        )
    except runner.ImportBusyError as exc:
        raise api_error(
            409, 'import_busy', str(exc), import_id=exc.import_id, project=_slug()
        ) from None
    return job_wire(view, project=_slug())


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
            '; '.join(e.describe() for e in errors)[:300],
            issues=issues_to_wire(i for i in prepared.issues if i.blocking),
            project=_slug(),
        )
    if prepared.blocking and not (body.options.force and prepared.force_allowed()):
        raise api_error(
            422,
            'import_blocked',
            'the dataset has blocking issues',
            issues=issues_to_wire(i for i in prepared.issues if i.blocking),
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
    views = [v for v in await control.list_views() if status is None or v.status == status]
    window = views[(page - 1) * page_size : page * page_size]
    return DatasetImportList(
        items=[job_wire(v, project=_slug()) for v in window],
        total=len(views),
        page=page,
        page_size=page_size,
    )


@router.get('/datasets/imports/{import_id}', response_model=DatasetImportJob)
async def get_dataset_import(import_id: str) -> DatasetImportJob:
    return job_wire(await _open(import_id), project=_slug())


@router.get('/datasets/imports/{import_id}/issues', response_model=DatasetIssuePage)
async def dataset_import_issues(
    import_id: str,
    code: str | None = None,
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=500)] = 50,
) -> DatasetIssuePage:
    found = (await _open(import_id)).record.report.get('job_issues') or []
    rows = [r for r in found if code is None or r.get('code') == code]
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
    window, total = await control.ledger_of(await _open(import_id)).page(
        split=split,
        label_state=label_state,
        status=status,
        offset=(page - 1) * page_size,
        limit=page_size,
    )
    return DatasetImportEntryPage(
        items=[
            DatasetImportEntry(
                **{k: v for k, v in r.items() if k in DatasetImportEntry.model_fields}
            )
            for r in window
        ],
        total=total,
        page=page,
        page_size=page_size,
    )


# ------------------------------------------------------------------ actions


@router.post('/datasets/imports/{import_id}/cancel', response_model=DatasetImportJob)
async def cancel_dataset_import(import_id: str) -> DatasetImportJob:
    return job_wire(await control.cancel(await _open(import_id)), project=_slug())


@router.post(
    '/datasets/imports/{import_id}/resume', response_model=DatasetImportJob, status_code=202
)
async def resume_dataset_import(import_id: str) -> DatasetImportJob:
    """Queue the rest of the import. Nothing is scanned in the request: the worker scans the
    dataset again and fails the job if it changed since the import started."""
    view = await _open(import_id)
    try:
        resumed = await control.resume(view)
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
    return job_wire(resumed, project=_slug())


@router.post('/datasets/imports/{import_id}/undo')
async def undo_dataset_import(
    import_id: str,
    body: DatasetUndoRequest,
    response: Response,
    opensearch: OpenSearchDep,
    registry: RegistryDep,
) -> DatasetUndoReportWire | DatasetImportJob:
    view = await _open(import_id)
    try:
        control.check_undoable(view)
    except runner.ImportNotUndoableError:
        raise api_error(
            409,
            'import_not_undoable',
            'an import that is running cannot be undone',
            import_id=import_id,
            project=_slug(),
        ) from None
    except runner.ImportBusyError as exc:
        raise api_error(
            409, 'import_busy', str(exc), import_id=exc.import_id, project=_slug()
        ) from None
    if body.dry_run:
        from src.config.region_fields import get_region_fields

        cfg = get_curation_config()
        ctx = UndoContext(
            import_id=import_id,
            opensearch=getattr(opensearch, 'client', opensearch),
            images_index=cfg.images_index,
            items_index=cfg.items_index,
            crop_cache_dir=cfg.crop_cache_dir,
            region_fields=get_region_fields(),
            registry=registry,
        )
        report = await undo_import(
            ctx,
            control.ledger_of(view),
            dry_run=True,
            remove_images=body.remove_images,
            deprecate_created_classes=body.deprecate_created_classes,
            created_classes=view.record.mapping.get('created_classes') or {},
        )
        return DatasetUndoReportWire(**report.to_dict())
    try:
        started = await control.start_undo(
            view,
            remove_images=body.remove_images,
            deprecate_created_classes=body.deprecate_created_classes,
        )
    except runner.ImportBusyError as exc:
        raise api_error(
            409, 'import_busy', str(exc), import_id=exc.import_id, project=_slug()
        ) from None
    response.status_code = 202
    return job_wire(started, project=_slug())


__all__ = ['router']
