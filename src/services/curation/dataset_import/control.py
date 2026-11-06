"""Start, resume, undo, cancel and read imports: the one place the routes call.

An import is its ``dataset_imports`` row plus one or two jobs (:mod:`.status`). A start inserts
the row and its ``dataset_import`` job in one transaction (``start_job`` ``with_rows``). The
``dataset_import`` and ``dataset_import_undo`` kinds share one exclusivity group, so at most one
import of a project runs or undoes at a time.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from src.config.project_context import current_project_id
from src.core.exceptions import RequestRejectedError
from src.db.repos import jobs as job_repo
from src.db.session import async_session_factory, project_tx
from src.services.curation.dataset_import import rows
from src.services.curation.dataset_import.actions import resume_blocker, undo_blocker
from src.services.curation.dataset_import.runner import (
    ImportBusyError,
    ImportNotResumableError,
    ImportNotUndoableError,
    pinned_mapping,
    scan_document,
)
from src.services.curation.dataset_import.status import ACTIVE_STATUSES, state_of
from src.services.jobs import service as jobs


if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from src.services.curation.dataset_import.options import DatasetImportRequest
    from src.services.curation.dataset_import.prepare import PreparedImport


@dataclass(frozen=True)
class ImportView:
    """An import as the routes serve it."""

    record: rows.ImportRecord
    run: job_repo.JobRow | None
    undo: job_repo.JobRow | None
    status: str
    mode: str
    other_live_id: str | None

    @property
    def import_id(self) -> str:
        return str(self.record.id)

    @property
    def state(self) -> dict[str, Any]:
        return {'status': self.status, 'mode': self.mode}


async def _job(session: AsyncSession, job_id: uuid.UUID | None) -> job_repo.JobRow | None:
    if job_id is None:
        return None
    return await session.run_sync(lambda sync: job_repo.get(sync, job_id))


async def _views(session: AsyncSession) -> list[ImportView]:
    records = await rows.list_newest_first(session)
    pairs = [
        (r, await _job(session, r.job_id), await _job(session, r.undo_job_id)) for r in records
    ]
    states = [(r, run, undo, state_of(r, run, undo)) for r, run, undo in pairs]
    live = [str(r.id) for r, _run, _undo, st in states if st['status'] in ACTIVE_STATUSES]
    return [
        ImportView(
            r,
            run,
            undo,
            st['status'],
            st['mode'],
            next((i for i in live if i != str(r.id)), None),
        )
        for r, run, undo, st in states
    ]


async def list_views() -> list[ImportView]:
    """Every import of the bound project, newest first."""
    async with async_session_factory()() as session, project_tx(session, current_project_id()):
        return await _views(session)


async def view_of(import_id: str) -> ImportView | None:
    """The import, or ``None`` when the id is malformed or names none in this project."""
    try:
        wanted = uuid.UUID(import_id)
    except ValueError:
        return None
    return next((v for v in await list_views() if v.record.id == wanted), None)


async def live_import_id() -> str | None:
    """The import that is running or being undone, if any."""
    return next((v.import_id for v in await list_views() if v.status in ACTIVE_STATUSES), None)


async def prior_of(prepared: PreparedImport) -> ImportView | None:
    """The newest earlier import of this same request, by its key (or by ``reuse_key`` when
    its created classes now exist)."""
    keys = [k for k in (prepared.import_key, prepared.reuse_key) if k]
    async with async_session_factory()() as session, project_tx(session, current_project_id()):
        found = await rows.latest_with_key(session, keys)
    return None if found is None else await view_of(str(found.id))


async def _holder(active_job_id: str | None) -> str | None:
    if active_job_id is None:
        return None
    async with async_session_factory()() as session, project_tx(session, current_project_id()):
        holder = await rows.holder_of_active_job(session, uuid.UUID(active_job_id))
    return None if holder is None else str(holder)


async def _busy(exc: RequestRejectedError) -> ImportBusyError | None:
    detail = exc.detail if isinstance(exc.detail, dict) else {}
    if detail.get('error') != 'job_busy':
        return None
    return ImportBusyError(await _holder(detail.get('active_job_id')))


async def start(
    prepared: PreparedImport,
    request: DatasetImportRequest,
    issues: list[dict[str, Any]],
    *,
    freeze_test: bool,
) -> ImportView:
    """Insert the import and queue its run. Raises :class:`ImportBusyError` while another
    import of the project runs."""
    project = current_project_id()
    import_id = uuid.uuid4()
    mapping = pinned_mapping(prepared)
    scan = scan_document(prepared, request, issues, freeze_test=freeze_test)

    async def with_rows(session: AsyncSession, job: job_repo.JobRow) -> None:
        await rows.insert_import(
            session,
            import_id=import_id,
            project_id=project,
            job_id=job.id,
            source=request.source.path,
            source_format=prepared.scan.format,
            mapping=mapping,
            request=request.model_dump(mode='json'),
            scan=scan,
        )

    try:
        async with async_session_factory()() as session:
            await jobs.start_job(
                session,
                'dataset_import',
                project,
                params={'import_id': str(import_id)},
                with_rows=with_rows,
            )
    except RequestRejectedError as exc:
        busy = await _busy(exc)
        if busy is not None:
            raise busy from None
        raise
    view = await view_of(str(import_id))
    assert view is not None
    return view


def check_resumable(view: ImportView) -> None:
    if resume_blocker(view.state):
        raise ImportNotResumableError(view.import_id)
    if view.other_live_id is not None:
        raise ImportBusyError(view.other_live_id)


def check_undoable(view: ImportView) -> None:
    """An import that finished (or was cut off) can be undone; one a live job is still running
    cannot, and neither can one while another import runs."""
    if undo_blocker(view.state):
        raise ImportNotUndoableError(view.import_id)
    if view.other_live_id is not None:
        raise ImportBusyError(view.other_live_id)


async def resume(view: ImportView) -> ImportView:
    """Continue an interrupted import under its own job; a failed or cancelled run (terminal)
    continues as a new job on the same import. Either way the worker scans the dataset again
    and refuses if it changed."""
    check_resumable(view)
    project = current_project_id()
    try:
        async with async_session_factory()() as session:
            if view.run is not None and view.run.status == 'interrupted':
                await jobs.resume_job(session, view.run.id, project)
            else:

                async def point_at(rows_session: AsyncSession, job: job_repo.JobRow) -> None:
                    await rows.set_run_job(rows_session, view.record.id, job.id)

                await jobs.start_job(
                    session,
                    'dataset_import',
                    project,
                    params={'import_id': view.import_id},
                    with_rows=point_at,
                )
    except RequestRejectedError as exc:
        busy = await _busy(exc)
        if busy is not None:
            raise busy from None
        raise
    refreshed = await view_of(view.import_id)
    assert refreshed is not None
    return refreshed


async def start_undo(
    view: ImportView, *, remove_images: bool, deprecate_created_classes: bool
) -> ImportView:
    check_undoable(view)
    project = current_project_id()

    async def point_at(session: AsyncSession, job: job_repo.JobRow) -> None:
        await rows.set_undo_job(session, view.record.id, job.id)

    try:
        async with async_session_factory()() as session:
            await jobs.start_job(
                session,
                'dataset_import_undo',
                project,
                params={
                    'import_id': view.import_id,
                    'remove_images': remove_images,
                    'deprecate_created_classes': deprecate_created_classes,
                },
                with_rows=point_at,
            )
    except RequestRejectedError as exc:
        busy = await _busy(exc)
        if busy is not None:
            raise busy from None
        raise
    refreshed = await view_of(view.import_id)
    assert refreshed is not None
    return refreshed


async def cancel(view: ImportView) -> ImportView:
    """Ask the running import to stop; an import that is not running is returned as it is."""
    from src.services.curation.dataset_import.actions import cancel_blocker

    if cancel_blocker(view.state) is not None or view.run is None:
        return view
    async with async_session_factory()() as session:
        try:
            await jobs.cancel_job(session, view.run.id, current_project_id(), requested_by='api')
        except RequestRejectedError as exc:
            if not (isinstance(exc.detail, dict) and exc.detail.get('error') == 'job_finished'):
                raise
    refreshed = await view_of(view.import_id)
    assert refreshed is not None
    return refreshed


def ledger_of(view_or_id: ImportView | str) -> rows.Ledger:
    import_id = view_or_id if isinstance(view_or_id, str) else view_or_id.import_id
    return rows.Ledger(rows.project_sessions(), import_id)


__all__ = [
    'ImportView',
    'cancel',
    'check_resumable',
    'check_undoable',
    'ledger_of',
    'list_views',
    'live_import_id',
    'prior_of',
    'resume',
    'start',
    'start_undo',
    'view_of',
]
