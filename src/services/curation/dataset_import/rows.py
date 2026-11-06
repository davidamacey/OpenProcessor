"""An import as rows: ``dataset_imports`` and the ``dataset_import_images`` ledger.

``dataset_imports`` holds what the old per-import directory held. ``options`` is the request as
accepted (source, mapping, options); ``mapping`` is the resolved mapping with its real class ids
plus the pinned region profile (a resume consumes THIS, never a re-resolution); ``scan`` is the
scan summary and the run facts (``import_key``, ``source_sha``, the resolved ``freeze_test``,
``test_split``, ``freeze_record``); ``report`` is the accumulated import report.

The ledger has one row per image (``image_key`` is the scan entry's ``rel_path``). A row is born
``pending``, carries the write-ahead plan in ``detail`` before any document is touched, and ends
``written``, ``skipped`` or ``failed``; a finished undo marks it ``undone``. ``detail`` holds the
per-image undo facts (item actions, snapshot index, holdout prior, ``image_created``, the boxes
with their ``box_id`` and ``bbox_norm``) and the per-image report.
"""

from __future__ import annotations

import datetime as dt
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from sqlalchemy import Integer, cast, column, func, select, table, text, update
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, insert as pg_insert
from sqlalchemy.types import Text, Uuid

from src.db.models.datasets import DatasetImport
from src.services.curation.dataset_import.report import ImportReport


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence
    from contextlib import AbstractAsyncContextManager

    from sqlalchemy.ext.asyncio import AsyncSession

    from src.services.curation.dataset_import.scan import ScanEntry

Sessions = Callable[[], 'AbstractAsyncContextManager[AsyncSession]']
"""Opens a session in the import's project scope; commits on a clean exit."""

# `detail` comes with the near-duplicate revision; a lightweight table keeps this module
# independent of the ORM model until it carries the column.
_LEDGER = table(
    'dataset_import_images',
    column('import_id', Uuid()),
    column('image_key', Text),
    column('image_id', Text),
    column('item_ids', ARRAY(Text())),
    column('status', Text),
    column('error', Text),
    column('detail', JSONB),
)

_ORD = cast(_LEDGER.c.detail['ord'].astext, Integer)  # a row's place in chunk order

FINAL_STATUSES = ('written', 'skipped', 'failed')
_LEGACY_STATUS = {'written': 'ok', 'pending': 'pending', 'skipped': 'skipped', 'failed': 'failed'}
_COLUMN_STATUS = {'ok': 'written', 'skipped': 'skipped', 'failed': 'failed'}
SEED_BATCH = 1000
_ACTIVE_JOB_STATUSES = ('queued', 'running', 'cancelling', 'paused')


def project_sessions() -> Sessions:
    """Sessions in the bound project's scope, for a caller that is not a job."""
    from contextlib import asynccontextmanager

    from src.config.project_context import current_project_id
    from src.db.session import async_session_factory, project_tx

    project = current_project_id()

    @asynccontextmanager
    async def open_session() -> AsyncIterator[AsyncSession]:
        async with async_session_factory()() as session, project_tx(session, project):
            yield session

    return open_session


@dataclass(frozen=True)
class ImportRecord:
    id: uuid.UUID
    job_id: uuid.UUID | None
    undo_job_id: uuid.UUID | None
    format: str
    mapping: dict[str, Any]
    request: dict[str, Any]
    scan: dict[str, Any]
    report: dict[str, Any]
    created_at: dt.datetime
    undone_at: dt.datetime | None


def _record(row: DatasetImport) -> ImportRecord:
    return ImportRecord(
        id=row.id,
        job_id=row.job_id,
        undo_job_id=row.undo_job_id,
        format=row.format,
        mapping=dict(row.mapping or {}),
        request=dict(row.options or {}),
        scan=dict(row.scan or {}),
        report=dict(row.report or {}),
        created_at=row.created_at,
        undone_at=row.undone_at,
    )


async def insert_import(
    session: AsyncSession,
    *,
    import_id: uuid.UUID,
    project_id: uuid.UUID,
    job_id: uuid.UUID,
    source: str,
    source_format: str,
    mapping: dict[str, Any],
    request: dict[str, Any],
    scan: dict[str, Any],
) -> None:
    """The import row, in the transaction that inserts its job (``start_job`` ``with_rows``)."""
    session.add(
        DatasetImport(
            id=import_id,
            project_id=project_id,
            job_id=job_id,
            source_key=source,
            format=source_format,
            mapping=mapping,
            options=request,
            scan=scan,
        )
    )
    await session.flush()


async def get(session: AsyncSession, import_id: uuid.UUID) -> ImportRecord | None:
    row = await session.get(DatasetImport, import_id)
    return None if row is None else _record(row)


async def list_newest_first(session: AsyncSession) -> list[ImportRecord]:
    rows = (
        await session.execute(select(DatasetImport).order_by(DatasetImport.created_at.desc()))
    ).scalars()
    return [_record(r) for r in rows]


async def of_job(session: AsyncSession, job_id: uuid.UUID) -> ImportRecord | None:
    """The import a job (its run or its undo) belongs to."""
    row = (
        await session.execute(
            select(DatasetImport).where(
                (DatasetImport.job_id == job_id) | (DatasetImport.undo_job_id == job_id)
            )
        )
    ).scalar_one_or_none()
    return None if row is None else _record(row)


async def latest_with_key(session: AsyncSession, keys: Sequence[str]) -> ImportRecord | None:
    """The newest import whose ``import_key`` is one of ``keys``."""
    if not keys:
        return None
    row = (
        await session.execute(
            select(DatasetImport)
            .where(DatasetImport.scan['import_key'].astext.in_(list(keys)))
            .order_by(DatasetImport.created_at.desc())
            .limit(1)
        )
    ).scalar_one_or_none()
    return None if row is None else _record(row)


async def set_run_job(session: AsyncSession, import_id: uuid.UUID, job_id: uuid.UUID) -> None:
    """Point the import at a new run (a failed or cancelled run resumes as a new job)."""
    await session.execute(
        update(DatasetImport).where(DatasetImport.id == import_id).values(job_id=job_id)
    )


async def set_undo_job(session: AsyncSession, import_id: uuid.UUID, job_id: uuid.UUID) -> None:
    await session.execute(
        update(DatasetImport).where(DatasetImport.id == import_id).values(undo_job_id=job_id)
    )


async def set_mapping(session: AsyncSession, import_id: uuid.UUID, mapping: dict[str, Any]) -> None:
    await session.execute(
        update(DatasetImport).where(DatasetImport.id == import_id).values(mapping=mapping)
    )


async def merge_scan(session: AsyncSession, import_id: uuid.UUID, facts: dict[str, Any]) -> None:
    """Merge ``facts`` into the scan document (one atomic ``||``, no read-modify-write)."""
    await session.execute(
        update(DatasetImport)
        .where(DatasetImport.id == import_id)
        .values(scan=func.coalesce(DatasetImport.scan, text("'{}'::jsonb")).op('||')(facts))
    )


async def set_report(session: AsyncSession, import_id: uuid.UUID, report: dict[str, Any]) -> None:
    await session.execute(
        update(DatasetImport).where(DatasetImport.id == import_id).values(report=report)
    )


async def mark_undone(session: AsyncSession, import_id: uuid.UUID) -> None:
    """The import is undone: its ledger rows and the import row say so."""
    await session.execute(
        update(_LEDGER)
        .where(_LEDGER.c.import_id == import_id, _LEDGER.c.status.in_(('written', 'skipped')))
        .values(status='undone')
    )
    await session.execute(
        update(DatasetImport)
        .where(DatasetImport.id == import_id)
        .values(undone_at=dt.datetime.now(dt.UTC))
    )


async def holder_of_active_job(session: AsyncSession, job_id: uuid.UUID | None) -> uuid.UUID | None:
    """The import the active job ``job_id`` belongs to (for the ``import_busy`` answer)."""
    if job_id is None:
        return None
    record = await of_job(session, job_id)
    return None if record is None else record.id


def legacy_row(
    key: str, status: str, image_id: str | None, error: str | None, detail: dict[str, Any]
) -> dict[str, Any]:
    """A ledger row in the shape the importer and the undo have always read."""
    out: dict[str, Any] = {
        'rel_path': key,
        'status': _LEGACY_STATUS.get(status, status),
        **{k: v for k, v in detail.items() if k != 'ord'},
    }
    if image_id is not None:
        out['image_id'] = image_id
    if error:
        out['error'] = error
    return out


@dataclass(frozen=True)
class Claimed:
    key: str
    order: int
    detail: dict[str, Any]

    @property
    def prior(self) -> dict[str, Any] | None:
        """The write-ahead row an earlier attempt left on this image, if it got that far."""
        return self.detail if self.detail.get('items') is not None else None


class Ledger:
    """One import's ledger rows. Every method opens its own short transaction."""

    def __init__(self, sessions: Sessions, import_id: uuid.UUID | str) -> None:
        self.sessions = sessions
        self._id = uuid.UUID(str(import_id))
        self.import_id = str(self._id)

    # ----------------------------------------------------------------- seeding / claiming

    async def seed(self, entries: Sequence[ScanEntry]) -> None:
        """One ``pending`` row per scan entry (``ord`` is its place in chunk order); a row that
        exists already, from an earlier attempt, is left alone."""
        for start in range(0, len(entries), SEED_BATCH):
            values = [
                {
                    'import_id': self._id,
                    'image_key': e.rel_path,
                    'status': 'pending',
                    'detail': {
                        'ord': start + offset,
                        'source_stem': e.source_stem,
                        'split': e.split,
                        'label_state': e.label_state,
                    },
                }
                for offset, e in enumerate(entries[start : start + SEED_BATCH])
            ]
            async with self.sessions() as session:
                await session.execute(pg_insert(_LEDGER).values(values).on_conflict_do_nothing())

    async def claim(self, limit: int, *, after: int) -> list[Claimed]:
        """The next ``limit`` pending images past order ``after``. ``FOR UPDATE SKIP LOCKED``: a
        second runner on the same import skips what another transaction holds."""
        async with self.sessions() as session:
            rows = (
                await session.execute(
                    select(_LEDGER.c.image_key, _LEDGER.c.detail)
                    .where(
                        _LEDGER.c.import_id == self._id,
                        _LEDGER.c.status == 'pending',
                        after < _ORD,
                    )
                    .order_by(_ORD)
                    .limit(limit)
                    .with_for_update(skip_locked=True)
                )
            ).all()
        return [Claimed(k, int(d['ord']), dict(d)) for k, d in rows]

    # ---------------------------------------------------------------------------- writes

    async def write_ahead(self, key: str, row: dict[str, Any]) -> None:
        """Commit what this image is about to do, before any document is touched."""
        await self._patch(key, 'pending', row)

    async def finish(self, key: str, row: dict[str, Any]) -> None:
        """End the image: ``row['status']`` is ``ok``, ``skipped`` or ``failed``."""
        await self._patch(key, _COLUMN_STATUS[row['status']], row)

    async def _patch(self, key: str, status: str, row: dict[str, Any]) -> None:
        detail = {k: v for k, v in row.items() if k not in ('status', 'rel_path', 'image_id')}
        items = detail.get('items') or []
        error = (row.get('error') or None) if status == 'failed' else None
        async with self.sessions() as session:
            written = await session.execute(
                update(_LEDGER)
                .where(_LEDGER.c.import_id == self._id, _LEDGER.c.image_key == key)
                .values(
                    status=status,
                    image_id=row.get('image_id'),
                    item_ids=[i['crop_id'] for i in items if i.get('crop_id')],
                    error=error,
                    detail=_LEDGER.c.detail.op('||')(detail),
                )
            )
        if written.rowcount != 1:  # type: ignore[attr-defined]
            raise LookupError(f'import {self.import_id} has no ledger row for {key!r}')

    # ----------------------------------------------------------------------------- reads

    async def rows(self, *, statuses: Sequence[str] | None = None) -> AsyncIterator[dict[str, Any]]:
        """Every image's row in chunk order, a page at a time."""
        last = -1
        while True:
            query = (
                select(
                    _LEDGER.c.image_key,
                    _LEDGER.c.status,
                    _LEDGER.c.image_id,
                    _LEDGER.c.error,
                    _LEDGER.c.detail,
                )
                .where(_LEDGER.c.import_id == self._id, last < _ORD)
                .order_by(_ORD)
                .limit(SEED_BATCH)
            )
            if statuses is not None:
                query = query.where(_LEDGER.c.status.in_(list(statuses)))
            async with self.sessions() as session:
                page = (await session.execute(query)).all()
            if not page:
                return
            for key, status, image_id, error, detail in page:
                last = int(detail['ord'])
                yield legacy_row(key, status, image_id, error, dict(detail))

    async def page(
        self,
        *,
        split: str | None,
        label_state: str | None,
        status: str | None,
        offset: int,
        limit: int,
    ) -> tuple[list[dict[str, Any]], int]:
        """One page of rows and the filtered total, for the entries route."""
        where = [_LEDGER.c.import_id == self._id]
        if split is not None:
            where.append(_LEDGER.c.detail['split'].astext == split)
        if label_state is not None:
            where.append(_LEDGER.c.detail['label_state'].astext == label_state)
        if status is not None:
            wanted = [c for c, legacy in _LEGACY_STATUS.items() if legacy == status]
            where.append(_LEDGER.c.status.in_(wanted or [status]))
        async with self.sessions() as session:
            total = (
                await session.execute(select(func.count()).select_from(_LEDGER).where(*where))
            ).scalar_one()
            found = (
                await session.execute(
                    select(
                        _LEDGER.c.image_key,
                        _LEDGER.c.status,
                        _LEDGER.c.image_id,
                        _LEDGER.c.error,
                        _LEDGER.c.detail,
                    )
                    .where(*where)
                    .order_by(_ORD)
                    .offset(offset)
                    .limit(limit)
                )
            ).all()
        return [legacy_row(k, s, i, e, dict(d)) for k, s, i, e, d in found], int(total)

    async def finished_report(self) -> tuple[ImportReport, int]:
        """The sum of the per-image reports of the finished images, and how many there are."""
        report = ImportReport()
        count = 0
        async for row in self.rows(statuses=(*FINAL_STATUSES, 'undone')):
            report.add(ImportReport.from_dict(row.get('report') or {}))
            count += 1
        return report, count

    async def save_report(self, report: ImportReport, *, issues: list[dict[str, Any]]) -> None:
        async with self.sessions() as session:
            await set_report(session, self._id, {**report.to_dict(), 'job_issues': issues})

    async def mark_undone(self) -> None:
        async with self.sessions() as session:
            await mark_undone(session, self._id)


__all__ = [
    'FINAL_STATUSES',
    'Claimed',
    'ImportRecord',
    'Ledger',
    'Sessions',
    'get',
    'holder_of_active_job',
    'insert_import',
    'latest_with_key',
    'legacy_row',
    'list_newest_first',
    'mark_undone',
    'merge_scan',
    'of_job',
    'project_sessions',
    'set_mapping',
    'set_report',
    'set_run_job',
    'set_undo_job',
]
