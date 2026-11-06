"""The bound project's class registry, read and written through PostgreSQL.

Each method opens its own short ``project_tx`` (https://docs.sqlalchemy.org/en/20/orm/session_transaction.html),
so a caller needs no session and no OpenSearch client. Reads return a :class:`RegistryDoc` (a
copy; the cache is keyed by ``class_registries.revision`` and costs one indexed read). The
project's id is looked up once per slug: a slug is retired when its project is deleted, so it
never names a second project.
"""

from __future__ import annotations

import datetime as dt
from typing import TYPE_CHECKING, Any

from sqlalchemy import select

from src.clients.curation_opensearch import ClassRegistryFile, RegistryClassEntry
from src.config.project_context import current_project
from src.db.models.projects import Project
from src.db.repos import classes as repo
from src.db.repos.classes import ClassRegistryError
from src.db.session import async_session_factory, deployment_tx, project_tx


if TYPE_CHECKING:
    import uuid
    from collections.abc import Awaitable, Callable

    from sqlalchemy.ext.asyncio import AsyncSession


class RegistryDoc(ClassRegistryFile):
    """The registry as read: the class list plus the id counter and the revision."""

    next_class_id: int = 0
    registry_revision: int = 0


def entry_from_row(row: repo.ClassRow) -> RegistryClassEntry:
    return RegistryClassEntry(
        class_id=row.class_id,
        class_name=row.name,
        group=row.class_group or 'unknown',
        added_at=row.added_at.isoformat(),
        deprecated=row.deprecated,
        notes=row.notes,
        merged_into=row.merged_into,
        hotkey_letter=row.hotkey_letter,
    )


def _doc(rows: repo.RegistryRows) -> RegistryDoc:
    return RegistryDoc(
        classes=[entry_from_row(c) for c in rows.classes],
        next_class_id=rows.next_class_id,
        registry_revision=rows.revision,
        updated_at=max(
            (c.updated_at for c in rows.classes), default=dt.datetime.now(dt.UTC)
        ).isoformat(),
    )


_project_ids: dict[str, uuid.UUID] = {}


async def project_id_for(slug: str) -> uuid.UUID:
    """The project's id by slug; ``ClassRegistryError`` when no such project exists."""
    found = _project_ids.get(slug)
    if found is not None:
        return found
    async with (
        async_session_factory()() as session,
        deployment_tx(session, 'resolve a project slug to its id for the class registry'),
    ):
        pid = (
            await session.execute(select(Project.id).where(Project.slug == slug))
        ).scalar_one_or_none()
    if pid is None:
        raise ClassRegistryError(f'unknown project {slug!r}')
    _project_ids[slug] = pid
    return pid


class ClassRegistry:
    """One project's registry. Obtain the bound project's with :func:`get_class_registry`."""

    def __init__(self, project: str) -> None:
        self.project = project
        self._cache: RegistryDoc | None = None

    async def _run[T](self, fn: Callable[[AsyncSession, uuid.UUID], Awaitable[T]]) -> T:
        pid = await project_id_for(self.project)
        async with async_session_factory()() as session, project_tx(session, pid):
            return await fn(session, pid)

    async def snapshot(self) -> RegistryDoc:
        """The current registry (a copy); an empty one for a project with no registry row yet."""
        cached = self._cache

        async def read(session: AsyncSession, pid: uuid.UUID) -> RegistryDoc:
            current = await repo.current_revision(session, pid)
            if current is None:
                return RegistryDoc()
            if cached is not None and cached.registry_revision == current:
                return cached
            rows = await repo.read_registry(session, pid)
            return RegistryDoc() if rows is None else _doc(rows)

        doc = await self._run(read)
        self._cache = doc
        return doc.model_copy(deep=True)

    async def _write[T](self, fn: Callable[[AsyncSession, uuid.UUID], Awaitable[T]]) -> T:
        result = await self._run(fn)
        self._cache = None
        return result

    async def copy_from(self, source_slug: str) -> int:
        """Give this project the classes and id counter of project ``source_slug`` (project
        clone), replacing its own. One transaction over both projects; returns the class count."""
        source = await project_id_for(source_slug)
        target = await project_id_for(self.project)
        async with (
            async_session_factory()() as session,
            deployment_tx(session, 'copy a class registry into a cloned project'),
        ):
            count = await repo.copy_registry(session, source, target)
        self._cache = None
        return count

    async def add_class(
        self,
        name: str,
        group: str = 'unknown',
        notes: str = '',
        hotkey_letter: str | None = None,
    ) -> int:
        """Append a class; refuses a duplicate active name. Returns the new id."""
        row = await self._write(
            lambda s, p: repo.add_class(s, p, name, group, notes, hotkey_letter)
        )
        return row.class_id

    async def ensure_class(
        self, name: str, group: str = 'unknown', notes: str = ''
    ) -> tuple[int, bool]:
        """Idempotent by name: ``(class_id, created)`` of the active class called ``name``."""
        entry, created = await self.ensure_entry(name, group, notes)
        return entry.class_id, created

    async def ensure_entry(
        self, name: str, group: str = 'unknown', notes: str = ''
    ) -> tuple[RegistryClassEntry, bool]:
        """:meth:`ensure_class` returning the whole entry (callers need its spelling)."""
        row, created = await self._write(lambda s, p: repo.ensure_class(s, p, name, group, notes))
        return entry_from_row(row), created

    async def update_class(
        self,
        class_id: int,
        *,
        name: str | None = None,
        group: str | None = None,
        hotkey_letter: str | None = None,
        set_hotkey: bool = False,
    ) -> RegistryClassEntry:
        """One write for a whole ``PUT``. ``hotkey_letter`` applies only with ``set_hotkey``."""
        row = await self._write(
            lambda s, p: repo.update_class(
                s,
                p,
                class_id,
                name=name,
                group=group,
                hotkey_letter=hotkey_letter,
                set_hotkey=set_hotkey,
            )
        )
        return entry_from_row(row)

    async def merge_class(self, source_id: int, target_id: int) -> dict[str, Any]:
        """Deprecate ``source_id`` into ``target_id``; items are not relabelled here."""
        return await self._write(lambda s, p: repo.merge_class(s, p, source_id, target_id))

    async def set_deprecated(self, class_id: int, deprecated: bool) -> RegistryClassEntry:
        row = await self._write(lambda s, p: repo.set_deprecated(s, p, class_id, deprecated))
        return entry_from_row(row)

    async def set_hotkeys(
        self, letters: dict[int, str | None], *, only_if_unset: bool = False
    ) -> dict[int, str | None]:
        return await self._write(
            lambda s, p: repo.set_hotkeys(s, p, letters, only_if_unset=only_if_unset)
        )


_registries: dict[str, ClassRegistry] = {}


def registry_for(slug: str) -> ClassRegistry:
    """The registry of project ``slug`` (one cached instance per slug per process)."""
    store = _registries.get(slug)
    if store is None:
        store = ClassRegistry(slug)
        _registries[slug] = store
    return store


def get_class_registry() -> ClassRegistry:
    """The bound project's registry."""
    return registry_for(current_project().record.slug)


__all__ = [
    'ClassRegistry',
    'ClassRegistryError',
    'RegistryDoc',
    'entry_from_row',
    'get_class_registry',
    'project_id_for',
    'registry_for',
]
