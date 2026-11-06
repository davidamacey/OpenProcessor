"""The class registry in PostgreSQL: one registry row and its classes per project.

Every function takes a caller-owned session that is already inside a ``project_tx`` (or the
upgrade's ``deployment_tx``) and never commits. Writes lock the registry row
(``SELECT ... FOR UPDATE``, https://www.postgresql.org/docs/current/sql-select.html#SQL-FOR-UPDATE-SHARE)
so two writers of one project serialise and ids are allocated from ``next_class_id`` and never
reused. A name compares by its ``name_key`` (:func:`src.utils.class_names.normalize_class_name`);
this module is the only writer of that column. The partial unique indexes
(``uq_project_classes_active_name``, ``uq_project_classes_hotkey``) are the last line of defence.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from sqlalchemy import delete, func, insert, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import IntegrityError

from src.db.models.classes import ClassRegistry, ProjectClass, SearchProjectionTask
from src.utils.class_names import normalize_class_name


if TYPE_CHECKING:
    import uuid
    from collections.abc import Sequence

    from sqlalchemy.ext.asyncio import AsyncSession

ACTIVE_NAME_INDEX = 'uq_project_classes_active_name'
HOTKEY_INDEX = 'uq_project_classes_hotkey'


class ClassRegistryError(RuntimeError):
    """A registry validation or mutation failure."""


class ClassNotFoundError(ClassRegistryError):
    """No class has the id."""


class ClassNameInvalidError(ClassRegistryError):
    """A name with no letter or digit (it has no ``name_key``)."""


class ClassNameTakenError(ClassRegistryError):
    """An active class already has this name."""


class HotkeyTakenError(ClassRegistryError):
    """Another class of the project already has this hotkey."""


@dataclass(frozen=True)
class ClassRow:
    class_id: int
    name: str
    class_group: str | None
    deprecated: bool
    merged_into: int | None
    hotkey_letter: str | None
    notes: str
    added_at: dt.datetime
    updated_at: dt.datetime
    name_key: str


@dataclass(frozen=True)
class RegistryRows:
    revision: int
    next_class_id: int
    classes: list[ClassRow]


@dataclass(frozen=True)
class ClassLoad:
    """A class as the 0.6.0 registry document holds it; ``name_key`` is derived on load."""

    class_id: int
    name: str
    class_group: str | None = None
    deprecated: bool = False
    merged_into: int | None = None
    hotkey_letter: str | None = None
    notes: str = ''
    added_at: dt.datetime | None = None
    updated_at: dt.datetime | None = None


def name_key(name: str) -> str:
    """The comparison key of a class name; raises when nothing slug-safe remains."""
    key = normalize_class_name(name)
    if not key:
        raise ClassNameInvalidError('class_name must contain a letter or digit')
    return key


def _row(c: ProjectClass) -> ClassRow:
    return ClassRow(
        class_id=c.class_id,
        name=c.name,
        class_group=c.class_group,
        deprecated=c.deprecated,
        merged_into=c.merged_into,
        hotkey_letter=c.hotkey_letter,
        notes=c.notes,
        added_at=c.added_at,
        updated_at=c.updated_at,
        name_key=c.name_key,
    )


async def ensure_registry(session: AsyncSession, project_id: uuid.UUID) -> None:
    """Create the project's empty registry row when it has none."""
    await session.execute(
        pg_insert(ClassRegistry).values(project_id=project_id).on_conflict_do_nothing()
    )


async def _lock(session: AsyncSession, project_id: uuid.UUID) -> ClassRegistry:
    """The registry row, locked for this transaction; created first when absent."""
    stmt = select(ClassRegistry).where(ClassRegistry.project_id == project_id).with_for_update()
    reg = (await session.execute(stmt)).scalar_one_or_none()
    if reg is None:
        await ensure_registry(session, project_id)
        reg = (await session.execute(stmt)).scalar_one()
    return reg


async def _bump(session: AsyncSession, project_id: uuid.UUID, **extra: Any) -> None:
    await session.execute(
        update(ClassRegistry)
        .where(ClassRegistry.project_id == project_id)
        .values(revision=ClassRegistry.revision + 1, updated_at=func.now(), **extra)
    )


async def current_revision(session: AsyncSession, project_id: uuid.UUID) -> int | None:
    """The registry revision (one indexed read), or ``None`` when the project has no registry."""
    return (
        await session.execute(
            select(ClassRegistry.revision).where(ClassRegistry.project_id == project_id)
        )
    ).scalar_one_or_none()


async def read_registry(session: AsyncSession, project_id: uuid.UUID) -> RegistryRows | None:
    """The revision, the id counter and every class (deprecated ones included), by id."""
    reg = (
        await session.execute(select(ClassRegistry).where(ClassRegistry.project_id == project_id))
    ).scalar_one_or_none()
    if reg is None:
        return None
    rows = (
        (
            await session.execute(
                select(ProjectClass)
                .where(ProjectClass.project_id == project_id)
                .order_by(ProjectClass.class_id)
            )
        )
        .scalars()
        .all()
    )
    return RegistryRows(reg.revision, reg.next_class_id, [_row(c) for c in rows])


async def get_class(session: AsyncSession, project_id: uuid.UUID, class_id: int) -> ClassRow | None:
    c = await _fetch(session, project_id, class_id)
    return None if c is None else _row(c)


async def _fetch(
    session: AsyncSession, project_id: uuid.UUID, class_id: int, *, lock: bool = False
) -> ProjectClass | None:
    """One class row read fresh (``populate_existing``), so an earlier bulk UPDATE is seen."""
    stmt = (
        select(ProjectClass)
        .where(ProjectClass.project_id == project_id, ProjectClass.class_id == class_id)
        .execution_options(populate_existing=True)
    )
    if lock:
        stmt = stmt.with_for_update()
    return (await session.execute(stmt)).scalar_one_or_none()


async def find_by_name(session: AsyncSession, project_id: uuid.UUID, name: str) -> ClassRow | None:
    """The class ``name`` resolves to: an active class wins over a deprecated one, then the
    exact spelling, then the lowest id (:func:`src.utils.class_names.resolve_class_by_name`).
    ``None`` for a name with no key or no match."""
    key = normalize_class_name(name)
    if not key:
        return None
    rows = (
        (
            await session.execute(
                select(ProjectClass).where(
                    ProjectClass.project_id == project_id, ProjectClass.name_key == key
                )
            )
        )
        .scalars()
        .all()
    )
    if not rows:
        return None
    best = min(rows, key=lambda c: (c.deprecated, c.name != name, c.class_id))
    return _row(best)


async def _insert_class(
    session: AsyncSession,
    reg: ClassRegistry,
    name: str,
    group: str,
    notes: str,
    hotkey_letter: str | None,
) -> ClassRow:
    stripped = name.strip()
    key = name_key(stripped)
    new_id = reg.next_class_id
    try:
        async with session.begin_nested():
            await session.execute(
                insert(ProjectClass).values(
                    project_id=reg.project_id,
                    class_id=new_id,
                    name=stripped,
                    class_group=group,
                    notes=notes,
                    hotkey_letter=hotkey_letter,
                    name_key=key,
                )
            )
    except IntegrityError as exc:
        raise _translate(exc, stripped) from exc
    await _bump(session, reg.project_id, next_class_id=new_id + 1)
    created = await get_class(session, reg.project_id, new_id)
    assert created is not None
    return created


def _translate(exc: IntegrityError, name: str) -> ClassRegistryError:
    text = str(exc.orig)
    if ACTIVE_NAME_INDEX in text:
        return ClassNameTakenError(f'duplicate class_name {name!r}')
    if HOTKEY_INDEX in text:
        return HotkeyTakenError('hotkey_letter is already used by another class')
    return ClassRegistryError(str(exc.orig))


async def ensure_class(
    session: AsyncSession,
    project_id: uuid.UUID,
    name: str,
    group: str = 'unknown',
    notes: str = '',
) -> tuple[ClassRow, bool]:
    """Idempotent by name: the active class called ``name``, created when absent.

    The unlocked read answers the common case; a miss locks the registry row, re-checks and
    allocates, so concurrent writers of one name get one id."""
    found = await find_by_name(session, project_id, name)
    if found is not None and not found.deprecated:
        return found, False
    name_key(name)
    reg = await _lock(session, project_id)
    found = await find_by_name(session, project_id, name)
    if found is not None and not found.deprecated:
        return found, False
    return await _insert_class(session, reg, name, group, notes, None), True


async def add_class(
    session: AsyncSession,
    project_id: uuid.UUID,
    name: str,
    group: str = 'unknown',
    notes: str = '',
    hotkey_letter: str | None = None,
) -> ClassRow:
    """Append a class; an active class with the same key is a :class:`ClassNameTakenError`."""
    name_key(name)
    reg = await _lock(session, project_id)
    found = await find_by_name(session, project_id, name)
    if found is not None and not found.deprecated:
        raise ClassNameTakenError(f'duplicate class_name {name.strip()!r} (id={found.class_id})')
    return await _insert_class(session, reg, name, group, notes, hotkey_letter)


async def _queue_projection(
    session: AsyncSession, project_id: uuid.UUID, kind: str, payload: dict[str, Any]
) -> None:
    await session.execute(
        insert(SearchProjectionTask).values(project_id=project_id, kind=kind, payload=payload)
    )


async def update_class(
    session: AsyncSession,
    project_id: uuid.UUID,
    class_id: int,
    *,
    name: str | None = None,
    group: str | None = None,
    hotkey_letter: str | None = None,
    set_hotkey: bool = False,
) -> ClassRow:
    """Name, group and hotkey change together. A rename writes a ``class_rename`` outbox row in
    the same transaction. ``hotkey_letter`` applies only with ``set_hotkey`` (``None`` clears)."""
    await _lock(session, project_id)
    target = await _fetch(session, project_id, class_id, lock=True)
    if target is None:
        raise ClassNotFoundError(f'class_id {class_id} not found')
    values: dict[str, Any] = {}
    renamed = False
    if name is not None:
        stripped = name.strip()
        if not stripped:
            raise ClassRegistryError('new_name must be non-empty')
        if stripped != target.name:
            key = name_key(stripped)
            clash = await find_by_name(session, project_id, stripped)
            if (
                clash is not None
                and not clash.deprecated
                and clash.class_id != class_id
                and clash.name_key == key
            ):
                raise ClassNameTakenError(
                    f'rename target {stripped!r} already in use (id={clash.class_id})'
                )
            values.update(name=stripped, name_key=key)
            renamed = True
    if group is not None:
        values['class_group'] = group
    if set_hotkey:
        values['hotkey_letter'] = hotkey_letter
    if values:
        try:
            async with session.begin_nested():
                await session.execute(
                    update(ProjectClass)
                    .where(ProjectClass.project_id == project_id, ProjectClass.class_id == class_id)
                    .values(updated_at=func.now(), **values)
                )
        except IntegrityError as exc:
            raise _translate(exc, str(values.get('name', ''))) from exc
        await _bump(session, project_id)
        if renamed:
            await _queue_projection(
                session,
                project_id,
                'class_rename',
                {'class_id': class_id, 'new_name': values['name']},
            )
    row = await get_class(session, project_id, class_id)
    assert row is not None
    return row


async def set_deprecated(
    session: AsyncSession, project_id: uuid.UUID, class_id: int, deprecated: bool
) -> ClassRow:
    """Deprecating clears the hotkey. Restoring a class whose name an active class holds is a
    :class:`ClassNameTakenError` (the unique index violation, caught in a savepoint)."""
    await _lock(session, project_id)
    target = await _fetch(session, project_id, class_id, lock=True)
    if target is None:
        raise ClassNotFoundError(f'class_id {class_id} not found')
    values: dict[str, Any] = {'deprecated': deprecated, 'updated_at': func.now()}
    if deprecated:
        values['hotkey_letter'] = None
    else:
        values['merged_into'] = None
    try:
        async with session.begin_nested():
            await session.execute(
                update(ProjectClass)
                .where(ProjectClass.project_id == project_id, ProjectClass.class_id == class_id)
                .values(**values)
            )
    except IntegrityError as exc:
        if ACTIVE_NAME_INDEX in str(exc.orig):
            msg = f'cannot restore class_id {class_id}: name {target.name!r} already in use'
            raise ClassNameTakenError(msg) from exc
        raise _translate(exc, target.name) from exc
    await _bump(session, project_id)
    row = await get_class(session, project_id, class_id)
    assert row is not None
    return row


async def merge_class(
    session: AsyncSession, project_id: uuid.UUID, source_id: int, target_id: int
) -> dict[str, Any]:
    """Deprecate ``source_id`` with ``merged_into=target_id`` and write a ``class_merge`` outbox
    row in the same transaction. Items are not relabelled here; the id stays burned."""
    if source_id == target_id:
        raise ClassRegistryError('cannot merge a class into itself')
    await _lock(session, project_id)
    source = await get_class(session, project_id, source_id)
    target = await get_class(session, project_id, target_id)
    if source is None:
        raise ClassNotFoundError(f'source class_id {source_id} not found')
    if target is None:
        raise ClassNotFoundError(f'target class_id {target_id} not found')
    if target.deprecated:
        raise ClassRegistryError(f'target class_id {target_id} is deprecated; cannot merge into it')
    await session.execute(
        update(ProjectClass)
        .where(ProjectClass.project_id == project_id, ProjectClass.class_id == source_id)
        .values(deprecated=True, merged_into=target_id, hotkey_letter=None, updated_at=func.now())
    )
    await _bump(session, project_id)
    await _queue_projection(
        session, project_id, 'class_merge', {'source_ids': [source_id], 'target_id': target_id}
    )
    return {
        'source_id': source_id,
        'target_id': target_id,
        'deprecated': True,
        'source_name': source.name,
        'target_name': target.name,
    }


async def set_hotkeys(
    session: AsyncSession,
    project_id: uuid.UUID,
    letters: dict[int, str | None],
    *,
    only_if_unset: bool = False,
) -> dict[int, str | None]:
    """Set several hotkeys at once (a swap included). With ``only_if_unset`` a letter is written
    only where the class has none now. Returns the letters actually applied."""
    await _lock(session, project_id)
    current = {
        c.class_id: c.hotkey_letter
        for c in (await read_registry(session, project_id) or RegistryRows(0, 0, [])).classes
    }
    applied: dict[int, str | None] = {}
    for cid, letter in letters.items():
        if cid not in current:
            continue
        if only_if_unset and current[cid] is not None:
            continue
        if current[cid] != letter:
            applied[cid] = letter
    if not applied:
        return applied
    where = ProjectClass.project_id == project_id
    try:
        async with session.begin_nested():
            await session.execute(
                update(ProjectClass)
                .where(where, ProjectClass.class_id.in_(applied))
                .values(hotkey_letter=None)
            )
            for cid, letter in applied.items():
                if letter is not None:
                    await session.execute(
                        update(ProjectClass)
                        .where(where, ProjectClass.class_id == cid)
                        .values(hotkey_letter=letter, updated_at=func.now())
                    )
    except IntegrityError as exc:
        raise _translate(exc, '') from exc
    await _bump(session, project_id)
    return applied


async def copy_registry(
    session: AsyncSession, source_project: uuid.UUID, target_project: uuid.UUID
) -> int:
    """Give ``target_project`` the classes and id counter of ``source_project`` (project clone),
    replacing what it has. Returns the number of classes copied. Runs under ``deployment_tx``
    when the two projects differ, since one scope sees one project."""
    source = await read_registry(session, source_project)
    await _lock(session, target_project)
    await session.execute(
        update(ProjectClass)
        .where(ProjectClass.project_id == target_project)
        .values(merged_into=None)
    )
    await session.execute(delete(ProjectClass).where(ProjectClass.project_id == target_project))
    classes = source.classes if source is not None else []
    await _insert_many(session, target_project, classes)
    await _bump(
        session, target_project, next_class_id=source.next_class_id if source is not None else 0
    )
    return len(classes)


async def _insert_many(
    session: AsyncSession, project_id: uuid.UUID, classes: Sequence[ClassRow | ClassLoad]
) -> None:
    """Insert in two passes: ``merged_into`` points at a row that may come later."""
    if not classes:
        return
    now = dt.datetime.now(dt.UTC)
    rows = [
        {
            'project_id': project_id,
            'class_id': c.class_id,
            'name': c.name.strip(),
            'class_group': c.class_group,
            'deprecated': c.deprecated,
            'hotkey_letter': c.hotkey_letter,
            'notes': c.notes,
            'added_at': c.added_at or now,
            'updated_at': c.updated_at or c.added_at or now,
            'name_key': name_key(c.name),
        }
        for c in sorted(classes, key=lambda c: c.class_id)
    ]
    await session.execute(insert(ProjectClass), rows)
    for c in classes:
        if c.merged_into is not None:
            await session.execute(
                update(ProjectClass)
                .where(ProjectClass.project_id == project_id, ProjectClass.class_id == c.class_id)
                .values(merged_into=c.merged_into)
            )


async def load_registry(
    session: AsyncSession,
    project_id: uuid.UUID,
    classes: Sequence[ClassLoad],
    *,
    revision: int,
    next_class_id: int | None = None,
    migrated_from: str | None = None,
    migrated_at: dt.datetime | None = None,
) -> None:
    """The 0.6.0 upgrade's write: the registry row and its classes with their ids and
    timestamps preserved, names through the same ``name_key`` path as a normal add. The id
    counter is ``next_class_id`` (the document's, which never goes back) or one past the highest
    id. Raises when the project already has a registry row or any unique rule is violated;
    nothing is committed here."""
    highest = max((c.class_id for c in classes), default=-1) + 1
    counter = highest if next_class_id is None else max(next_class_id, highest)
    if await current_revision(session, project_id) is not None:
        raise ClassRegistryError(f'project {project_id} already has a class registry')
    await session.execute(
        insert(ClassRegistry).values(
            project_id=project_id,
            next_class_id=counter,
            revision=revision,
            migrated_from=migrated_from,
            migrated_at=migrated_at,
        )
    )
    await _insert_many(session, project_id, classes)
