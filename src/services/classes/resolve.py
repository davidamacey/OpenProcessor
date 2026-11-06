"""Class name to id and id to name for the bound project's registry.

The one path every writer that sets a class on an item goes through (D-47): a name is turned
into ``(class_id, class_name)`` here, so an item stores the registry's id and spelling, never a
caller's. The per-process cache is keyed by ``class_registries.revision`` (one indexed read per
lookup, a full read only when the revision moved).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

from src.db.repos.classes import ClassNameInvalidError, ClassRegistryError
from src.services.classes.registry import get_class_registry
from src.utils.class_names import normalize_class_name, resolve_class_by_name


if TYPE_CHECKING:
    from src.services.classes.registry import ClassRegistry


class ResolvedClass(NamedTuple):
    """A registry class: its id and the name the registry spells it with (an item stores this
    name, not the spelling the caller asked for)."""

    class_id: int
    class_name: str


async def ensure_class_by_name(
    registry: ClassRegistry, name: str, *, group: str, notes: str = ''
) -> ResolvedClass:
    """The active class named ``name``, added under ``group`` first when the registry lacks it.

    Names compare by :func:`~src.utils.class_names.normalize_class_name`; a new class keeps the
    caller's trimmed spelling. The lookup and the add are one locked transaction, so a concurrent
    writer adding the same name never yields a second id. A deprecated-only name is never
    reused: a new active class takes it (re-activation is the explicit restore route)."""
    if not normalize_class_name(name):
        raise ClassNameInvalidError('class_name must contain a letter or digit')
    entry, _created = await registry.ensure_entry(name, group=group, notes=notes)
    return ResolvedClass(entry.class_id, entry.class_name)


async def class_id_for_name(name: str) -> int | None:
    """The id of the bound project's active class called ``name``, or ``None``."""
    doc = await get_class_registry().snapshot()
    found = resolve_class_by_name(doc.classes, name).active
    return None if found is None else found.class_id


async def class_name_for_id(class_id: int) -> str | None:
    """The registry's spelling for ``class_id`` (deprecated classes included), or ``None``."""
    entry = (await get_class_registry().snapshot()).get(class_id)
    return None if entry is None else entry.class_name


__all__ = [
    'ClassRegistryError',
    'ResolvedClass',
    'class_id_for_name',
    'class_name_for_id',
    'ensure_class_by_name',
]
