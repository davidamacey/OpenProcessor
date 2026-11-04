"""Resolve a class NAME to its registry id, adding the class when it is new.

Class identity is by name: registry ids are per project and never assumed
equal to a model's or another project's ids. The one name-to-id path for the
writers that create classes as a side effect (the region class seed and the
open-vocabulary pass).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

from src.clients.curation_opensearch.registry import ClassRegistryError
from src.utils.class_names import normalize_class_name


if TYPE_CHECKING:
    from opensearchpy import AsyncOpenSearch

    from src.clients.curation_opensearch.registry import ClassRegistry


class ResolvedClass(NamedTuple):
    """A registry class: its id and the name the registry spells it with (an
    item stores this name, not the spelling the caller asked for)."""

    class_id: int
    class_name: str


async def ensure_class_by_name(
    registry: ClassRegistry,
    client: AsyncOpenSearch,
    name: str,
    *,
    group: str,
    notes: str = '',
) -> ResolvedClass:
    """The active class named ``name`` (see :func:`~src.utils.class_names.resolve_class_by_name`),
    added under ``group`` first when the registry lacks it. Names compare by
    :func:`~src.utils.class_names.normalize_class_name` (the one name-equality
    rule); a new class keeps the caller's (trimmed) spelling. The lookup and the add are one
    optimistic-concurrency mutation, so a concurrent writer adding the same name is not an
    error and never yields a second id. A deprecated-only name is never reused: a new active
    class takes it (re-activation is the explicit restore route)."""
    if not normalize_class_name(name):
        raise ClassRegistryError('class_name must contain a letter or digit')
    entry, _created = await registry.ensure_entry(client, name, group=group, notes=notes)
    return ResolvedClass(entry.class_id, entry.class_name)


__all__ = ['ResolvedClass', 'ensure_class_by_name']
