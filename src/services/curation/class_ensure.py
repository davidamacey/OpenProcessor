"""Resolve a class NAME to its registry id, adding the class when it is new.

Class identity is by name: registry ids are per project and never assumed
equal to a model's or another project's ids. The one name-to-id path for the
writers that create classes as a side effect (the region class seed and the
open-vocabulary pass).
"""

from __future__ import annotations

from typing import Any, NamedTuple, Protocol

from src.clients.curation_opensearch import ClassRegistryError
from src.utils.class_names import normalize_class_name, resolve_class_by_name


class NamedClassRegistry(Protocol):
    """What :func:`ensure_class_by_name` needs of a registry (the real
    :class:`~src.clients.curation_opensearch.ClassRegistry` provides it)."""

    def load(self) -> Any: ...

    def add_class(self, name: str, group: str = ..., notes: str = ...) -> int: ...


class ResolvedClass(NamedTuple):
    """A registry class: its id and the name the registry spells it with (an
    item stores this name, not the spelling the caller asked for)."""

    class_id: int
    class_name: str


def ensure_class_by_name(
    registry: NamedClassRegistry, name: str, *, group: str, notes: str = ''
) -> ResolvedClass:
    """The active class named ``name`` (see :func:`~src.utils.class_names.resolve_class_by_name`), added under ``group`` first
    when the registry lacks it. Names compare by
    :func:`~src.utils.class_names.normalize_class_name` (the one name-equality
    rule); a new class keeps the caller's (trimmed) spelling. A concurrent
    writer adding the same name first is not an error."""
    wanted = normalize_class_name(name)
    if not wanted:
        raise ClassRegistryError('class_name must contain a letter or digit')

    def find() -> ResolvedClass | None:
        match = resolve_class_by_name(registry.load().classes, name)
        if match.active is None:
            # A deprecated-only name is never reused: a new active class
            # takes it (re-activation is the explicit restore route).
            return None
        return ResolvedClass(match.active.class_id, match.active.class_name)

    found = find()
    if found is not None:
        return found
    try:
        return ResolvedClass(
            registry.add_class(name.strip(), group=group, notes=notes), name.strip()
        )
    except ClassRegistryError:
        found = find()
        if found is None:
            raise
        return found


__all__ = ['NamedClassRegistry', 'ResolvedClass', 'ensure_class_by_name']
