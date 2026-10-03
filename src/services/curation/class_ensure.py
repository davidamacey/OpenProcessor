"""Resolve a class NAME to its registry id, adding the class when it is new.

Class identity is by name: registry ids are per project and never assumed
equal to a model's or another project's ids. The one name-to-id path for the
writers that create classes as a side effect (the region class seed and the
open-vocabulary pass).
"""

from __future__ import annotations

from typing import Any, Protocol

from src.clients.curation_opensearch import ClassRegistryError


class NamedClassRegistry(Protocol):
    """What :func:`ensure_class_by_name` needs of a registry (the real
    :class:`~src.clients.curation_opensearch.ClassRegistry` provides it)."""

    def load(self) -> Any: ...

    def add_class(self, name: str, group: str = ..., notes: str = ...) -> int: ...


def ensure_class_by_name(
    registry: NamedClassRegistry, name: str, *, group: str, notes: str = ''
) -> int:
    """The id of the non-deprecated class named ``name`` (trimmed,
    case-insensitive), adding it under ``group`` first when the registry
    lacks it. A concurrent writer adding the same name first is not an error."""
    wanted = name.strip().casefold()

    def find() -> int | None:
        for entry in registry.load().classes:
            if not entry.deprecated and entry.class_name.casefold() == wanted:
                return entry.class_id
        return None

    found = find()
    if found is not None:
        return found
    try:
        return registry.add_class(name.strip(), group=group, notes=notes)
    except ClassRegistryError:
        found = find()
        if found is None:
            raise
        return found


__all__ = ['ensure_class_by_name']
