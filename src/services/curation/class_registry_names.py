"""The bound project's active (non-deprecated) class names, and one item class by name.

An unreadable class registry raises: a validation gate must not pass as
"no classes", and a pack or profile check must not run against an empty set.
"""

from __future__ import annotations


def registry_class_names() -> frozenset[str]:
    from src.clients.curation_opensearch.registry import get_class_registry

    return frozenset(c.class_name for c in get_class_registry().load().classes if not c.deprecated)


def registry_item_class(class_name: str) -> tuple[int, str] | None:
    """``(class_id, class_name)`` of the item class named ``class_name`` (compared
    normalized), or ``None`` when the registry has no such item class."""
    from src.clients.curation_opensearch.registry import get_class_registry
    from src.services.curation.region_class import item_classes
    from src.utils.class_names import normalize_class_name

    wanted = normalize_class_name(class_name)
    for entry in item_classes(get_class_registry().load().classes):
        if normalize_class_name(entry.class_name) == wanted:
            return entry.class_id, entry.class_name
    return None
