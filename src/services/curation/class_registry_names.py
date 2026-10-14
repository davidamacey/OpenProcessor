"""The bound project's active (non-deprecated) class names.

An unreadable class registry raises: a validation gate must not pass as
"no classes", and a pack or profile check must not run against an empty set.
"""

from __future__ import annotations


def registry_class_names() -> frozenset[str]:
    from src.clients.curation_opensearch.registry import get_class_registry

    return frozenset(c.class_name for c in get_class_registry().load().classes if not c.deprecated)
