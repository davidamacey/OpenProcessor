"""Seed the active region profile's class into the bound project's class
registry, so region export, the region gallery and class lists resolve it by
name. Idempotent; called for every project at API startup and when a
project is created."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.clients.curation_opensearch import get_class_registry
from src.services.detection.profile_registry import get_active_region_profile


if TYPE_CHECKING:
    from collections.abc import Iterable

    from src.clients.curation_opensearch import RegistryClassEntry


REGION_CLASS_GROUP = 'region'


def ensure_region_class() -> int | None:
    """The region class's id, adding it first if the registry lacks it.
    ``None`` when no region profile names a region class."""
    profile = get_active_region_profile()
    name = (profile.region_class_name if profile else '').strip()
    if not name:
        return None
    registry = get_class_registry()
    for entry in registry.load().classes:
        if not entry.deprecated and entry.class_name.lower() == name.lower():
            return entry.class_id
    return registry.add_class(
        name, group=REGION_CLASS_GROUP, notes='seeded from the active region profile'
    )


def item_classes(classes: Iterable[RegistryClassEntry]) -> list[RegistryClassEntry]:
    """The classes an item labeler may assign to a whole item: every active
    class except the region class, which only ever labels a sub-box."""
    profile = get_active_region_profile()
    region = (profile.region_class_name if profile else '').strip().lower()
    # ``group == 'region'`` marks a class seeded by ensure_region_class: it
    # stays a sub-box class after the profile is deactivated.
    return [
        c
        for c in classes
        if not c.deprecated and c.group != REGION_CLASS_GROUP and c.class_name.lower() != region
    ]
