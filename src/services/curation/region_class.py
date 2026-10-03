"""Seed the active region profile's class into the bound project's class
registry, so region export, the region gallery and class lists resolve it by
name. Idempotent; called for every project at API startup and when a
project is created."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.clients.curation_opensearch import get_class_registry
from src.services.curation.class_ensure import ensure_class_by_name
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
    return ensure_class_by_name(
        get_class_registry(),
        name,
        group=REGION_CLASS_GROUP,
        notes='seeded from the active region profile',
    ).class_id


def is_region_class(entry: RegistryClassEntry) -> bool:
    """Whether ``entry`` is a sub-box (region) class: seeded under the
    ``region`` group, or named by the active profile. The group is stored, so
    the answer does not flip with a process whose config snapshot has not
    loaded the active profile yet."""
    if entry.group == REGION_CLASS_GROUP:
        return True
    profile = get_active_region_profile()
    name = (profile.region_class_name if profile else '').strip().lower()
    return bool(name) and entry.class_name.strip().lower() == name


def item_classes(classes: Iterable[RegistryClassEntry]) -> list[RegistryClassEntry]:
    """The classes an item labeler may assign to a whole item: every active
    class except the region class, which only ever labels a sub-box."""
    return [c for c in classes if not c.deprecated and not is_region_class(c)]
