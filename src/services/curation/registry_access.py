"""The bound project's class registry for a service that wants only its snapshot."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.services.classes.registry import get_class_registry, registry_for


if TYPE_CHECKING:
    from src.config.projects import ProjectRecord
    from src.services.classes.registry import RegistryDoc


async def current_registry() -> RegistryDoc:
    """A snapshot of the bound project's registry."""
    return await get_class_registry().snapshot()


async def project_registry(record: ProjectRecord) -> RegistryDoc:
    """Another project's registry (a read; nothing is created)."""
    return await registry_for(record.slug).snapshot()


__all__ = ['current_registry', 'project_registry']
