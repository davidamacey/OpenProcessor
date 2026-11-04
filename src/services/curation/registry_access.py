"""The bound project's class registry for a service that has no OpenSearch client in hand."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.clients.curation_opensearch.registry import get_class_registry
from src.config.project_context import bind_project
from src.services.projects.guard import make_curation_opensearch


if TYPE_CHECKING:
    from src.clients.curation_opensearch.registry import RegistryDoc
    from src.config.projects import ProjectRecord


async def current_registry() -> RegistryDoc:
    """A snapshot of the bound project's registry document (409 ``registry_not_migrated``
    while its legacy file has no document)."""
    return await get_class_registry().snapshot(await make_curation_opensearch())


async def project_registry(client: Any, record: ProjectRecord) -> RegistryDoc:
    """Another project's registry, read under a read-only binding of that project (the guard
    refuses an index the bound project does not own). Never creates its document."""
    with bind_project(record, read_only=True):
        return await get_class_registry().snapshot(client, create=False)


__all__ = ['current_registry', 'project_registry']
