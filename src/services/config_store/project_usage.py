"""Read something from every project's own config index.

Activation is per project while promoted models and VLM endpoints are
deployment-wide, so "who uses X" is answered from each project's LIVE
config docs (a read-only bind per project, straight from OpenSearch --
never a cached snapshot: a delete or an unshare must not race a stale view
of another project's activation).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.config import get_curation_config
from src.config.project_context import bind_project
from src.services.projects.registry import get_project_registry


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable


async def read_each_project[T](read: Callable[[str], Awaitable[T]]) -> dict[str, T]:
    """``slug -> read(configs_index)`` for every project that exists and is
    not being deleted (``ProjectRegistry.existing_projects``), each call inside a
    read-only bind of that project. A read that raises propagates: callers
    use this to REFUSE a change, so an unreadable project must block it."""
    registry = get_project_registry()
    await registry.ensure_fresh()
    found: dict[str, T] = {}
    for record in sorted(registry.existing_projects(), key=lambda r: r.slug):
        with bind_project(record, read_only=True):
            found[record.slug] = await read(get_curation_config().configs_index)
    return found


async def active_detector_users(client: Any, model_name: str) -> list[tuple[str, str]]:
    """``(project slug, active profile name)`` for every project whose ACTIVE
    detection profile (the revision it pinned) names ``model_name`` as its
    ``detector_model``."""
    from opensearchpy.exceptions import NotFoundError

    from src.services.config_store.index import config_doc_id, get_activation

    async def active_detector(index: str) -> tuple[str, str] | None:
        activation = await get_activation(client, index, 'detection_profile')
        name = (activation or {}).get('name')
        revision = (activation or {}).get('revision')
        if not name or revision is None:
            return None
        try:
            doc = await client.get(
                index=index, id=config_doc_id('region_profile', name, int(revision))
            )
        except NotFoundError:
            return None
        return name, str((doc['_source'].get('body') or {}).get('detector_model') or '')

    return [
        (slug, found[0])
        for slug, found in (await read_each_project(active_detector)).items()
        if found is not None and found[1] == model_name
    ]


__all__ = ['active_detector_users', 'read_each_project']
