"""Which projects run which VLM endpoint (W9.6).

The registry is deployment-wide but activation is per project, so "is this
endpoint in use" is answered from every project's LIVE ``activation:vlm``
doc (a read-only bind per project, straight from OpenSearch -- never a
cached snapshot: a delete must not race a stale view of another project's
activation).
"""

from __future__ import annotations

from typing import Any

from opensearchpy.exceptions import NotFoundError

from src.config import get_curation_config
from src.config.project_context import bind_project
from src.core.logging import get_logger
from src.services.config_store.index import get_activation
from src.services.projects.registry import get_project_registry


logger = get_logger(__name__)


async def activations_by_project(client: Any) -> dict[str, dict[str, Any] | None]:
    """``slug -> activation:vlm doc`` (``None`` = the project never activated
    one, so it runs the ``env`` built-in) for every project that has a
    configs index. Raises when a project's doc cannot be read: callers use
    this to REFUSE a delete, so an unreadable project must block it."""
    registry = get_project_registry()
    await registry.ensure_fresh()
    found: dict[str, dict[str, Any] | None] = {}
    for slug, record in sorted(registry.snapshot().items()):
        with bind_project(record, read_only=True):
            index = get_curation_config().configs_index
            try:
                found[slug] = await get_activation(client, index, 'vlm')
            except NotFoundError:
                found[slug] = None
    return found


def slugs_running(activations: dict[str, dict[str, Any] | None], name: str) -> list[str]:
    """Projects whose activation names ``name``. ``env`` is also what a
    project with no activation doc runs."""
    out: list[str] = []
    for slug, doc in activations.items():
        active = (doc or {}).get('name') if doc else ('env' if name == 'env' else None)
        if active == name:
            out.append(slug)
    return out


async def projects_using(client: Any, name: str) -> list[str]:
    return slugs_running(await activations_by_project(client), name)


__all__ = ['activations_by_project', 'projects_using', 'slugs_running']
