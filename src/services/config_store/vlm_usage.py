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

from src.core.logging import get_logger
from src.services.config_store.index import get_activation
from src.services.config_store.project_usage import read_each_project


logger = get_logger(__name__)


async def activations_by_project(client: Any) -> dict[str, dict[str, Any] | None]:
    """``slug -> activation:vlm doc`` (``None`` = the project never activated
    one, so it runs the ``env`` built-in) for every project that has a
    configs index. Raises when a project's doc cannot be read: callers use
    this to REFUSE a delete, so an unreadable project must block it."""

    async def vlm_activation(index: str) -> dict[str, Any] | None:
        try:
            return await get_activation(client, index, 'vlm')
        except NotFoundError:
            return None

    return await read_each_project(vlm_activation)


def slugs_running(activations: dict[str, dict[str, Any] | None], name: str) -> list[str]:
    """Projects whose activation names ``name``. ``env`` is also what a
    project with no activation doc runs."""
    out: list[str] = []
    for slug, doc in activations.items():
        active = (doc or {}).get('name') if doc else ('env' if name == 'env' else None)
        if active == name:
            out.append(slug)
    return out


__all__ = ['activations_by_project', 'slugs_running']
