"""Shared per-project discovery/dispatch helpers for the round-robin
curation workers (``vlm_worker.py``, ``cluster_refresh_daemon.py``) that
run in multi-project mode (no ``--project``) instead of binding one
project for the whole process.

Split out of those two scripts so each stays under the repo's 700 LOC
ratchet. Resolution here is always fresh (never cached at import time
or across cycles) -- ``PROJECT_SCOPED_FIELDS`` must be read while the
target project's ``bind_project`` block is active.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from src.config.projects import ProjectRecord


def project_items_index(record: ProjectRecord | Any) -> str:
    """``record``'s own items index name."""
    from src.config.curation import IndexRole, get_curation_config, index_name
    from src.config.project_context import bind_project

    with bind_project(record):
        return index_name(get_curation_config(), IndexRole.ITEMS)


def project_api_prefix() -> str:
    """The deployment's curation API prefix (a global field -- no bind needed)."""
    from src.config.curation import get_curation_config

    return get_curation_config().api_prefix.rstrip('/')


def project_paused(record: ProjectRecord | Any) -> bool:
    """Minimal per-project pause primitive: presence of
    ``<project_state_dir>/pipeline_paused.flag``. No ``pipeline.paused``
    flag or route exists elsewhere in this codebase to reuse -- this is
    the simplest thing that lets a round-robin worker skip one paused
    project without touching the others."""
    from src.config.curation import get_curation_config
    from src.config.project_context import bind_project

    with bind_project(record):
        return (get_curation_config().project_state_dir / 'pipeline_paused.flag').exists()


def scoped_url(api: str, api_prefix: str, slug: str, path: str) -> str:
    """``{api}{api_prefix}/projects/{slug}{path}`` -- mirrors the scoped
    mount formula (``src.routers.curation._mounting.scoped_prefix``)
    since a script has no request context for ``project_api_base()``."""
    return f'{api}{api_prefix}/projects/{slug}{path}'
