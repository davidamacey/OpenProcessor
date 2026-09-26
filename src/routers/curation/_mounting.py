"""Mount the curation routers twice: scoped under
``{api_prefix}/projects/{project}`` (the documented form) and unscoped at
``{api_prefix}`` as a hidden alias bound to the ``default`` project (see
``docs/design/openprocessor_internal/projects_plan.md`` §3.2).

The routers themselves keep their absolute ``{api_prefix}/...`` prefixes,
so a test (or tool) that includes one router directly keeps its paths.
:func:`relative_router` strips that prefix from a copy of the route table
so the same endpoints can be re-rooted under either mount point.
"""

from __future__ import annotations

import copy
import os
from typing import TYPE_CHECKING

from fastapi import APIRouter, Depends

from src.routers.curation._project_deps import (
    bind_default_project,
    bind_path_project,
    install_project_exception_handlers,
)


if TYPE_CHECKING:
    from fastapi import FastAPI


def scoped_prefix(api_prefix: str) -> str:
    """``{api_prefix}/projects/{project}``: the path template every scoped
    curation route starts with."""
    return f'{api_prefix}/projects/{{project}}'


def relative_router(router: APIRouter, api_prefix: str) -> APIRouter:
    """A copy of ``router`` whose routes have ``api_prefix`` stripped from
    the front of their path. Endpoints, dependencies, response models and
    tags are shared with the original; only the path changes."""
    relative = APIRouter(default_response_class=router.default_response_class)
    for route in router.routes:
        path = getattr(route, 'path', '')
        if not path.startswith(api_prefix):
            raise ValueError(f'route {path!r} is not under the curation prefix {api_prefix!r}')
        clone = copy.copy(route)
        clone.path = path[len(api_prefix) :]  # type: ignore[attr-defined]
        relative.routes.append(clone)
    return relative


def unscoped_alias_enabled() -> bool:
    """``OP_UNSCOPED_ALIAS=default`` (the default) mounts the unscoped
    alias; ``off`` makes unscoped curation paths 404. Deployment config
    (does the alias mount exist), not a second code path."""
    return os.environ.get('OP_UNSCOPED_ALIAS', 'default').strip().lower() != 'off'


def mount_curation(
    app: FastAPI,
    *,
    api_prefix: str,
    global_router: APIRouter,
    scoped_routers: list[APIRouter],
) -> None:
    """Register, in this order (first match wins):

    1. ``global_router`` at ``api_prefix`` -- routes that are global by
       nature (``/projects``, the deployment ``/health`` and ``/events``),
       so the alias below can never shadow them;
    2. every scoped router under ``{api_prefix}/projects/{project}`` with
       :func:`bind_path_project`;
    3. the same routers at ``api_prefix`` with :func:`bind_default_project`,
       hidden from OpenAPI (the unscoped alias).
    """
    app.include_router(global_router, prefix=api_prefix)
    relative = [relative_router(r, api_prefix) for r in scoped_routers]
    for r in relative:
        app.include_router(
            r,
            prefix=scoped_prefix(api_prefix),
            dependencies=[Depends(bind_path_project)],
        )
    if unscoped_alias_enabled():
        for r in relative:
            app.include_router(
                r,
                prefix=api_prefix,
                dependencies=[Depends(bind_default_project)],
                include_in_schema=False,
            )


def mount_all_curation_routers(app: FastAPI) -> None:
    """Mount the global router and every curation router on ``app``
    (see :func:`mount_curation` for the order)."""
    from src.routers.curation import router as curation_router
    from src.routers.curation.projects import global_router
    from src.routers.curation_images import crops_router, router as images_router
    from src.routers.curation_train import router as train_router
    from src.routers.curation_umap import router as umap_router

    mount_curation(
        app,
        api_prefix=curation_router.prefix,
        global_router=global_router,
        scoped_routers=[curation_router, images_router, crops_router, umap_router, train_router],
    )
    install_project_exception_handlers(app)
