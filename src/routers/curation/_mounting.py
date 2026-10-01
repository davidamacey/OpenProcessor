"""Mount the curation routers (see
``docs/design/openprocessor_internal/projects_plan.md`` §3.2 and its
"fresh build, no backwards compatibility" owner decision).

Every curation route lives under ``{api_prefix}/projects/{project}`` and
binds that project through :func:`bind_path_project`. The only routes
outside a project are the global ones on ``global_router``:
``{api_prefix}/projects``, ``{api_prefix}/health`` and
``{api_prefix}/events``. There is no unscoped alias: an unscoped curation
path is a 404.

The routers carry prefixes relative to the project root (``/crops``,
``/train``, ...), so they are mounted exactly once.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import Depends

from src.routers.curation._project_deps import bind_path_project, install_project_exception_handlers


if TYPE_CHECKING:
    from fastapi import APIRouter, FastAPI


def scoped_prefix(api_prefix: str) -> str:
    """``{api_prefix}/projects/{project}``: the path template every scoped
    curation route starts with."""
    return f'{api_prefix}/projects/{{project}}'


def mount_curation(
    app: FastAPI,
    *,
    api_prefix: str,
    global_router: APIRouter,
    scoped_routers: list[APIRouter],
) -> None:
    """Register ``global_router`` at ``api_prefix``, then every scoped
    router under ``{api_prefix}/projects/{project}`` with
    :func:`bind_path_project`."""
    app.include_router(global_router, prefix=api_prefix)
    for router in scoped_routers:
        app.include_router(
            router,
            prefix=scoped_prefix(api_prefix),
            dependencies=[Depends(bind_path_project)],
        )


def curation_scoped_routers() -> list[APIRouter]:
    """Every project-scoped curation router, in mount order."""
    from src.routers.curation import router as curation_router
    from src.routers.curation_images import crops_router, router as images_router
    from src.routers.curation_train import router as train_router
    from src.routers.curation_umap import router as umap_router

    return [curation_router, images_router, crops_router, umap_router, train_router]


def mount_all_curation_routers(app: FastAPI) -> None:
    """Mount the global router and every curation router on ``app``
    (see :func:`mount_curation`)."""
    import src.routers.curation.global_status
    import src.routers.curation.projects_combine  # noqa: F401 - registers /projects/combine* on global_router
    from src.config.curation import base_curation_config
    from src.routers.curation.projects import global_router

    mount_curation(
        app,
        api_prefix=base_curation_config().api_prefix,
        global_router=global_router,
        scoped_routers=curation_scoped_routers(),
    )
    install_project_exception_handlers(app)
