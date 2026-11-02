"""Mount curation routers on a bare test app exactly as production does:
under ``/curation/projects/{project}``, binding the project per request.
Test URLs are therefore the production ones (``SCOPED`` + route path)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import APIRouter, FastAPI


if TYPE_CHECKING:
    from collections.abc import Iterable


API = '/curation'
# The project every unit test binds (tests/conftest.py).
SCOPED = f'{API}/projects/default'


def mount_curation_routers(app: FastAPI, *routers: APIRouter) -> FastAPI:
    """Mount ``routers`` (default: every scoped curation router) plus the
    project exception handlers on ``app``; return ``app``."""
    from src.routers.curation._mounting import curation_scoped_routers, mount_curation
    from src.routers.curation._project_deps import install_project_exception_handlers

    chosen: Iterable[APIRouter] = routers or curation_scoped_routers()
    mount_curation(app, api_prefix=API, global_router=APIRouter(), scoped_routers=list(chosen))
    install_project_exception_handlers(app)
    return app


def curation_test_app(*routers: APIRouter) -> FastAPI:
    """A new app with ``routers`` mounted (see :func:`mount_curation_routers`)."""
    return mount_curation_routers(FastAPI(), *routers)


def mark_config_store_loaded() -> None:
    """The bound project's config store as a worker that loaded it and found
    nothing active (``loaded_at`` is in the future so ``ensure_fresh`` never
    reads the test's fake). For tests that ``reset_config_stores()`` themselves."""
    import dataclasses
    import time

    from src.services.config_store import get_config_store

    store = get_config_store()
    store.current = dataclasses.replace(store.current, loaded_at=time.monotonic() + 1e9)
