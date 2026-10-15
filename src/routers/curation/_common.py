"""Shared curation router foundations.

Holds the `router` object, dependency adapters, Pydantic request/response
models and index-name constants. All
sub-modules import from here. _common.py MUST NOT import from sub-modules.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException

from src.clients.curation_opensearch.registry import ClassRegistry
from src.config import get_curation_config
from src.config.curation_accessors import (  # noqa: F401 - re-exported for the router modules
    classes_index,
    images_index,
    items_index,
    labels_confirmed_index,
)
from src.core.dependencies import get_curation_opensearch
from src.core.logging import get_logger
from src.core.wire_json import WireRoute
from src.routers.curation._api_error import api_error
from src.routers.curation._item_models import CropsPageResponse, ItemDoc  # noqa: F401 - re-export


def get_class_registry() -> ClassRegistry:
    """Indirection wrapper so tests patching
    ``src.routers.curation.get_class_registry`` reach every sub-module
    call site. The submodules import this wrapper rather than the
    upstream symbol so the mock applied in ``__init__``'s namespace
    propagates here at call-time.
    """
    from src.routers import curation as _pkg

    return _pkg.get_class_registry()


logger = get_logger(__name__)

config = get_curation_config()


# ``WireRoute``: routes declared ``response_model=None`` return hand-built wire
# dicts and are rendered directly (see src/core/wire_json.py).
router = APIRouter(
    tags=[config.api_tag],
    route_class=WireRoute,
)


# Index names are resolved per bound project at call time -- see
# src.config.curation_accessors.items_index() and friends (re-exported here).


# OpenSearch's index.max_result_window default. from+size past this 500s
# ("Result window is too large") instead of paging -- reject it explicitly
# with a 422 before it ever reaches OpenSearch.
MAX_RESULT_WINDOW = 10_000


def guard_page_depth(page: int, page_size: int) -> None:
    """Raise ``HTTPException(422)`` when ``(page-1)*page_size + page_size``
    would exceed :data:`MAX_RESULT_WINDOW` -- otherwise OpenSearch 500s past
    ``index.max_result_window`` and the app would surface that as a bare
    503/500 instead of a clear, cheap client-side rejection. Cursor-based
    pagination (``search_after``) is the documented way past this limit;
    This route doesn't add a cursor param, so depth is capped instead."""
    if (page - 1) * page_size + page_size > MAX_RESULT_WINDOW:
        raise HTTPException(
            status_code=422,
            detail=(
                f'page {page} at page_size {page_size} exceeds the {MAX_RESULT_WINDOW} '
                'result-window depth limit; use a smaller page_size or narrow the filter'
            ),
        )


def is_not_found(exc: BaseException) -> bool:
    """A single-doc read failed because the doc doesn't exist (vs. an outage)."""
    return (
        isinstance(exc, KeyError)
        or getattr(exc, 'status_code', None) == 404
        or 'NotFound' in type(exc).__name__
    )


def _registry_dep() -> ClassRegistry:
    return get_class_registry()


RegistryDep = Annotated[ClassRegistry, Depends(_registry_dep)]


# The one curation OpenSearch dependency (also used by the image, crop
# and UMAP routers): the raw ``AsyncOpenSearch`` with the project guard.
_raw_opensearch_dep = get_curation_opensearch


OpenSearchDep = Annotated[Any, Depends(_raw_opensearch_dep)]


def _require_region_profile_dep() -> Any:
    """No-profile gating contract: region data/write routes 409 with a
    clear detail when no region profile is configured, rather than
    quietly operating on a region concept that cannot exist yet."""
    from src.services.detection.profile_registry import get_active_region_profile

    profile = get_active_region_profile()
    if profile is None:
        raise api_error(409, 'no_active_profile', 'no region profile is configured')
    return profile


RegionProfileDep = Annotated[Any, Depends(_require_region_profile_dep)]
