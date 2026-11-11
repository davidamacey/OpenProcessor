"""Shared curation router foundations.

Holds the `router` object, dependency adapters, Pydantic request/response
models and index-name constants. All
sub-modules import from here. _common.py MUST NOT import from sub-modules.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends

from src.config import get_curation_config
from src.core.dependencies import get_curation_opensearch
from src.core.logging import get_logger
from src.core.wire_json import WireRoute
from src.routers.curation._api_error import api_error
from src.services.classes.registry import ClassRegistry


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


def is_not_found(exc: BaseException) -> bool:
    """A single-doc read failed because the doc doesn't exist (vs. an outage)."""
    return (
        isinstance(exc, KeyError)
        or getattr(exc, 'status_code', None) == 404
        or 'NotFound' in type(exc).__name__
    )


async def _registry_dep() -> ClassRegistry:
    """The bound project's class registry."""
    return get_class_registry()


RegistryDep = Annotated[ClassRegistry, Depends(_registry_dep)]


# The one curation OpenSearch dependency (also used by the image, crop
# and UMAP routers): the raw ``AsyncOpenSearch`` with the project guard.
_raw_opensearch_dep = get_curation_opensearch


OpenSearchDep = Annotated[Any, Depends(_raw_opensearch_dep)]


async def _require_region_profile_dep() -> Any:
    """No-profile gating contract: region data/write routes 409 with a
    clear detail when no region profile is configured, rather than
    quietly operating on a region concept that cannot exist yet.

    "None" from this worker's snapshot is not final: an activation made through another API
    worker is only here after the next poll tick, so the store's epoch is checked first
    (``confirmed_active_region_profile``). 409 only when the store
    agrees; an unreadable store is 503, never a guess."""
    from src.services.config_store.store import ConfigStoreUnavailableError
    from src.services.detection.profile_registry import confirmed_active_region_profile

    try:
        profile = await confirmed_active_region_profile()
    except ConfigStoreUnavailableError as exc:
        raise api_error(503, 'config_store_unavailable', str(exc)) from exc
    if profile is None:
        raise api_error(409, 'no_active_profile', 'no region profile is configured')
    return profile


RegionProfileDep = Annotated[Any, Depends(_require_region_profile_dep)]
