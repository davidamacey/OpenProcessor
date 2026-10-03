"""No-profile gating contract (owner directive, naming-sweep continuation).

With no active region profile, every region data/write route 409s with a
clear detail ("no region profile is configured") instead of silently
operating on a region concept that cannot exist yet. Reuses the
``app_client``/``fake_opensearch``/``fake_triton_pool`` fixtures from
``test_router_wireup.py`` via the shared ``conftest.py`` fixture
directory convention -- redefined locally here since pytest fixtures
aren't shared across modules without a conftest.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest


@pytest.fixture
def fake_opensearch() -> AsyncMock:
    fake = AsyncMock()
    fake.indices = AsyncMock()
    fake.indices.exists = AsyncMock(return_value=True)
    fake.indices.create = AsyncMock(return_value={'acknowledged': True})
    fake.indices.refresh = AsyncMock(return_value={'_shards': {}})
    fake.search = AsyncMock(return_value={'hits': {'hits': [], 'total': {'value': 0}}})
    fake.count = AsyncMock(return_value={'count': 0})
    fake.bulk = AsyncMock(return_value={'errors': False, 'items': []})
    fake.update = AsyncMock(return_value={'result': 'updated'})
    fake.update_by_query = AsyncMock(return_value={'updated': 0})
    fake.get = AsyncMock(return_value={'_source': {}})
    fake.index = AsyncMock(return_value={'result': 'created'})
    fake.msearch = AsyncMock(return_value={'responses': []})
    fake.mget = AsyncMock(return_value={'docs': []})
    return fake


@pytest.fixture
def app_client(fake_opensearch: AsyncMock) -> Any:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from src.core.dependencies import get_async_triton, get_opensearch
    from src.routers.curation import _raw_opensearch_dep, router as curation_router

    fake_triton_pool = AsyncMock()
    fake_triton_pool.health_check = AsyncMock(return_value=True)

    app = FastAPI()
    from _curation_app import mount_curation_routers

    mount_curation_routers(app, curation_router)
    app.dependency_overrides[get_opensearch] = lambda: fake_opensearch
    app.dependency_overrides[_raw_opensearch_dep] = lambda: fake_opensearch
    app.dependency_overrides[get_async_triton] = lambda: fake_triton_pool
    with TestClient(app) as client:
        yield client


# (method, path, json_body) for every region data/write route gated on an
# active region profile.
_GATED_ROUTES: tuple[tuple[str, str, dict[str, Any] | None], ...] = (
    ('get', '/curation/projects/default/regions', None),
    ('put', '/curation/projects/default/crops/crop1/regions', {'boxes': []}),
    (
        'patch',
        '/curation/projects/default/crops/crop1/region_meta',
        {'region_status': 'no_region_visible'},
    ),
    (
        'put',
        '/curation/projects/default/crops/batch_regions',
        {'crop_ids': ['crop1'], 'boxes': []},
    ),
    (
        'post',
        '/curation/projects/default/regions/batch_status',
        {'crop_ids': ['crop1'], 'region_status': 'detected'},
    ),
    ('post', '/curation/projects/default/crops/crop1/region/undo', None),
    ('post', '/curation/projects/default/crops/region/undo_batch', {'crop_ids': ['crop1']}),
    ('post', '/curation/projects/default/vlm/verify_regions', {'crop_ids': ['crop1']}),
    ('post', '/curation/projects/default/vlm/verify_region_batch', {'crops': []}),
    ('post', '/curation/projects/default/vlm/region_visible_batch', {'crops': []}),
    ('post', '/curation/projects/default/regions/clusters/refine/5', None),
    ('get', '/curation/projects/default/regions/clusters', None),
)


@pytest.mark.parametrize(('method', 'path', 'body'), _GATED_ROUTES)
def test_gated_route_409s_without_an_active_profile(
    app_client: Any, method: str, path: str, body: dict[str, Any] | None
) -> None:
    resp = (
        getattr(app_client, method)(path, json=body)
        if body is not None
        else getattr(app_client, method)(path)
    )
    assert resp.status_code == 409, resp.text
    detail = resp.json()['detail']
    assert detail['error'] == 'no_active_profile'
    assert detail['message'] == 'no region profile is configured'


@pytest.mark.parametrize(('method', 'path', 'body'), _GATED_ROUTES)
def test_gated_route_is_not_409_with_an_active_profile(
    app_client: Any,
    method: str,
    path: str,
    body: dict[str, Any] | None,
    reference_region_profile: None,
) -> None:
    """Same routes, active profile -- must get past the gate (whatever
    status the route itself returns for this fake/empty backend, as long
    as it isn't the gate's 409)."""
    resp = (
        getattr(app_client, method)(path, json=body)
        if body is not None
        else getattr(app_client, method)(path)
    )
    assert resp.status_code != 409 or 'no region profile is configured' not in resp.text


# Region-named routes that deliberately answer without an active profile.
# Everything else containing "region" in its path MUST carry the shared gate.
_UNGATED_REGION_ROUTES = frozenset(
    {
        # profile CRUD / activation: the way a profile becomes active at all
        '/region_profiles',
        # static vocabularies and read-only job snapshots
        '/regions/statuses',
        '/regions/vocabulary',
        '/regions/cluster/status',
        '/regions/fp_centroids/status',
        # documented to answer 200 with empty region_dependencies
        '/ingest/region_drain',
    }
)


def _has_gate(dependant: Any) -> bool:
    from src.routers.curation._common import _require_region_profile_dep

    return dependant.call is _require_region_profile_dep or any(
        _has_gate(d) for d in dependant.dependencies
    )


def test_every_region_route_carries_the_gate_or_is_documented_exempt() -> None:
    from fastapi.routing import APIRoute

    from src.routers.curation import router as curation_router

    missing: list[str] = []
    for route in curation_router.routes:
        if not isinstance(route, APIRoute) or 'region' not in route.path:
            continue
        path = route.path
        if path.startswith('/region_profiles') or path in _UNGATED_REGION_ROUTES:
            continue
        if not _has_gate(route.dependant):
            missing.append(f'{sorted(route.methods)} {path}')
    assert not missing, f'region routes without the no-profile gate: {missing}'
