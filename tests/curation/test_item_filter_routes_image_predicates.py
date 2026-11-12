"""Every route that takes the shared item filter answers an image predicate (area, location,
near-duplicate group) with a result or a served 4xx, never a 5xx.

The image predicates are asked of the images index and joined into the items query by
``resolve_filter_join``; a route that builds the items query without that join raises
``ImageJoinMissingError``. The routes are enumerated from the OpenAPI document, so a route that
starts taking the filter fails ``test_every_filter_route_is_mapped`` until it is listed here.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from curation.query_fakes import QueryFakeOpenSearch
from src.config import IndexRole, index_name
from src.config.curation_accessors import base_curation_config
from src.core.db_session import get_db


CFG = base_curation_config()
ITEMS = CFG.items_index
IMAGES = index_name(CFG, IndexRole.IMAGES)
GROUP = '00000000-0000-0000-0000-000000000001'

# Each predicate as a query-string fragment and as an ItemFilter body.
PREDICATES: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {
    'has_location': ({'has_location': 'true'}, {'has_location': True}),
    'geo_bbox': (
        {'geo_bbox': '0,0,5,5'},
        {'geo_bbox': {'west': 0, 'south': 0, 'east': 5, 'north': 5}},
    ),
    'near_dup_group': ({'near_dup_group': GROUP}, {'near_dup_group': GROUP}),
}

Body = Callable[[dict[str, Any]], dict[str, Any]]


def _selection(f: dict[str, Any]) -> dict[str, Any]:
    return {'filter': f}


# Routes that take the filter in a query string: the path with its parameters filled in.
QUERY_ROUTES: dict[tuple[str, str], str] = {
    ('GET', '/crops'): '/crops',
    ('GET', '/crops/locate'): '/crops/locate?crop_id=a1',
    ('GET', '/clusters'): '/clusters',
    ('GET', '/regions'): '/regions',
    ('GET', '/detections/summary'): '/detections/summary',
    ('GET', '/search/text'): '/search/text?text=car',
    ('GET', '/search/similar'): '/search/similar?crop_id=a1',
    ('GET', '/images'): '/images',
    ('GET', '/images/locate'): '/images/locate?image_id=img-a',
    ('GET', '/map/cells'): '/map/cells?zoom=4',
    ('POST', '/pipeline/auto_label/start'): '/pipeline/auto_label/start',
    ('GET', '/review/tabs'): '/review/tabs',
    ('GET', '/review/{tab}'): '/review/all',
    ('GET', '/review/{tab}/locate'): '/review/all/locate?crop_id=a1',
    ('POST', '/search/image'): '/search/image',
    ('GET', '/stats/classes'): '/stats/classes',
    ('GET', '/stats/dataset'): '/stats/dataset',
    ('GET', '/table'): '/table',
    ('GET', '/viz/projection'): '/viz/projection',
}

# Routes that take it in a JSON body: the body for a filter ``f``.
BODY_ROUTES: dict[tuple[str, str], Body] = {
    ('PUT', '/crops/batch_label'): lambda f: {'selection': _selection(f), 'class_id': 1},
    ('POST', '/crops/move'): lambda f: {
        'selection': _selection(f),
        'cluster_uid': '00000000-0000-0000-0000-000000000002',
    },
    ('POST', '/crops/batch_exclude'): lambda f: {'selection': _selection(f), 'reason': 'ignore'},
    ('POST', '/crops/batch_unexclude'): lambda f: {'selection': _selection(f)},
    ('POST', '/export/yolo'): lambda f: {'item_filter': f, 'version_tag': 'v1'},
    ('POST', '/assist/runs'): lambda f: {
        'target': {'kind': 'filter', 'filter': f},
        'actions': ['classify'],
    },
    ('POST', '/reprocess'): lambda f: {'targets': {'filter': f}, 'scopes': ['embed']},
    ('POST', '/select/diverse'): lambda f: {'scope': {'item_filter': f}, 'k': 3},
    ('POST', '/selection/summary'): _selection,
}


def _app_spec() -> dict[str, Any]:
    from _curation_app import mount_curation_routers

    from src.routers.curation import router as curation_router

    app = FastAPI()
    mount_curation_routers(app, curation_router)
    return app.openapi()


def _refs(node: Any, out: set[str]) -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            if key == '$ref':
                out.add(value.rsplit('/', 1)[-1])
            else:
                _refs(value, out)
    elif isinstance(node, list):
        for value in node:
            _refs(value, out)


def _takes_filter_in_body(op: dict[str, Any], schemas: dict[str, Any]) -> bool:
    seen: set[str] = set()
    todo: set[str] = set()
    _refs(op.get('requestBody', {}), todo)
    while todo:
        name = todo.pop()
        if name in seen:
            continue
        seen.add(name)
        _refs(schemas[name], todo)
    return any('geo_bbox' in schemas[n].get('properties', {}) for n in seen)


def _filter_routes() -> tuple[set[tuple[str, str]], set[tuple[str, str]]]:
    spec = _app_spec()
    prefix = '/curation/projects/{project}'
    query: set[tuple[str, str]] = set()
    body: set[tuple[str, str]] = set()
    for path, ops in spec['paths'].items():
        for method, op in ops.items():
            key = (method.upper(), path.removeprefix(prefix))
            if any(p['name'] == 'has_location' for p in op.get('parameters', [])):
                query.add(key)
            if _takes_filter_in_body(op, spec['components']['schemas']):
                body.add(key)
    return query, body


def test_every_filter_route_is_mapped() -> None:
    query, body = _filter_routes()
    assert query == set(QUERY_ROUTES)
    assert body == set(BODY_ROUTES)


def _world() -> QueryFakeOpenSearch:
    def item(crop_id: str, image: str) -> dict[str, Any]:
        return {
            'crop_id': crop_id,
            'image_id': image,
            'bbox_norm': [0, 0, 1, 1],
            'class_id': 1,
            'class_name': 'car',
        }

    return QueryFakeOpenSearch(
        {
            ITEMS: {'a1': item('a1', 'img-a'), 'b1': item('b1', 'img-b')},
            IMAGES: {
                'img-a': {
                    'image_id': 'img-a',
                    'capture.location': {'lat': 1, 'lon': 2},
                    'near_dup_group_id': GROUP,
                },
                'img-b': {'image_id': 'img-b'},
            },
        }
    )


# These handlers go on to read the database after the filter is resolved (display numbers,
# the dataset snapshot, the export directory, the cluster move); the offline tier has none, so they may answer
# 5xx here. The missing-join guard below still covers them.
NEEDS_DATABASE = frozenset(
    {
        ('GET', '/clusters'),
        ('GET', '/images'),
        ('GET', '/stats/dataset'),
        ('POST', '/crops/move'),
        ('POST', '/export/yolo'),
    }
)
SCOPED = '/curation/projects/default'


@pytest.fixture
def missing_joins(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Every ImageJoinMissingError built during the request, even one a route turns into a 503."""
    from src.services.curation import item_filter

    seen: list[str] = []

    class Recording(item_filter.ImageJoinMissingError):
        def __init__(self, *args: Any) -> None:
            seen.append(str(args))
            super().__init__(*args)

    monkeypatch.setattr(item_filter, 'ImageJoinMissingError', Recording)
    return seen


@pytest.fixture(scope='module')
def _app() -> FastAPI:
    from _curation_app import mount_curation_routers

    from src.routers.curation import router as curation_router

    return mount_curation_routers(FastAPI(), curation_router)


@pytest.fixture
def client(_app: FastAPI, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    from _msearch_fake import install_msearch

    from src.routers.curation import _raw_opensearch_dep

    async def _policy() -> str:
        return 'store'

    for target in (
        'src.services.curation.image_join.get_location_policy',
        'src.routers.curation.selection_summary.get_location_policy',
    ):
        monkeypatch.setattr(target, _policy)
    monkeypatch.setenv('OP_SELECT_DIVERSE_ENABLED', '1')
    fake = install_msearch(_world())
    _app.dependency_overrides[_raw_opensearch_dep] = lambda: fake
    _app.dependency_overrides[get_db] = lambda: MagicMock()
    yield TestClient(_app, raise_server_exceptions=False)
    _app.dependency_overrides.clear()


def _check(response: Any, key: tuple[str, str], missing_joins: list[str]) -> None:
    assert not missing_joins, 'the query was built without the image join'
    if key not in NEEDS_DATABASE:
        assert response.status_code < 500, response.text


@pytest.mark.parametrize('predicate', sorted(PREDICATES))
@pytest.mark.parametrize(('method', 'route'), sorted(QUERY_ROUTES))
def test_query_routes_answer_an_image_predicate(
    client: TestClient,
    missing_joins: list[str],
    method: str,
    route: str,
    predicate: str,
) -> None:
    query, _ = PREDICATES[predicate]
    url = SCOPED + QUERY_ROUTES[(method, route)]
    response = client.request(method, url, params=query)
    _check(response, (method, route), missing_joins)


@pytest.mark.parametrize('predicate', sorted(PREDICATES))
@pytest.mark.parametrize(('method', 'route'), sorted(BODY_ROUTES))
def test_body_routes_answer_an_image_predicate(
    client: TestClient,
    missing_joins: list[str],
    method: str,
    route: str,
    predicate: str,
) -> None:
    _, body = PREDICATES[predicate]
    response = client.request(method, SCOPED + route, json=BODY_ROUTES[(method, route)](body))
    _check(response, (method, route), missing_joins)
