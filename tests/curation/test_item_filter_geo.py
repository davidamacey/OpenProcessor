"""The area, location and near-duplicate-group predicates of the shared item filter."""

from __future__ import annotations

import json
import uuid
from typing import Any

import pytest
from _embedding_registry_fake import DEFAULT_WRITE_FIELDS
from fastapi import FastAPI
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from curation.query_fakes import QueryFakeOpenSearch
from src.config import IndexRole, index_name
from src.config.curation_accessors import base_curation_config
from src.domain.geo_box import GeoBox
from src.domain.location_policy import snap_box, snap_polygon
from src.routers.curation import (
    _item_filter_params as params_module,
    _raw_opensearch_dep,
    images_view,
    map as map_route,
    router as curation_router,
)
from src.services.curation import image_join
from src.services.curation.item_filter import ItemFilter, item_filter_clauses


def test_coarse_policy_rounds_a_box_outward() -> None:
    box = GeoBox(west=8.1234, south=47.1234, east=8.5678, north=47.5678)
    assert snap_box(box, 'store_coarse') == GeoBox(west=8.12, south=47.12, east=8.57, north=47.57)
    assert snap_box(box, 'store') == box


def test_drop_policy_matches_nothing() -> None:
    box = GeoBox(west=8.0, south=47.0, east=9.0, north=48.0)
    assert snap_box(box, 'drop') is None
    assert snap_polygon([(8.0, 47.0), (9.0, 47.0), (8.5, 48.0)], 'drop') is None


def test_coarse_policy_rounds_polygon_vertices() -> None:
    got = snap_polygon([(8.1234, 47.1234), (9.5678, 47.0), (8.5, 48.0)], 'store_coarse')
    assert got == [(8.12, 47.12), (9.57, 47.0), (8.5, 48.0)]


def test_unknown_policy_fails_closed() -> None:
    with pytest.raises(ValueError, match='location_policy'):
        snap_box(GeoBox(west=0, south=0, east=1, north=1), 'bogus')


# --- the filter, the join and the routes -------------------------------------------------------

CFG = base_curation_config()
ITEMS = CFG.items_index
IMAGES = index_name(CFG, IndexRole.IMAGES)
BASE = '/curation/projects/default'
GROUP = uuid.UUID('11111111-1111-1111-1111-111111111111')
CLUSTER_A = uuid.UUID('aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa')
CLUSTER_B = uuid.UUID('bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb')
# img-a and img-b are inside the box; img-c is far away; img-d has no position.
BOX = '7,46,9,48'
TRIANGLE = '7.0,46.0;9.0,46.0;8.0,48.0'


def _image(n: str, indexed: str, **extra: Any) -> dict[str, Any]:
    return {
        'image_id': n,
        'image_path': f'/data/{n}.jpg',
        'width': 640,
        'height': 480,
        'source': 'disk',
        'indexed_at': indexed,
        **extra,
    }


def _item(n: str, image: str, **extra: Any) -> dict[str, Any]:
    return {
        'crop_id': n,
        'image_id': image,
        'crop_rank_in_image': 1,
        'bbox_norm': [0.1, 0.1, 0.5, 0.5],
        'class_id': 3,
        'class_name': 'sedan',
        **extra,
    }


def _world() -> QueryFakeOpenSearch:
    return QueryFakeOpenSearch(
        {
            ITEMS: {
                'a1': _item('a1', 'img-a', cluster_uid=str(CLUSTER_A)),
                'b1': _item('b1', 'img-b', cluster_uid=str(CLUSTER_B)),
                'c1': _item('c1', 'img-c', cluster_uid=str(CLUSTER_B)),
                'd1': _item('d1', 'img-d', near_dup_group_id=str(GROUP)),
            },
            IMAGES: {
                'img-a': _image(
                    'img-a', '2026-09-04', **{'capture.location': {'lat': 47.0, 'lon': 8.0}}
                ),
                'img-b': _image(
                    'img-b', '2026-09-03', **{'capture.location': {'lat': 46.5, 'lon': 8.2}}
                ),
                'img-c': _image(
                    'img-c',
                    '2026-09-02',
                    near_dup_group_id=str(GROUP),
                    **{'capture.location': {'lat': 10.0, 'lon': 100.0}},
                ),
                'img-d': _image('img-d', '2026-09-01'),
            },
        }
    )


def _policy(monkeypatch: pytest.MonkeyPatch, policy: str = 'store') -> None:
    async def _get() -> str:
        return policy

    for module in (image_join, images_view, map_route):
        monkeypatch.setattr(module, 'get_location_policy', _get)


def _client(fake: Any, monkeypatch: pytest.MonkeyPatch, policy: str = 'store') -> TestClient:
    from _curation_app import mount_curation_routers

    from src.core.db_session import get_db

    _policy(monkeypatch, policy)
    app = FastAPI()
    mount_curation_routers(app, curation_router)
    app.dependency_overrides[_raw_opensearch_dep] = lambda: fake

    async def no_database() -> Any:
        yield None

    app.dependency_overrides[get_db] = no_database
    return TestClient(app, raise_server_exceptions=False)


def _filter_routes() -> list[tuple[str, str]]:
    """Every (method, path) whose dependency tree holds the shared item filter."""

    def holds(dependant: Any) -> bool:
        return any(
            d.call is params_module.item_filter_query or holds(d) for d in dependant.dependencies
        )

    return sorted(
        (method, r.path)
        for r in curation_router.routes
        if isinstance(r, APIRoute) and holds(r.dependant)
        for method in (r.methods or ())
    )


# Route-specific parameters a route needs besides the filter, and the path parameters.
EXTRA_PARAMS: dict[str, dict[str, Any]] = {
    '/map/cells': {'zoom': 6},
    '/crops/locate': {'crop_id': 'a1'},
    '/images/locate': {'target_image_id': 'img-a'},
    '/review/{tab}/locate': {'crop_id': 'a1'},
    '/search/similar': {'crop_id': 'a1'},
    '/search/text': {'q': 'car'},
}
PATH_VALUES = {'tab': 'to_label'}


def _sent(fake: QueryFakeOpenSearch) -> str:
    return json.dumps(getattr(fake, 'sent', []))


class _Recording(QueryFakeOpenSearch):
    def __init__(self, indexes: Any) -> None:
        super().__init__(indexes)
        self.sent: list[dict[str, Any]] = []

    async def search(self, **kw: Any) -> dict[str, Any]:
        self.sent.append(kw.get('body') or {})
        return await super().search(**kw)

    async def msearch(self, *, body: list[dict[str, Any]], **_: Any) -> dict[str, Any]:
        self.sent.extend(body[1::2])
        return {'responses': [{'hits': {'total': {'value': 0}, 'hits': []}} for _ in body[::2]]}


def _recording_world() -> _Recording:
    return _Recording({name: dict(docs) for name, docs in _world().store.items()})


@pytest.mark.parametrize(('method', 'path'), _filter_routes())
def test_geo_bbox_narrows_every_filter_route(
    monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest, method: str, path: str
) -> None:
    if path == '/regions':
        request.getfixturevalue('reference_region_profile')
    fake = _recording_world()
    url = BASE + path.format(**PATH_VALUES)
    got = _client(fake, monkeypatch).request(
        method, url, params={'geo_bbox': BOX, **EXTRA_PARAMS.get(path, {})}
    )
    if path == '/pipeline/auto_label/start':  # a job's worker has no join: refused up front
        assert got.status_code == 422
        assert 'area, location or near-duplicate-group' in got.text
        return
    # Routes that need the database or an encoder fail after they ran their search.
    # A refusal names the parameter; a served body may echo it as null.
    assert got.status_code < 400 or 'geo_bbox' not in got.text, (path, got.text[:300])
    sent = _sent(fake)
    asked_images = '"geo_bounding_box"' in sent
    joined = '"image_id": ["img-a", "img-b"]' in sent
    assert asked_images or joined, (method, path)
    assert '"img-c"' not in sent.replace('"img-c": ', ''), (method, path)


def test_every_filter_route_is_known() -> None:
    assert {p for _, p in _filter_routes()} >= {
        '/crops',
        '/review/{tab}',
        '/clusters',
        '/images',
        '/map/cells',
        '/stats/dataset',
    }


def test_geo_bbox_returns_only_the_items_and_images_inside(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client(_world(), monkeypatch)
    crops = client.get(f'{BASE}/crops', params={'geo_bbox': BOX}).json()
    assert sorted(c['crop_id'] for c in crops['crops']) == ['a1', 'b1']
    assert crops['join_truncated'] is False
    images = client.get(f'{BASE}/images', params={'geo_bbox': BOX}).json()
    assert sorted(c['image_id'] for c in images['items']) == ['img-a', 'img-b']
    assert images['join_truncated'] is False
    unlocated = client.get(f'{BASE}/crops', params={'has_location': 'false'}).json()
    assert [c['crop_id'] for c in unlocated['crops']] == ['d1']


def test_geo_polygon_matches_a_triangle(monkeypatch: pytest.MonkeyPatch) -> None:
    # (8.0, 47.0) is inside the triangle; (8.2, 46.5) is inside too; (100, 10) is not.
    crops = _client(_world(), monkeypatch).get(
        f'{BASE}/crops', params={'geo_polygon': '7.9,46.0;8.1,46.0;8.0,47.5'}
    )
    assert sorted(c['crop_id'] for c in crops.json()['crops']) == ['a1']


@pytest.mark.parametrize('raw', ['1,2;3,4', 'a,b;c,d;e,f', '1,2;3,4;5,95'])
def test_bad_polygon_is_refused_with_the_served_text(
    monkeypatch: pytest.MonkeyPatch, raw: str
) -> None:
    got = _client(_world(), monkeypatch).get(f'{BASE}/crops', params={'geo_polygon': raw})
    assert got.status_code == 422
    assert 'geo_polygon needs 3 to 64 lon,lat vertices' in got.text


def test_filter_without_join_fails_closed() -> None:
    f = ItemFilter(geo_bbox=GeoBox(west=0, south=0, east=1, north=1))
    with pytest.raises(image_join.ImageJoinMissingError):
        item_filter_clauses(f, embedding=DEFAULT_WRITE_FIELDS)
    joined = item_filter_clauses(
        f,
        embedding=DEFAULT_WRITE_FIELDS,
        image_join=image_join.ImageJoin(image_ids=('x',), truncated=False),
    )
    assert joined == [{'terms': {'image_id': ['x']}}]


def test_join_truncated_is_served_past_the_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(image_join, 'JOIN_MAX_IMAGES', 1)
    crops = _client(_world(), monkeypatch).get(f'{BASE}/crops', params={'geo_bbox': BOX}).json()
    assert crops['join_truncated'] is True
    assert len(crops['crops']) == 1


def test_area_filters_follow_the_location_policy_on_the_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dropped = _client(_world(), monkeypatch, 'drop').get(f'{BASE}/crops', params={'geo_bbox': BOX})
    assert dropped.json()['crops'] == []


def test_near_dup_group_matches_item_and_image_level_members(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _client(_world(), monkeypatch)
    crops = client.get(f'{BASE}/crops', params={'near_dup_group': str(GROUP)}).json()
    # d1 carries the group itself; c1 is on img-c, which is in the group.
    assert sorted(c['crop_id'] for c in crops['crops']) == ['c1', 'd1']
    images = client.get(f'{BASE}/images', params={'near_dup_group': str(GROUP)}).json()
    assert [c['image_id'] for c in images['items']] == ['img-c']


@pytest.mark.parametrize(
    ('method', 'path'), [r for r in _filter_routes() if r[1] in {'/crops', '/clusters'}]
)
def test_cluster_uids_narrows_every_filter_route(
    monkeypatch: pytest.MonkeyPatch, method: str, path: str
) -> None:
    fake = _recording_world()
    got = _client(fake, monkeypatch).request(
        method,
        BASE + path,
        params={'cluster_uid': [str(CLUSTER_A), str(CLUSTER_B)]},
    )
    assert str(CLUSTER_A) in _sent(fake)
    assert str(CLUSTER_B) in _sent(fake)
    if path == '/crops':
        assert sorted(c['crop_id'] for c in got.json()['crops']) == ['a1', 'b1', 'c1']


def test_a_reprocess_refuses_image_predicates_and_suggests_nothing_for_them() -> None:
    from pydantic import ValidationError

    from src.services.curation.detections_summary import suggested_embed_request
    from src.services.curation.reprocess_models import ReprocessFilter

    box = GeoBox(west=0, south=0, east=1, north=1)
    with pytest.raises(ValidationError, match='area, location or near-duplicate-group'):
        ReprocessFilter(geo_bbox=box)
    assert suggested_embed_request(ItemFilter(geo_bbox=box)) is None
    assert suggested_embed_request(ItemFilter()) is not None


@pytest.mark.asyncio
async def test_join_stays_inside_the_result_window() -> None:
    """OpenSearch refuses from + size above 10000: the cap probe must not ask for one more."""
    seen: dict[str, Any] = {}

    class _Search:
        async def search(self, *, index: str, body: dict[str, Any]) -> dict[str, Any]:
            seen.update(body)
            hits = [{'_id': f'i{n}'} for n in range(min(body['size'], 3))]
            return {'hits': {'hits': hits, 'total': {'value': 10_001, 'relation': 'gte'}}}

    ids, truncated = await image_join.joined_image_ids(_Search(), index='images', clauses=[])
    assert seen['size'] <= 10_000
    assert truncated is True
    assert ids == ['i0', 'i1', 'i2']
