"""Route walk: every curation GET route that lists items declares how its
entries are unique within a page. A new list route must be added to one of
the two sets below, which forces the author to decide."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI


# One entry per distinct OpenSearch document (single search per page, or a
# ranked distinct-id list hydrated by mget), so ``crop_id`` is unique.
ONE_ENTRY_PER_DOC = {
    '/audit/queue',
    '/classes/{class_id}/crops',
    '/crops',
    '/crops/{crop_id}/context',
    '/review/{tab}',
    '/search/text',
    '/viz/projection',
}
# Per-box rows: ``crop_id`` repeats by design; ``row_key`` (required) is unique.
ROW_KEYED = {'/regions', '/regions/suspected_false_positives', '/regions/training_candidates'}
PREFIX = '/curation/projects/{project}'


def _app() -> FastAPI:
    from _curation_app import mount_curation_routers

    from src.routers.curation import router as curation_router

    app = FastAPI()
    mount_curation_routers(app, curation_router)
    return app


def _list_routes(schema: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """``{route: item schema}`` for GET routes whose 200 body has an array of crop items."""
    comps = schema['components']['schemas']

    def resolve(s: dict[str, Any]) -> dict[str, Any]:
        while '$ref' in s:
            s = comps[s['$ref'].split('/')[-1]]
        return s

    def props(s: dict[str, Any]) -> dict[str, Any]:
        s = resolve(s)
        merged = dict(s.get('properties', {}))
        for part in s.get('allOf', []):
            merged.update(props(part))
        return merged

    found: dict[str, dict[str, Any]] = {}
    for path, ops in schema['paths'].items():
        body = (
            ops.get('get', {})
            .get('responses', {})
            .get('200', {})
            .get('content', {})
            .get('application/json', {})
            .get('schema')
        )
        if not body or not path.startswith(PREFIX + '/'):
            continue
        for raw in props(body).values():
            field = resolve(raw)
            item = field.get('items') if field.get('type') == 'array' else None
            if item and 'crop_id' in props(item):
                found[path.removeprefix(PREFIX)] = resolve(item)
    return found


def test_every_item_list_route_declares_its_uniqueness() -> None:
    routes = _list_routes(_app().openapi())
    undeclared = set(routes) - ONE_ENTRY_PER_DOC - ROW_KEYED
    assert not undeclared, f'new item list route(s) need a uniqueness decision: {undeclared}'
    stale = (ONE_ENTRY_PER_DOC | ROW_KEYED) - set(routes)
    assert not stale, f'allowlisted route(s) no longer list items: {stale}'


def test_row_keyed_routes_require_row_key() -> None:
    routes = _list_routes(_app().openapi())
    for route in ROW_KEYED:
        assert 'row_key' in routes[route].get('required', []), route
