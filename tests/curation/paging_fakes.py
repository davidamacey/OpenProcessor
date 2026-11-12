"""An OpenSearch double that orders by a full ``sort`` list and honours ``search_after``.

``QueryFakeOpenSearch`` sorts on the first key only. The cursor tests need the whole sort (a field,
then the ``crop_id`` tiebreak), a ``missing`` placement, ``search_after`` that compares the whole key,
and the real window limit: ``from + size`` past 10,000 fails the way the cluster does.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any

from curation.query_fakes import QueryFakeOpenSearch, matches


if TYPE_CHECKING:
    import pytest
    from fastapi.testclient import TestClient

WINDOW = 10_000


def _key_parts(sort: list[dict[str, Any] | str]) -> list[tuple[str, str, str]]:
    parts = []
    for entry in sort:
        field, spec = (entry, 'asc') if isinstance(entry, str) else next(iter(entry.items()))
        order = spec if isinstance(spec, str) else spec.get('order', 'asc')
        missing = 'last' if isinstance(spec, str) else str(spec.get('missing', '_last')).strip('_')
        parts.append((field, order, missing))
    return parts


def _compare(a: Any, b: Any, order: str, missing: str) -> int:
    if a is None or b is None:
        if a is None and b is None:
            return 0
        first = (missing == 'first') == (a is None)
        return -1 if first else 1
    result = (a > b) - (a < b)
    return -result if order == 'desc' else result


class PagingOpenSearch(QueryFakeOpenSearch):
    """``search`` over the items index with total ordering and a real result window."""

    def __init__(self, indexes: dict[str, dict[str, dict[str, Any]]] | None = None) -> None:
        super().__init__(indexes)
        self.bodies: list[dict[str, Any]] = []

    async def search(self, *, index: str, body: dict[str, Any], **_kw: Any) -> dict[str, Any]:
        self.bodies.append(body)
        if body.get('from', 0) + body.get('size', 10) > WINDOW:
            msg = 'Result window is too large'
            raise RuntimeError(msg)
        pool = [
            {**doc, 'crop_id': doc.get('crop_id', doc_id)}
            for doc_id, doc in self.docs(index).items()
            if matches(doc, body.get('query'))
        ]
        parts = _key_parts(body.get('sort') or [])

        def values(doc: dict[str, Any]) -> list[Any]:
            return [doc.get(field) for field, _o, _m in parts]

        def cmp_values(a: list[Any], b: list[Any]) -> int:
            for (_f, order, missing), x, y in zip(parts, a, b, strict=False):
                if (c := _compare(x, y, order, missing)) != 0:
                    return c
            return 0

        pool.sort(key=functools.cmp_to_key(lambda a, b: cmp_values(values(a), values(b))))
        total = len(pool)
        after = body.get('search_after')
        if after is not None:
            pool = [d for d in pool if cmp_values(values(d), after) > 0]
        start = body.get('from', 0)
        hits = [
            {'_id': d['crop_id'], '_source': d, 'sort': values(d)}
            for d in pool[start : start + body.get('size', 10)]
        ]
        return {'hits': {'total': {'value': total}, 'hits': hits}}


class AggregatingPagingOpenSearch(PagingOpenSearch):
    """Total ordering for page reads; aggregation reads (the item join) go to the base fake."""

    async def search(self, *, index: str, body: dict[str, Any], **kw: Any) -> dict[str, Any]:
        if 'aggs' in body or 'aggregations' in body:
            return await QueryFakeOpenSearch.search(self, index=index, body=body, **kw)
        return await super().search(index=index, body=body, **kw)


def images_client(fake: Any, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """``crops_client`` with the location policy of the image routes pinned to ``store``."""
    from src.routers.curation import images_view, table_view

    async def _policy() -> str:
        return 'store'

    for module in (images_view, table_view):
        monkeypatch.setattr(module, 'get_location_policy', _policy)
    return crops_client(fake, monkeypatch)


def crops_client(fake: Any, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from src.routers.curation import _raw_opensearch_dep, router as curation_router
    from src.services.curation import index_bootstrap

    monkeypatch.setattr(index_bootstrap, '_INDEXES_BOOTSTRAPPED', {'default'})
    app = FastAPI()
    from _curation_app import mount_curation_routers

    mount_curation_routers(app, curation_router)
    app.dependency_overrides[_raw_opensearch_dep] = lambda: fake
    return TestClient(app)
