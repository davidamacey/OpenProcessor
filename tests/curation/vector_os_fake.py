"""An OpenSearch double that serves vector reads in every shape the readers use.

It answers a plain scroll, a point in time with ``search_after`` and slices, ``_source``
filtering and ``docvalue_fields`` (a base64 little-endian float32 string under ``fields``, the
default 3.7 response). Every search body is kept in ``searches`` so a test can assert the request.
The query is not evaluated: a test seeds exactly the documents the query would match.
"""

from __future__ import annotations

import base64
from typing import Any

import numpy as np


class VectorOS:
    def __init__(
        self, docs: list[tuple[str, dict[str, Any]]], vector_field: str = 'pe_embedding'
    ) -> None:
        """``docs``: ``(id, fields)`` in index order; a vector is a float32 array under ``vector_field``."""
        self.docs = docs
        self.field = vector_field
        self.searches: list[dict[str, Any]] = []
        self.pits_open = 0
        self.pits_deleted = 0
        self._scrolls: dict[str, int] = {}
        self._scroll_body: dict[str, Any] = {}

    async def count(
        self, *, index: str | None = None, body: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        return {'count': len(self.docs)}

    async def create_pit(self, **_kw: Any) -> dict[str, str]:
        self.pits_open += 1
        return {'pit_id': 'pit-1'}

    async def delete_pit(self, **_kw: Any) -> dict[str, Any]:
        self.pits_deleted += 1
        return {}

    def _hit(self, pos: int, body: dict[str, Any]) -> dict[str, Any]:
        doc_id, fields = self.docs[pos]
        hit: dict[str, Any] = {'_id': doc_id, 'sort': [pos]}
        wanted = body.get('_source')
        if wanted:
            src: dict[str, Any] = {}
            for key in wanted:
                if key not in fields:
                    continue
                value = fields[key]
                src[key] = value.tolist() if isinstance(value, np.ndarray) else value
            hit['_source'] = src
        if self.field in (body.get('docvalue_fields') or []) and self.field in fields:
            raw = np.asarray(fields[self.field], dtype='<f4').tobytes()
            hit['fields'] = {self.field: [base64.b64encode(raw).decode('ascii')]}
        return hit

    async def search(
        self, *, body: dict[str, Any], index: str | None = None, **kw: Any
    ) -> dict[str, Any]:
        self.searches.append(body)
        positions = list(range(len(self.docs)))
        if 'slice' in body:
            positions = [p for p in positions if p % body['slice']['max'] == body['slice']['id']]
        after = (body.get('search_after') or [-1])[0]
        positions = [p for p in positions if p > after]
        page = positions[: body['size']]
        out: dict[str, Any] = {'hits': {'hits': [self._hit(p, body) for p in page]}}
        if kw.get('scroll'):
            self._scroll_body = body
            self._scrolls['s'] = page[-1] + 1 if page else len(self.docs)
            out['_scroll_id'] = 's'
        return out

    async def scroll(self, *, scroll_id: str, **_kw: Any) -> dict[str, Any]:
        start = self._scrolls[scroll_id]
        page = list(range(start, min(start + self._scroll_body['size'], len(self.docs))))
        self._scrolls[scroll_id] = start + len(page)
        hits = [self._hit(p, self._scroll_body) for p in page]
        return {'_scroll_id': scroll_id, 'hits': {'hits': hits}}

    async def clear_scroll(self, **_kw: Any) -> dict[str, Any]:
        return {}


def random_docs(
    n: int, dim: int = 8, *, seed: int = 0, missing_every: int = 7
) -> list[tuple[str, dict[str, Any]]]:
    """``n`` documents with shuffled ids and random float32 vectors; every ``missing_every``-th has none."""
    rng = np.random.default_rng(seed)
    ids = [f'doc-{i:05d}' for i in rng.permutation(n)]
    docs: list[tuple[str, dict[str, Any]]] = []
    for i, doc_id in enumerate(ids):
        fields: dict[str, Any] = {'image_id': f'img-{i % 5}', 'detector_class_name': 'car'}
        if i % missing_every != 0:
            fields['pe_embedding'] = rng.standard_normal(dim).astype(np.float32)
        docs.append((doc_id, fields))
    return docs


def docvalue_hit(doc_id: str, vector: list[float], field: str = 'pe_embedding') -> dict[str, Any]:
    """A hit that carries ``vector`` the way a ``docvalue_fields`` request returns it."""
    raw = np.asarray(vector, dtype='<f4').tobytes()
    return {'_id': doc_id, 'fields': {field: [base64.b64encode(raw).decode('ascii')]}}


def no_point_in_time() -> Any:
    """``create_pit`` for a mock client: the page reader then reads with a plain scroll."""
    from unittest.mock import AsyncMock

    from opensearchpy.exceptions import OpenSearchException

    return AsyncMock(side_effect=OpenSearchException(400, 'point in time is not served here'))
