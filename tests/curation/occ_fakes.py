"""Shared fake-OpenSearch helpers for occ_skip_on_conflict_bulk callers.

Since P2-13 (reference-line audit) rewrote ``occ_skip_on_conflict_bulk``
to use batched ``_mget`` + ``_bulk`` instead of per-doc ``get``/
``update``, every test double that stands in for an ``AsyncOpenSearch``
client in a code path that calls it needs ``mget`` + ``bulk`` methods
(not ``get``/``update``). This module centralizes that fake-response
construction so it isn't duplicated across test files.

Ported ahead of its originally-scheduled wave (the plan slots
``src/clients/occ.py`` itself into a later curation wave) because
``tests/curation/test_clustering_orchestrator_extra.py`` (Chunk 4)
needs it for ``auto_promote_clusters``'s OCC-bulk write path — these
helpers only build plain response-dict shapes and have zero import
dependency on ``src.clients.occ``, so porting them early carries no
forward-reference risk.
"""

from __future__ import annotations

from typing import Any


_FAKE_INDEX = 'op_items'


def make_mget_response(
    docs: dict[str, dict[str, Any]],
    *,
    seq_no: int = 1,
    primary_term: int = 1,
) -> dict[str, Any]:
    """Build a ``client.mget`` response body for the given ``{id: source}``.

    Every doc is marked ``found`` with the given ``_seq_no``/
    ``_primary_term`` (bump per-doc via a wrapping dict if a test needs
    per-doc versions).
    """
    return {
        'docs': [
            {
                '_index': _FAKE_INDEX,
                '_id': doc_id,
                '_version': 1,
                '_seq_no': seq_no,
                '_primary_term': primary_term,
                'found': True,
                '_source': source,
            }
            for doc_id, source in docs.items()
        ],
    }


def make_bulk_update_item(
    doc_id: str,
    *,
    status: int = 200,
    error_type: str | None = None,
) -> dict[str, Any]:
    """Build one ``items[]`` entry of a ``client.bulk`` response.

    ``status=409`` + ``error_type='version_conflict_engine_exception'``
    reproduces the live-OpenSearch conflict shape confirmed against a
    real 3.6.0 cluster.
    """
    item: dict[str, Any] = {
        'update': {
            '_index': _FAKE_INDEX,
            '_id': doc_id,
            'status': status,
        },
    }
    if status in (200, 201):
        item['update']['result'] = 'updated'
        item['update']['_seq_no'] = 2
        item['update']['_primary_term'] = 1
    else:
        item['update']['error'] = {
            'type': error_type or 'exception',
            'reason': f'{doc_id} failed',
        }
    return item


def make_bulk_response(items: list[dict[str, Any]]) -> dict[str, Any]:
    errors = any((i.get('update', {}).get('status') not in (200, 201)) for i in items)
    return {'errors': errors, 'items': items}


class FakeOccOpenSearch:
    """Minimal AsyncOpenSearch double covering the mget+bulk surface used
    by occ_skip_on_conflict_bulk.

    ``sources``: ``{doc_id: source_dict}`` for docs mget should find
    (any id not present is a "not found" fetch error).
    ``bulk_status``: optional ``{doc_id: (status, error_type)}`` override
    for the bulk response; defaults to 200 for every updated doc.
    """

    def __init__(
        self,
        sources: dict[str, dict[str, Any]],
        *,
        bulk_status: dict[str, tuple[int, str | None]] | None = None,
        seq_no: int = 1,
        primary_term: int = 1,
    ) -> None:
        self.sources = sources
        self.bulk_status = bulk_status or {}
        self.seq_no = seq_no
        self.primary_term = primary_term
        self.mget_calls: list[list[str]] = []
        self.bulk_calls: list[list[dict[str, Any]]] = []

    async def mget(self, *, body: dict[str, Any]) -> dict[str, Any]:
        ids = [d['_id'] for d in body['docs']]
        self.mget_calls.append(ids)
        found = {doc_id: self.sources[doc_id] for doc_id in ids if doc_id in self.sources}
        return make_mget_response(found, seq_no=self.seq_no, primary_term=self.primary_term)

    async def bulk(
        self,
        *,
        body: list[dict[str, Any]],
        refresh: bool | str = False,  # noqa: ARG002 - part of the AsyncOpenSearch.bulk signature
    ) -> dict[str, Any]:
        self.bulk_calls.append(body)
        items = []
        for action, _doc in zip(body[0::2], body[1::2], strict=True):
            doc_id = action['update']['_id']
            status, error_type = self.bulk_status.get(doc_id, (200, None))
            items.append(make_bulk_update_item(doc_id, status=status, error_type=error_type))
        return make_bulk_response(items)


class FakeIngestOpenSearch:
    """AsyncOpenSearch double covering the surface
    :func:`src.clients.occ.occ_upsert_bulk` and
    :class:`src.services.curation.ingest.CurationIngestService` need:
    ``search``/``msearch`` (dedup), ``mget``/``bulk``/``update``/``get``
    (the create-then-OCC-update upsert path), and a blind ``bulk`` index
    for the images doc.

    Backed by two plain dicts (``images``, ``items``) rather than
    real OpenSearch query evaluation — ``search``/``msearch`` only
    understand the ``term: {imohash: ...}`` query ingest issues for
    dedup, which is all this double needs to support.
    """

    def __init__(
        self,
        *,
        images: dict[str, dict[str, Any]] | None = None,
        items: dict[str, dict[str, Any]] | None = None,
        create_conflict_ids: set[str] | None = None,
    ) -> None:
        self.images: dict[str, dict[str, Any]] = dict(images or {})
        self.items: dict[str, dict[str, Any]] = dict(items or {})
        self._seq: dict[str, int] = dict.fromkeys(self.items, 1)
        # Ids that should 409 on their FIRST bulk `create` attempt, to
        # exercise occ_upsert_bulk's create-conflict -> refetch -> OCC
        # update fallback path.
        self._create_conflict_pending: set[str] = set(create_conflict_ids or set())
        self.bulk_calls: list[list[dict[str, Any]]] = []
        self.update_calls: list[dict[str, Any]] = []

    async def search(self, *, index: str, body: dict[str, Any]) -> dict[str, Any]:
        from src.config import get_curation_config

        term = ((body.get('query') or {}).get('term') or {}).get('imohash')
        store = self.items if index == get_curation_config().items_index else self.images
        hits = [
            {'_id': doc.get('image_id', doc_id), '_source': doc}
            for doc_id, doc in store.items()
            if term is None or doc.get('imohash') == term
        ]
        return {'hits': {'hits': hits[: body.get('size', len(hits))]}}

    async def msearch(self, *, body: list[dict[str, Any]]) -> dict[str, Any]:
        responses = []
        for line in body[1::2]:
            term = ((line.get('query') or {}).get('term') or {}).get('imohash')
            hits = [
                {'_id': doc.get('image_id', doc_id), '_source': doc}
                for doc_id, doc in self.images.items()
                if doc.get('imohash') == term
            ]
            responses.append({'hits': {'hits': hits[:1]}})
        return {'responses': responses}

    async def mget(self, *, body: dict[str, Any], index: str) -> dict[str, Any]:  # noqa: ARG002
        ids = body['ids']
        docs = []
        for doc_id in ids:
            if doc_id in self.items:
                docs.append(
                    {
                        '_id': doc_id,
                        'found': True,
                        '_source': self.items[doc_id],
                        '_seq_no': self._seq.get(doc_id, 1),
                        '_primary_term': 1,
                    }
                )
            else:
                docs.append({'_id': doc_id, 'found': False})
        return {'docs': docs}

    async def bulk(
        self,
        *,
        body: list[dict[str, Any]],
        refresh: bool | str = False,  # noqa: ARG002
    ) -> dict[str, Any]:
        from src.config import get_curation_config

        items_index = get_curation_config().items_index
        self.bulk_calls.append(body)
        items: list[dict[str, Any]] = []
        for action, doc in zip(body[0::2], body[1::2], strict=True):
            if 'index' in action:
                meta = action['index']
                doc_id = meta['_id']
                if meta['_index'] != items_index:
                    self.images[doc_id] = doc
                else:
                    self.items[doc_id] = doc
                    self._seq[doc_id] = 1
                items.append({'index': {'_id': doc_id, 'status': 201}})
            elif 'create' in action:
                meta = action['create']
                doc_id = meta['_id']
                if doc_id in self._create_conflict_pending:
                    self._create_conflict_pending.discard(doc_id)
                    items.append({'create': {'_id': doc_id, 'status': 409}})
                elif doc_id in self.items:
                    items.append({'create': {'_id': doc_id, 'status': 409}})
                else:
                    self.items[doc_id] = doc
                    self._seq[doc_id] = 1
                    items.append({'create': {'_id': doc_id, 'status': 201}})
            else:  # pragma: no cover - defensive, ingest never emits bare 'update' actions in bulk
                msg = f'unsupported bulk action: {action}'
                raise ValueError(msg)
        errors = any(next(iter(i.values())).get('status') not in (200, 201) for i in items)
        return {'errors': errors, 'items': items}

    async def update(
        self,
        *,
        index: str,  # noqa: ARG002
        id: str,  # noqa: A002
        body: dict[str, Any],
        if_seq_no: int,
        if_primary_term: int,  # noqa: ARG002
        refresh: bool | str = False,  # noqa: ARG002
    ) -> dict[str, Any]:
        self.update_calls.append({'id': id, 'body': body, 'if_seq_no': if_seq_no})
        current_seq = self._seq.get(id, 1)
        if if_seq_no != current_seq:
            msg = f'version_conflict_engine_exception: {id}'
            raise RuntimeError(msg)
        self.items.setdefault(id, {}).update(body['doc'])
        self._seq[id] = current_seq + 1
        return {'_id': id, 'result': 'updated', '_seq_no': self._seq[id], '_primary_term': 1}

    async def get(self, *, index: str, id: str) -> dict[str, Any]:  # noqa: A002, ARG002
        return {
            '_id': id,
            '_source': self.items.get(id, {}),
            '_seq_no': self._seq.get(id, 1),
            '_primary_term': 1,
        }
