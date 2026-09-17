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
