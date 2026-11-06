"""The OpenSearch access of the upgrade: an unguarded admin client in two narrow wrappers.

The project guard (``src.services.projects.guard``) refuses any request that reaches a project's
index from outside the project and refuses the global control indexes while a project is bound;
the upgrade reads every project's control data, so it builds its own client (an allowed
construction site of ``tests/projects/test_no_frozen_project_config.py``) and narrows it:

* :class:`ControlReader` can issue only ``GET`` of a document, ``search`` with ``search_after``,
  ``_mapping`` / ``_settings`` reads and ``HEAD`` of an index. It has no other method.
* :class:`ControlWriter` can create an index, bulk-index documents and delete an index; the
  rollback and ``delete_sources`` use it. Nothing in the export holds one.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from opensearchpy import AsyncOpenSearch
from opensearchpy.exceptions import NotFoundError

from src.clients.opensearch.serializer import OrjsonSerializer
from src.config.env import env_optional_str


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterable

PAGE_SIZE = 1000
# Index settings OpenSearch assigns itself; a recreated index must not carry them.
SYSTEM_SETTINGS = frozenset({'uuid', 'creation_date', 'version', 'provided_name', 'resize'})


def build_client(url: str | None = None) -> AsyncOpenSearch:
    """An ``AsyncOpenSearch`` with no project guard, for the upgrade scripts only."""
    return AsyncOpenSearch(
        hosts=[url or env_optional_str('OPENSEARCH_URL') or 'http://localhost:9200'],
        use_ssl=False,
        serializer=OrjsonSerializer(),
        timeout=60,
    )


class ControlReader:
    """Read-only view of OpenSearch: documents, mappings, settings, existence."""

    def __init__(self, client: Any) -> None:
        self._client = client

    async def index_exists(self, name: str) -> bool:
        return bool(await self._client.indices.exists(index=name))

    async def definition(self, name: str) -> tuple[dict[str, Any], dict[str, Any]]:
        """``(mappings, settings)`` of an index, settings without the ones OpenSearch assigns."""
        mapping = await self._client.indices.get_mapping(index=name)
        settings = await self._client.indices.get_settings(index=name)
        (body,) = mapping.values()
        (cfg,) = settings.values()
        index_settings = dict(cfg.get('settings', {}).get('index', {}))
        clean = {k: v for k, v in index_settings.items() if k not in SYSTEM_SETTINGS}
        return dict(body.get('mappings', {})), {'index': clean}

    async def get_field(self, name: str, ids: list[str], field: str) -> dict[str, Any]:
        """``{id: value of field}`` for the documents of ``ids`` that exist and have the field
        (one ``mget`` of that single ``_source`` field)."""
        resp = await self._client.mget(index=name, body={'ids': ids}, _source_includes=[field])
        return {
            str(d['_id']): d['_source'][field]
            for d in resp['docs']
            if d.get('found') and field in (d.get('_source') or {})
        }

    async def iter_docs(self, name: str) -> AsyncIterator[tuple[str, dict[str, Any]]]:
        """Every document as ``(id, source)``, paged with ``search_after`` on ``_doc``."""
        after: list[Any] | None = None
        while True:
            body: dict[str, Any] = {
                'size': PAGE_SIZE,
                'sort': [{'_doc': 'asc'}],
                'query': {'match_all': {}},
            }
            if after is not None:
                body['search_after'] = after
            hits = (await self._client.search(index=name, body=body))['hits']['hits']
            for hit in hits:
                yield str(hit['_id']), hit['_source']
            if len(hits) < PAGE_SIZE:
                return
            after = hits[-1]['sort']


class ControlWriter:
    """The writes of the rollback and of ``delete_sources``; both tolerate "already so"."""

    def __init__(self, client: Any) -> None:
        self._client = client

    async def create_index(
        self, name: str, mappings: dict[str, Any], settings: dict[str, Any]
    ) -> None:
        await self._client.indices.create(
            index=name, body={'mappings': mappings, 'settings': settings}
        )

    async def bulk_index(self, name: str, docs: Iterable[tuple[str, dict[str, Any]]]) -> int:
        """Index ``docs`` in batches; raises when any item fails. Returns the count."""
        total = 0
        batch: list[Any] = []

        async def flush() -> None:
            resp = await self._client.bulk(body=batch, refresh=False)
            if resp.get('errors'):
                failed = [i for i in resp['items'] if next(iter(i.values())).get('error')]
                msg = f'{name}: {len(failed)} documents were refused: {failed[0]}'
                raise RuntimeError(msg)

        for doc_id, source in docs:
            batch.extend([{'index': {'_index': name, '_id': doc_id}}, source])
            total += 1
            if len(batch) >= 2 * PAGE_SIZE:
                await flush()
                batch = []
        if batch:
            await flush()
        await self._client.indices.refresh(index=name)
        return total

    async def count(self, name: str, query: dict[str, Any]) -> int:
        return int((await self._client.count(index=name, body={'query': query})).get('count') or 0)

    async def update_by_query(
        self, name: str, body: dict[str, Any], *, requests_per_second: int
    ) -> int:
        """Run an ``update_by_query`` as a polled task (never one blocking request); returns the
        number of documents updated."""
        from src.clients.opensearch.tasks import run_update_by_query_polled

        resp = await run_update_by_query_polled(
            self._client, index=name, body=body, slices=1, requests_per_second=requests_per_second
        )
        return int(resp.get('updated', 0))

    async def delete_index(self, name: str) -> bool:
        """Delete an index; ``False`` when it was already gone."""
        try:
            await self._client.indices.delete(index=name)
        except NotFoundError:
            return False
        return True
