"""Unit tests for the batched-mget/bulk rewrite of
``occ_skip_on_conflict_bulk`` (see ``src/clients/occ.py``).

No real OpenSearch instance is required — a fake client double covers
the ``mget``/``bulk`` surface. The live-OpenSearch confirmation of the
bulk-409 contract these fakes reproduce lives in
``tests/integration/test_ingest_occ.py``.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

from curation.occ_fakes import FakeOccOpenSearch, make_bulk_response, make_bulk_update_item
from src.clients.occ import occ_skip_on_conflict_bulk


def _noop_merger(_doc_id: str, _source: dict[str, Any]) -> dict[str, Any]:
    return {}


class TestMixedPageOutcomes:
    @pytest.mark.asyncio
    async def test_conflict_success_noop_and_not_found_classified_correctly(self) -> None:
        """One page with a 409, a 200, a merger-noop, and a missing id."""
        sources = {
            'conflict-doc': {'class_validated': True, 'region_validated': False},
            'ok-doc': {'class_validated': False},
            'noop-doc': {'class_validated': False},
            # 'missing-doc' intentionally absent from sources.
        }
        client = FakeOccOpenSearch(
            sources,
            bulk_status={'conflict-doc': (409, 'version_conflict_engine_exception')},
        )

        def merger(doc_id: str, _source: dict[str, Any]) -> dict[str, Any]:
            if doc_id == 'noop-doc':
                return {}
            return {'field': 'value'}

        result = await occ_skip_on_conflict_bulk(
            client,
            doc_ids=['conflict-doc', 'ok-doc', 'noop-doc', 'missing-doc'],
            merger=merger,
        )

        assert result['updated'] == 1
        assert result['skipped_due_to_conflict'] == 1
        assert result['errors'] == [
            {'doc_id': 'missing-doc', 'phase': 'fetch', 'error': 'not_found'}
        ]
        # Exactly one mget + one bulk for the single page.
        assert len(client.mget_calls) == 1
        assert len(client.bulk_calls) == 1


class TestPaging:
    @pytest.mark.asyncio
    async def test_multi_page_input_issues_n_mgets_and_n_bulks(self) -> None:
        """5 ids at page_size=2 -> 3 pages -> 3 mgets + 3 bulks (page 3 has 1 id)."""
        doc_ids = [f'doc-{i}' for i in range(5)]
        sources = {doc_id: {'class_validated': False} for doc_id in doc_ids}
        client = FakeOccOpenSearch(sources)

        def merger(_doc_id: str, _source: dict[str, Any]) -> dict[str, Any]:
            return {'field': 'value'}

        result = await occ_skip_on_conflict_bulk(
            client, doc_ids=doc_ids, merger=merger, page_size=2
        )

        assert result['updated'] == 5
        assert result['skipped_due_to_conflict'] == 0
        assert result['errors'] == []
        assert len(client.mget_calls) == 3
        assert len(client.bulk_calls) == 3
        assert [len(ids) for ids in client.mget_calls] == [2, 2, 1]


class TestBulkExceptionDegradesGracefully:
    @pytest.mark.asyncio
    async def test_bulk_level_exception_produces_errors_not_raise(self) -> None:
        sources = {'a': {'class_validated': False}, 'b': {'class_validated': False}}
        client = AsyncMock()
        client.mget = AsyncMock(
            return_value={
                'docs': [
                    {
                        '_index': 'op_items',
                        '_id': doc_id,
                        '_seq_no': 1,
                        '_primary_term': 1,
                        'found': True,
                        '_source': source,
                    }
                    for doc_id, source in sources.items()
                ]
            }
        )
        client.bulk = AsyncMock(side_effect=ConnectionError('OS unreachable'))

        def merger(_doc_id: str, _source: dict[str, Any]) -> dict[str, Any]:
            return {'field': 'value'}

        result = await occ_skip_on_conflict_bulk(client, doc_ids=['a', 'b'], merger=merger)

        assert result['updated'] == 0
        assert result['skipped_due_to_conflict'] == 0
        assert len(result['errors']) == 2
        assert {e['doc_id'] for e in result['errors']} == {'a', 'b'}
        assert all(e['phase'] == 'update' for e in result['errors'])


class TestRefreshForwarding:
    @pytest.mark.asyncio
    async def test_refresh_param_forwarded_verbatim(self) -> None:
        for refresh_value in (True, False, 'wait_for'):
            sources = {'a': {'class_validated': False}}
            mget_resp = {
                'docs': [
                    {
                        '_index': 'op_items',
                        '_id': 'a',
                        '_seq_no': 1,
                        '_primary_term': 1,
                        'found': True,
                        '_source': sources['a'],
                    }
                ]
            }
            client = AsyncMock()
            client.mget = AsyncMock(return_value=mget_resp)
            client.bulk = AsyncMock(
                return_value=make_bulk_response([make_bulk_update_item('a', status=200)])
            )

            def merger(_doc_id: str, _source: dict[str, Any]) -> dict[str, Any]:
                return {'field': 'value'}

            await occ_skip_on_conflict_bulk(
                client, doc_ids=['a'], merger=merger, refresh=refresh_value
            )
            bulk_call = client.bulk.await_args
            assert bulk_call is not None
            assert bulk_call.kwargs['refresh'] == refresh_value


class TestEmptyInput:
    @pytest.mark.asyncio
    async def test_empty_doc_ids_short_circuits(self) -> None:
        client = AsyncMock()
        result = await occ_skip_on_conflict_bulk(client, doc_ids=[], merger=_noop_merger)
        assert result == {'updated': 0, 'skipped_due_to_conflict': 0, 'errors': []}
        client.mget.assert_not_awaited()
        client.bulk.assert_not_awaited()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
