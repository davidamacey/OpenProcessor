"""Unit tests for ``src.clients.occ_bulk.occ_update_bulk``:
human-write-semantics batch OCC update (retry on 409 rather than
skip-on-conflict, unlike ``occ_skip_on_conflict_bulk``).

Split out of ``test_occ.py`` alongside the ``occ_update_bulk`` ->
``src/clients/occ_bulk.py`` move (kept ``occ.py`` under the 700-LOC
ratchet).
"""

from __future__ import annotations

from typing import Any

import pytest

from curation.occ_fakes import FakeOccOpenSearch, make_bulk_response, make_bulk_update_item
from src.clients.occ_bulk import occ_update_bulk


class TestOccUpdateBulk:
    @pytest.mark.asyncio
    async def test_one_mget_and_one_bulk_call_for_a_50_id_batch(self) -> None:
        doc_ids = [f'crop-{i}' for i in range(50)]
        sources = {doc_id: {'class_id': 1} for doc_id in doc_ids}
        client = FakeOccOpenSearch(sources)

        def merge_fn(_doc_id: str, _source: dict[str, Any]) -> dict[str, Any]:
            return {'class_id': 2}

        status = await occ_update_bulk(
            client, index='op_items', ids=doc_ids, merge_fn=merge_fn, refresh='wait_for'
        )

        assert all(v == 'updated' for v in status.values())
        assert len(client.mget_calls) == 1
        assert len(client.bulk_calls) == 1

    @pytest.mark.asyncio
    async def test_not_found_ids_are_reported(self) -> None:
        client = FakeOccOpenSearch({'a': {'class_id': 1}})

        def merge_fn(_doc_id: str, _source: dict[str, Any]) -> dict[str, Any]:
            return {'class_id': 2}

        status = await occ_update_bulk(
            client, index='op_items', ids=['a', 'missing'], merge_fn=merge_fn
        )
        assert status['a'] == 'updated'
        assert status['missing'] == 'not-found'

    @pytest.mark.asyncio
    async def test_conflict_retries_then_succeeds(self) -> None:
        """First bulk call 409s on 'a'; the retry round re-mgets and
        succeeds — proving occ_update_bulk (unlike occ_skip_on_conflict_bulk)
        retries a conflict instead of silently skipping it."""

        class _RetryOnceClient(FakeOccOpenSearch):
            def __init__(self) -> None:
                super().__init__({'a': {'class_id': 1}})
                self._bulk_attempt = 0

            async def bulk(
                self,
                *,
                body: list[dict[str, Any]],
                refresh: bool | str = False,  # noqa: ARG002
            ) -> dict[str, Any]:
                self._bulk_attempt += 1
                self.bulk_calls.append(body)
                if self._bulk_attempt == 1:
                    return make_bulk_response(
                        [
                            make_bulk_update_item(
                                'a', status=409, error_type='version_conflict_engine_exception'
                            )
                        ]
                    )
                return make_bulk_response([make_bulk_update_item('a', status=200)])

        client = _RetryOnceClient()

        def merge_fn(_doc_id: str, _source: dict[str, Any]) -> dict[str, Any]:
            return {'class_id': 2}

        status = await occ_update_bulk(client, index='op_items', ids=['a'], merge_fn=merge_fn)
        assert status['a'] == 'updated'
        assert len(client.bulk_calls) == 2
        assert len(client.mget_calls) == 2

    @pytest.mark.asyncio
    async def test_conflict_exhausted_after_max_retries(self) -> None:
        client = FakeOccOpenSearch(
            {'a': {'class_id': 1}},
            bulk_status={'a': (409, 'version_conflict_engine_exception')},
        )

        def merge_fn(_doc_id: str, _source: dict[str, Any]) -> dict[str, Any]:
            return {'class_id': 2}

        status = await occ_update_bulk(
            client, index='op_items', ids=['a'], merge_fn=merge_fn, max_retries=1
        )
        assert status['a'] == 'conflict-exhausted'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
