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
from opensearchpy.exceptions import (
    ConnectionError as OsConnectionError,
    OpenSearchException,
    RequestError,
)

from curation.occ_fakes import (
    VECTOR_EXCLUDES,
    FakeOccOpenSearch,
    make_bulk_response,
    make_bulk_update_item,
)
from src.clients.occ_bulk import occ_bulk_rounds, occ_update_bulk


class TestOccUpdateBulk:
    @pytest.mark.asyncio
    async def test_one_mget_and_one_bulk_call_for_a_50_id_batch(self) -> None:
        doc_ids = [f'crop-{i}' for i in range(50)]
        sources = {doc_id: {'class_id': 1} for doc_id in doc_ids}
        client = FakeOccOpenSearch(sources)

        def merge_fn(_doc_id: str, _source: dict[str, Any]) -> dict[str, Any]:
            return {'class_id': 2}

        status = await occ_update_bulk(
            client,
            vector_excludes=VECTOR_EXCLUDES,
            index='op_prj_default__items',
            ids=doc_ids,
            merge_fn=merge_fn,
            refresh='wait_for',
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
            client,
            vector_excludes=VECTOR_EXCLUDES,
            index='op_prj_default__items',
            ids=['a', 'missing'],
            merge_fn=merge_fn,
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
                refresh: bool | str = False,
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

        status = await occ_update_bulk(
            client,
            vector_excludes=VECTOR_EXCLUDES,
            index='op_prj_default__items',
            ids=['a'],
            merge_fn=merge_fn,
        )
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
            client,
            vector_excludes=VECTOR_EXCLUDES,
            index='op_prj_default__items',
            ids=['a'],
            merge_fn=merge_fn,
            max_retries=1,
        )
        assert status['a'] == 'conflict-exhausted'

    @pytest.mark.asyncio
    async def test_non_409_item_error_is_error_not_conflict(self) -> None:
        client = FakeOccOpenSearch(
            {'a': {'class_id': 1}}, bulk_status={'a': (400, 'mapper_parsing_exception')}
        )
        status = await occ_update_bulk(
            client,
            vector_excludes=VECTOR_EXCLUDES,
            index='op_prj_default__items',
            ids=['a'],
            merge_fn=lambda _i, _s: {'class_id': 2},
        )
        assert status['a'] == 'error'

    @pytest.mark.asyncio
    async def test_whole_request_400_is_error_not_conflict(self) -> None:
        client = _AlwaysConflict(
            transport_error=True, error=RequestError(400, 'mapper_parsing_exception', {})
        )
        status = await occ_update_bulk(
            client,
            vector_excludes=VECTOR_EXCLUDES,
            index='op_prj_default__items',
            ids=['a'],
            merge_fn=lambda _i, _s: {'class_id': 2},
            max_retries=3,
        )
        assert status['a'] == 'error'
        assert len(client.bulk_calls) == 1


def test_raise_on_errors_names_only_error_ids() -> None:
    from src.clients.occ_bulk import BulkWriteError, raise_on_errors

    raise_on_errors({'a': 'updated', 'b': 'conflict-exhausted', 'c': 'not-found'})
    with pytest.raises(BulkWriteError) as info:
        raise_on_errors({'a': 'updated', 'b': 'error', 'c': 'conflict-exhausted'})
    assert info.value.doc_ids == ['b']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])


class _AlwaysConflict(FakeOccOpenSearch):
    def __init__(
        self, *, transport_error: bool = False, error: OpenSearchException | None = None
    ) -> None:
        super().__init__({'a': {'class_id': 1, 'label_source': 'human'}})
        self.transport_error = transport_error
        self.error = error

    async def bulk(
        self,
        *,
        body: list[dict[str, Any]],
        refresh: bool | str = False,
    ) -> dict[str, Any]:
        self.bulk_calls.append(body)
        if self.transport_error:
            raise self.error or OsConnectionError('N/A', 'boom', Exception('x'))
        return make_bulk_response(
            [make_bulk_update_item('a', status=409, error_type='version_conflict_engine_exception')]
        )


async def _run_rounds(client: FakeOccOpenSearch, **kwargs: Any) -> list[tuple[Any, ...]]:
    seen: list[tuple[Any, ...]] = []
    await occ_bulk_rounds(
        client,
        index='op_prj_default__items',
        ids=['a'],
        prepare=lambda _id, doc: None if doc is None else {'class_id': 2},
        on_written=lambda i: seen.append(('written', i)),
        on_failed=lambda i, doc, why: seen.append(('failed', i, why, (doc or {})['_source'])),
        **kwargs,
    )
    return seen


class TestOccBulkRounds:
    @pytest.mark.asyncio
    async def test_conflict_is_retried_then_reported_once_with_the_last_doc(self) -> None:
        client = _AlwaysConflict()
        seen = await _run_rounds(client, max_retries=2)
        assert seen == [('failed', 'a', 'conflict', {'class_id': 1, 'label_source': 'human'})]
        assert len(client.bulk_calls) == 3

    @pytest.mark.asyncio
    async def test_transport_error_is_retried_by_default(self) -> None:
        client = _AlwaysConflict(transport_error=True)
        seen = await _run_rounds(client, max_retries=1)
        assert [s[:3] for s in seen] == [('failed', 'a', 'transport')]
        assert len(client.bulk_calls) == 2

    @pytest.mark.asyncio
    async def test_transport_error_fails_at_once_when_not_retried(self) -> None:
        client = _AlwaysConflict(transport_error=True)
        seen = await _run_rounds(client, max_retries=3, retry_transport=False)
        assert [s[:3] for s in seen] == [('failed', 'a', 'transport')]
        assert len(client.bulk_calls) == 1

    @pytest.mark.asyncio
    async def test_request_error_400_is_not_retried(self) -> None:
        client = _AlwaysConflict(
            transport_error=True, error=RequestError(400, 'mapper_parsing_exception', {})
        )
        seen = await _run_rounds(client, max_retries=3)
        assert [s[:3] for s in seen] == [('failed', 'a', 'transport')]
        assert len(client.bulk_calls) == 1


@pytest.mark.asyncio
async def test_relabel_surfaces_a_non_409_item_error_not_a_conflict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.clients.occ_bulk import BulkWriteError
    from src.services.curation import crop_bulk

    async def _no_excludes() -> list[str]:
        return VECTOR_EXCLUDES

    monkeypatch.setattr(crop_bulk, 'vector_source_excludes', _no_excludes)
    monkeypatch.setattr(crop_bulk, 'items_index', lambda: 'op_prj_default__items')
    client = FakeOccOpenSearch(
        {'a': {'class_id': 1}}, bulk_status={'a': (400, 'mapper_parsing_exception')}
    )
    with pytest.raises(BulkWriteError) as info:
        await crop_bulk.occ_bulk_human_relabel(
            client, ['a'], lambda _s: {'class_id': 2}, writer_id='t', max_retries=1
        )
    assert info.value.doc_ids == ['a']
