"""The OpenSearch re-index adapter against a fake client that records every request body."""

from __future__ import annotations

import datetime as dt
import uuid
from types import SimpleNamespace
from typing import Any

import pytest
from opensearchpy.exceptions import ConnectionError as OsConnectionError

from src.db.repos import embedding_spaces as repo
from src.domain.embedding_spaces import KnnMethodSpec, Role, Status, Target
from src.services.embedding_spaces import reindex_opensearch as mod, views
from src.services.embedding_spaces.reindex import DocRef, ReindexMode, VectorWrite


pytestmark = pytest.mark.asyncio


def _space(target: Target) -> repo.SpaceRow:
    box = target is Target.BOX
    return repo.SpaceRow(
        id=uuid.uuid4(),
        project_id=uuid.uuid4(),
        space_key='k',
        model_version_id=uuid.uuid4(),
        target=target,
        dim=3,
        role=Role.CANDIDATE,
        status=Status.BUILDING,
        index_role='images' if target is Target.FRAME else 'items',
        vector_field='boxvec_k.embedding' if box else 'vec_k',
        state_field=None if box or target is Target.FRAME else 'vec_k_state',
        nested_path='boxvec_k' if box else None,
        knn_method=KnnMethodSpec(),
        revision=1,
        built_by_job_id=None,
        ready_at=None,
        retired_at=None,
        created_at=dt.datetime.now(dt.UTC),
        created_by='t',
    )


class FakeOS:
    def __init__(self, hits: list[dict[str, Any]] | None = None) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.hits = hits or []
        self.bulk_items: list[dict[str, Any]] = []

    async def create_pit(self, **kw: Any) -> dict[str, str]:
        self.calls.append(('create_pit', kw))
        return {'pit_id': 'P1'}

    async def delete_pit(self, **kw: Any) -> None:
        self.calls.append(('delete_pit', kw))

    async def search(self, **kw: Any) -> dict[str, Any]:
        self.calls.append(('search', kw))
        return {'hits': {'hits': self.hits}}

    async def count(self, **kw: Any) -> dict[str, int]:
        self.calls.append(('count', kw))
        return {'count': 7}

    async def bulk(self, **kw: Any) -> dict[str, Any]:
        self.calls.append(('bulk', kw))
        return {'items': self.bulk_items}

    async def mget(self, **kw: Any) -> dict[str, Any]:
        self.calls.append(('mget', kw))
        return {'docs': self.hits}

    async def update_by_query(self, **kw: Any) -> dict[str, Any]:
        self.calls.append(('update_by_query', kw))
        return {}


@pytest.fixture(autouse=True)
def _cfg(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        mod, 'get_curation_config', lambda: SimpleNamespace(items_index='it', images_index='im')
    )


def _last(fake: FakeOS, name: str) -> dict[str, Any]:
    return next(kw for n, kw in reversed(fake.calls) if n == name)


async def test_page_uses_pit_and_search_after_on_id() -> None:
    fake = FakeOS([{'_id': 'a', '_source': {'bbox_norm': [0.1, 0.2, 0.3, 0.4]}}])
    refs = await mod.OpenSearchReindexIndex(fake).page(
        _space(Target.ITEM), ReindexMode.BUILD, {}, 'prev', 10
    )
    body = _last(fake, 'search')['body']
    assert body['pit'] == {'id': 'P1', 'keep_alive': mod.PIT_KEEP_ALIVE}
    assert body['sort'] == [{'_id': 'asc'}]
    assert body['search_after'] == ['prev']
    assert 'index' not in _last(fake, 'search')
    assert _last(fake, 'delete_pit')['body'] == {'pit_id': ['P1']}
    assert refs == [DocRef('a', (0.1, 0.2, 0.3, 0.4))]


async def test_backfill_page_excludes_documents_with_a_vector() -> None:
    fake = FakeOS()
    await mod.OpenSearchReindexIndex(fake).page(
        _space(Target.ITEM), ReindexMode.BACKFILL, {}, None, 5
    )
    body = _last(fake, 'search')['body']
    assert body['query']['bool']['must_not'] == [{'exists': {'field': 'vec_k'}}]
    assert 'search_after' not in body


async def test_box_page_expands_embeddable_boxes_only() -> None:
    boxes = [
        {'box_id': 'b1', 'bbox_norm': [0, 0, 0.5, 0.5], 'state': 'accepted'},
        {'box_id': 'b2', 'bbox_norm': [0, 0, 0.1, 0.1], 'state': 'rejected'},
    ]
    fake = FakeOS([{'_id': 'i1', '_source': {'region_boxes': boxes}}])
    refs = await mod.OpenSearchReindexIndex(fake).page(
        _space(Target.BOX), ReindexMode.BUILD, {}, None, 5
    )
    assert [r.doc_id for r in refs] == ['i1#b1']


async def test_unsupported_scope_is_refused() -> None:
    with pytest.raises(ValueError, match='not supported'):
        await mod.OpenSearchReindexIndex(FakeOS()).count(
            _space(Target.ITEM), ReindexMode.BUILD, {'x': 1}
        )


async def test_write_sends_bbox_guard_and_maps_outcomes() -> None:
    fake = FakeOS()
    fake.bulk_items = [
        {'update': {'status': 200, 'result': 'updated'}},
        {'update': {'status': 200, 'result': 'noop'}},
        {'update': {'status': 404}},
    ]
    out = await mod.OpenSearchReindexIndex(fake).write(
        _space(Target.ITEM),
        [
            VectorWrite('a', [1.0, 0.0], (0.1, 0.2)),
            VectorWrite('b', [1.0, 0.0], (0.1, 0.2)),
            VectorWrite('c', [1.0, 0.0], None),
        ],
    )
    assert out == {'a': 'written', 'b': 'moved', 'c': 'missing'}
    body = _last(fake, 'bulk')['body']
    assert body[0]['update'] == {'_index': 'it', '_id': 'a', 'retry_on_conflict': 3}
    params = body[1]['script']['params']
    assert params['bbox'] == [0.1, 0.2]
    assert params['field'] == 'vec_k'
    assert params['state_field'] == 'vec_k_state'


async def test_box_write_targets_the_item_and_the_box() -> None:
    fake = FakeOS()
    fake.bulk_items = [{'update': {'status': 200, 'result': 'updated'}}]
    await mod.OpenSearchReindexIndex(fake).write(
        _space(Target.BOX), [VectorWrite('i1#b1', [1.0], (0.0, 0.0, 0.5, 0.5))]
    )
    body = _last(fake, 'bulk')['body']
    assert body[0]['update']['_id'] == 'i1'
    params = body[1]['script']['params']
    assert (params['box_id'], params['path'], params['leaf']) == ('b1', 'boxvec_k', 'embedding')


async def test_failed_bulk_item_raises() -> None:
    fake = FakeOS()
    fake.bulk_items = [{'update': {'status': 500, 'error': 'boom'}}]
    with pytest.raises(RuntimeError, match='re-index write failed'):
        await mod.OpenSearchReindexIndex(fake).write(_space(Target.ITEM), [VectorWrite('a', [1.0])])


async def test_exists_count_and_clear() -> None:
    fake = FakeOS()
    index = mod.OpenSearchReindexIndex(fake)
    assert await index.count_with_vector(_space(Target.FRAME)) == 7
    assert _last(fake, 'count')['index'] == 'im'
    await index.clear(_space(Target.BOX))
    q = _last(fake, 'update_by_query')
    assert q['index'] == 'it'
    assert q['body']['script']['params']['field'] == 'boxvec_k'
    assert q['body']['query']['nested']['path'] == 'boxvec_k'


async def test_sample_is_random_and_box_sample_reads_inner_hit() -> None:
    fake = FakeOS(
        [
            {
                '_id': 'i1',
                'inner_hits': {
                    'boxvec_k': {
                        'hits': {'hits': [{'_source': {'box_id': 'b1', 'bbox_norm': [0, 0, 1, 1]}}]}
                    }
                },
            }
        ]
    )
    refs = await mod.OpenSearchReindexIndex(fake).sample(_space(Target.BOX), 3)
    assert 'random_score' in _last(fake, 'search')['body']['query']['function_score']
    assert refs == [DocRef('i1#b1', (0.0, 0.0, 1.0, 1.0))]


async def test_read_vectors_joins_box_ids() -> None:
    fake = FakeOS(
        [
            {
                '_id': 'i1',
                'found': True,
                '_source': {'boxvec_k': [{'box_id': 'b1', 'embedding': [1.0]}]},
            }
        ]
    )
    got = await mod.OpenSearchReindexIndex(fake).read_vectors(_space(Target.BOX), ['i1#b1'])
    assert got == {'i1#b1': [1.0]}


async def test_docs_with_vector_is_none_when_the_index_cannot_answer() -> None:
    class Down:
        async def count_with_vector(self, _space: Any) -> int:
            raise OsConnectionError('N/A', 'down', Exception('down'))

    assert await views.docs_with_vector(Down(), _space(Target.ITEM)) is None  # type: ignore[arg-type]
    assert (
        await views.docs_with_vector(mod.OpenSearchReindexIndex(FakeOS()), _space(Target.ITEM)) == 7
    )
