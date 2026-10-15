"""DuplicateDetectionService: the exact OpenSearch requests it sends and how it reads the replies."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest import mock

import numpy as np
import pytest

from src.services.duplicate_detection.service import DuplicateDetectionService


pytestmark = pytest.mark.asyncio


GLOBAL_INDEX = 'visual_search_global'


class _FakeClient:
    """Records every call and answers from canned replies."""

    def __init__(self, *, get: Any = None, search: Any = None, count: Any = None) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self._get, self._search, self._count = get, search, count

    async def get(self, **kwargs: Any) -> Any:
        self.calls.append(('get', kwargs))
        if isinstance(self._get, Exception):
            raise self._get
        return self._get

    async def search(self, **kwargs: Any) -> Any:
        self.calls.append(('search', kwargs))
        if isinstance(self._search, list):
            return self._search.pop(0)
        return self._search

    async def update(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(('update', kwargs))
        return {}

    async def count(self, **kwargs: Any) -> Any:
        self.calls.append(('count', kwargs))
        return self._count.pop(0)


def _service(client: _FakeClient) -> DuplicateDetectionService:
    return DuplicateDetectionService(SimpleNamespace(client=client))  # type: ignore[arg-type]


def _hit(hit_id: str, score: float, path: str = 'p', group: str | None = None) -> dict[str, Any]:
    source: dict[str, Any] = {'image_id': hit_id}
    if path is not None:
        source['image_path'] = path
    if group is not None:
        source['duplicate_group_id'] = group
    return {'_id': hit_id, '_score': score, '_source': source}


def _knn(vector: list[float], k: int, size: int) -> dict[str, Any]:
    return {
        'size': size,
        'query': {'knn': {'global_embedding': {'vector': vector, 'k': k}}},
        '_source': ['image_id', 'image_path', 'duplicate_group_id'],
    }


async def test_find_duplicates_sends_knn_and_drops_self_and_low_scores() -> None:
    client = _FakeClient(
        get={'_source': {'global_embedding': [0.1, 0.2], 'image_path': 'a'}},
        search={
            'hits': {
                'hits': [
                    _hit('a', 1.0),
                    _hit('b', 0.995, 'pb', 'g1'),
                    _hit('c', 0.99),
                    _hit('d', 0.98),
                ]
            }
        },
    )
    found = await _service(client).find_duplicates('a', threshold=0.99, max_results=3)

    assert client.calls == [
        (
            'get',
            {
                'index': GLOBAL_INDEX,
                'id': 'a',
                '_source': ['global_embedding', 'image_path'],
            },
        ),
        ('search', {'index': GLOBAL_INDEX, 'body': _knn([0.1, 0.2], 4, 4)}),
    ]
    assert [(m.image_id, m.image_path, m.similarity, m.duplicate_group_id) for m in found] == [
        ('b', 'pb', 0.995, 'g1'),
        ('c', 'p', 0.99, None),
    ]


async def test_find_duplicates_returns_empty_when_the_image_is_missing() -> None:
    client = _FakeClient(get=RuntimeError('not found'))
    assert await _service(client).find_duplicates('zzz') == []
    assert [name for name, _ in client.calls] == ['get']


@pytest.mark.parametrize('as_array', [True, False])
async def test_find_duplicates_by_embedding_converts_arrays_and_excludes_one_id(
    as_array: bool,
) -> None:
    client = _FakeClient(
        search={'hits': {'hits': [_hit('a', 0.999), _hit('b', 0.5), _hit('c', 0.991)]}}
    )
    vector: Any = np.array([0.5, 0.25]) if as_array else [0.5, 0.25]
    found = await _service(client).find_duplicates_by_embedding(
        vector, threshold=0.9, max_results=7, exclude_image_id='a'
    )

    assert client.calls == [('search', {'index': GLOBAL_INDEX, 'body': _knn([0.5, 0.25], 7, 7)})]
    assert [m.image_id for m in found] == ['c']


async def test_find_duplicates_by_embedding_keeps_every_id_without_an_exclusion() -> None:
    client = _FakeClient(search={'hits': {'hits': [_hit('a', 0.999)]}})
    found = await _service(client).find_duplicates_by_embedding([1.0])
    assert [m.image_id for m in found] == ['a']
    assert client.calls[0][1]['body']['size'] == 50


async def test_create_duplicate_group_updates_the_primary_then_each_duplicate() -> None:
    client = _FakeClient()
    with mock.patch('src.services.duplicate_detection.service.uuid.uuid4') as uuid4:
        uuid4.return_value = SimpleNamespace(hex='0123456789abcdef0123')
        group_id = await _service(client).create_duplicate_group('p', ['d1', 'd2'], [0.97, 0.95])

    assert group_id == 'dup_0123456789ab'
    assert client.calls == [
        (
            'update',
            {
                'index': GLOBAL_INDEX,
                'id': 'p',
                'body': {
                    'doc': {
                        'duplicate_group_id': group_id,
                        'is_duplicate_primary': True,
                        'duplicate_score': 1.0,
                    }
                },
            },
        ),
        *(
            (
                'update',
                {
                    'index': GLOBAL_INDEX,
                    'id': dup,
                    'body': {
                        'doc': {
                            'duplicate_group_id': group_id,
                            'is_duplicate_primary': False,
                            'duplicate_score': score,
                        }
                    },
                },
            )
            for dup, score in (('d1', 0.97), ('d2', 0.95))
        ),
    ]


def _bucket(key: str, count: int, primary: dict[str, Any] | None) -> dict[str, Any]:
    hits = [] if primary is None else [{'_source': primary}]
    return {'key': key, 'doc_count': count, 'primary': {'doc': {'hits': {'hits': hits}}}}


async def test_get_duplicate_groups_aggregates_and_paginates() -> None:
    buckets = [
        _bucket('g1', 3, {'image_id': 'i1', 'image_path': 'p1'}),
        _bucket('g2', 2, None),
        _bucket('g3', 2, {'image_id': 'i3'}),
    ]
    client = _FakeClient(search={'aggregations': {'groups': {'buckets': buckets}}})
    groups = await _service(client).get_duplicate_groups(min_size=2, page=1, size=2)

    assert client.calls == [
        (
            'search',
            {
                'index': GLOBAL_INDEX,
                'body': {
                    'size': 0,
                    'query': {'exists': {'field': 'duplicate_group_id'}},
                    'aggs': {
                        'groups': {
                            'terms': {
                                'field': 'duplicate_group_id',
                                'size': 10000,
                                'min_doc_count': 2,
                            },
                            'aggs': {
                                'primary': {
                                    'filter': {'term': {'is_duplicate_primary': True}},
                                    'aggs': {
                                        'doc': {
                                            'top_hits': {
                                                'size': 1,
                                                '_source': ['image_id', 'image_path'],
                                            }
                                        }
                                    },
                                }
                            },
                        }
                    },
                },
            },
        )
    ]
    assert [
        (g.group_id, g.primary_image_id, g.primary_image_path, g.member_count) for g in groups
    ] == [('g3', 'i3', '', 2)]

    first = await _service(
        _FakeClient(search={'aggregations': {'groups': {'buckets': buckets}}})
    ).get_duplicate_groups(size=2)
    assert [(g.group_id, g.primary_image_id, g.members) for g in first] == [
        ('g1', 'i1', []),
        ('g2', '', []),
    ]


_MEMBER_FIELDS = [
    'image_id',
    'image_path',
    'duplicate_group_id',
    'is_duplicate_primary',
    'duplicate_score',
    'width',
    'height',
    'indexed_at',
]


@pytest.mark.parametrize('include_primary', [True, False])
async def test_get_group_members_query_and_sort(include_primary: bool) -> None:
    client = _FakeClient(search={'hits': {'hits': [{'_source': {'image_id': 'x'}}]}})
    members = await _service(client).get_group_members('g1', include_primary=include_primary)

    term = {'term': {'duplicate_group_id': 'g1'}}
    query = (
        term
        if include_primary
        else {'bool': {'must': [term], 'must_not': [{'term': {'is_duplicate_primary': True}}]}}
    )
    assert client.calls == [
        (
            'search',
            {
                'index': GLOBAL_INDEX,
                'body': {
                    'size': 1000,
                    'query': query,
                    '_source': _MEMBER_FIELDS,
                    'sort': [
                        {'is_duplicate_primary': {'order': 'desc'}},
                        {'duplicate_score': {'order': 'desc'}},
                    ],
                },
            },
        )
    ]
    assert members == [{'image_id': 'x'}]


@pytest.mark.parametrize(('total', 'grouped', 'groups', 'average'), [(10, 4, 2, 2.0), (5, 0, 0, 0)])
async def test_get_stats_counts_and_averages(
    total: int, grouped: int, groups: int, average: float
) -> None:
    client = _FakeClient(
        count=[{'count': total}, {'count': grouped}],
        search={'aggregations': {'group_count': {'value': groups}}},
    )
    stats = await _service(client).get_stats()

    assert client.calls == [
        ('count', {'index': GLOBAL_INDEX}),
        (
            'count',
            {'index': GLOBAL_INDEX, 'body': {'query': {'exists': {'field': 'duplicate_group_id'}}}},
        ),
        (
            'search',
            {
                'index': GLOBAL_INDEX,
                'body': {
                    'size': 0,
                    'query': {'exists': {'field': 'duplicate_group_id'}},
                    'aggs': {'group_count': {'cardinality': {'field': 'duplicate_group_id'}}},
                },
            },
        ),
    ]
    assert stats == {
        'total_images': total,
        'grouped_images': grouped,
        'ungrouped_images': total - grouped,
        'duplicate_groups': groups,
        'average_group_size': average,
    }
