"""DuplicateDetectionService: the exact OpenSearch requests it sends and how it reads the replies.

The group fields are the ``near_dup_*`` projection of the current core run."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from src.services.duplicate_detection.service import DuplicateDetectionService


pytestmark = pytest.mark.asyncio


GLOBAL_INDEX = 'visual_search_global'


class _FakeClient:
    """Records every call and answers from canned replies."""

    def __init__(self, *, search: Any = None, count: Any = None) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self._search, self._count = search, count

    async def search(self, **kwargs: Any) -> Any:
        self.calls.append(('search', kwargs))
        return self._search

    async def count(self, **kwargs: Any) -> Any:
        self.calls.append(('count', kwargs))
        return self._count.pop(0)


def _service(client: _FakeClient) -> DuplicateDetectionService:
    return DuplicateDetectionService(SimpleNamespace(client=client))  # type: ignore[arg-type]


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
                    'query': {'exists': {'field': 'near_dup_group_id'}},
                    'aggs': {
                        'groups': {
                            'terms': {
                                'field': 'near_dup_group_id',
                                'size': 10000,
                                'min_doc_count': 2,
                            },
                            'aggs': {
                                'primary': {
                                    'filter': {'term': {'near_dup_is_representative': True}},
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
    'near_dup_group_id',
    'near_dup_is_representative',
    'near_dup_cosine_to_rep',
    'width',
    'height',
    'indexed_at',
]


@pytest.mark.parametrize('include_primary', [True, False])
async def test_get_group_members_query_and_sort(include_primary: bool) -> None:
    client = _FakeClient(search={'hits': {'hits': [{'_source': {'image_id': 'x'}}]}})
    members = await _service(client).get_group_members('g1', include_primary=include_primary)

    term = {'term': {'near_dup_group_id': 'g1'}}
    query = (
        term
        if include_primary
        else {
            'bool': {'must': [term], 'must_not': [{'term': {'near_dup_is_representative': True}}]}
        }
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
                        {'near_dup_is_representative': {'order': 'desc'}},
                        {'near_dup_cosine_to_rep': {'order': 'desc'}},
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
            {'index': GLOBAL_INDEX, 'body': {'query': {'exists': {'field': 'near_dup_group_id'}}}},
        ),
        (
            'search',
            {
                'index': GLOBAL_INDEX,
                'body': {
                    'size': 0,
                    'query': {'exists': {'field': 'near_dup_group_id'}},
                    'aggs': {'group_count': {'cardinality': {'field': 'near_dup_group_id'}}},
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
