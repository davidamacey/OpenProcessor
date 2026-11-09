"""OpenSearch request bodies and reply readers for the core duplicate group queries.

Pure functions: :class:`~src.services.duplicate_detection.service.DuplicateDetectionService`
sends the bodies and hands the replies back here. The group fields are the projection of the
current core run (``near_dup_core_group``): ``near_dup_group_id``, ``near_dup_is_representative``
and ``near_dup_cosine_to_rep``.
"""

from typing import Any

from src.services.duplicate_detection.models import DuplicateGroup


GROUP_EXISTS_QUERY: dict[str, Any] = {'exists': {'field': 'near_dup_group_id'}}

_MEMBER_SOURCE_FIELDS = [
    'image_id',
    'image_path',
    'near_dup_group_id',
    'near_dup_is_representative',
    'near_dup_cosine_to_rep',
    'width',
    'height',
    'indexed_at',
]


def groups_aggregation_body(min_size: int) -> dict[str, Any]:
    """Terms aggregation of every duplicate group with its primary image."""
    return {
        'size': 0,
        'query': GROUP_EXISTS_QUERY,
        'aggs': {
            'groups': {
                'terms': {
                    'field': 'near_dup_group_id',
                    'size': 10000,  # Get all groups
                    'min_doc_count': min_size,
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
    }


def group_from_bucket(bucket: dict[str, Any]) -> DuplicateGroup:
    """One aggregation bucket as a group whose members load lazily."""
    primary_hits = bucket['primary']['doc']['hits']['hits']
    if primary_hits:
        primary = primary_hits[0]['_source']
        primary_id = primary.get('image_id', '')
        primary_path = primary.get('image_path', '')
    else:
        primary_id = ''
        primary_path = ''

    return DuplicateGroup(
        group_id=bucket['key'],
        primary_image_id=primary_id,
        primary_image_path=primary_path,
        member_count=bucket['doc_count'],
        members=[],  # Lazy load members
    )


def group_members_body(group_id: str, include_primary: bool) -> dict[str, Any]:
    """Members of one group, the primary first, then by descending score."""
    query: dict[str, Any] = {'term': {'near_dup_group_id': group_id}}

    if not include_primary:
        query = {
            'bool': {
                'must': [query],
                'must_not': [{'term': {'near_dup_is_representative': True}}],
            }
        }

    return {
        'size': 1000,
        'query': query,
        '_source': _MEMBER_SOURCE_FIELDS,
        'sort': [
            {'near_dup_is_representative': {'order': 'desc'}},
            {'near_dup_cosine_to_rep': {'order': 'desc'}},
        ],
    }


GROUPED_COUNT_BODY: dict[str, Any] = {'query': GROUP_EXISTS_QUERY}
GROUP_CARDINALITY_BODY: dict[str, Any] = {
    'size': 0,
    'query': GROUP_EXISTS_QUERY,
    'aggs': {'group_count': {'cardinality': {'field': 'near_dup_group_id'}}},
}
