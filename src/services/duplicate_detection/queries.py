"""OpenSearch request bodies and reply readers for near-duplicate detection.

Pure functions: :class:`~src.services.duplicate_detection.service.DuplicateDetectionService`
sends the bodies and hands the replies back here.
"""

from typing import Any

from src.services.duplicate_detection.models import DuplicateGroup, DuplicateMatch


GROUP_EXISTS_QUERY: dict[str, Any] = {'exists': {'field': 'duplicate_group_id'}}

_MATCH_SOURCE_FIELDS = ['image_id', 'image_path', 'duplicate_group_id']
_MEMBER_SOURCE_FIELDS = [
    'image_id',
    'image_path',
    'duplicate_group_id',
    'is_duplicate_primary',
    'duplicate_score',
    'width',
    'height',
    'indexed_at',
]


def knn_body(vector: list[float], k: int) -> dict[str, Any]:
    """k-NN search over the global embedding returning ``k`` hits."""
    return {
        'size': k,
        'query': {
            'knn': {
                'global_embedding': {
                    'vector': vector,
                    'k': k,
                }
            }
        },
        '_source': _MATCH_SOURCE_FIELDS,
    }


def matches_from_hits(
    hits: list[dict[str, Any]], threshold: float, exclude_image_id: str | None
) -> list[DuplicateMatch]:
    """The hits scoring at least ``threshold``, minus ``exclude_image_id``, in reply order."""
    duplicates = []
    for hit in hits:
        hit_id = hit['_id']
        if exclude_image_id and hit_id == exclude_image_id:
            continue

        similarity = hit['_score']
        if similarity < threshold:
            continue

        source = hit['_source']
        duplicates.append(
            DuplicateMatch(
                image_id=hit_id,
                image_path=source.get('image_path', ''),
                similarity=similarity,
                duplicate_group_id=source.get('duplicate_group_id'),
            )
        )
    return duplicates


def group_member_doc(group_id: str, *, primary: bool, score: float) -> dict[str, Any]:
    """The partial doc that puts an image into a duplicate group."""
    return {
        'doc': {
            'duplicate_group_id': group_id,
            'is_duplicate_primary': primary,
            'duplicate_score': score,
        }
    }


def groups_aggregation_body(min_size: int) -> dict[str, Any]:
    """Terms aggregation of every duplicate group with its primary image."""
    return {
        'size': 0,
        'query': GROUP_EXISTS_QUERY,
        'aggs': {
            'groups': {
                'terms': {
                    'field': 'duplicate_group_id',
                    'size': 10000,  # Get all groups
                    'min_doc_count': min_size,
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
    query = {'term': {'duplicate_group_id': group_id}}

    if not include_primary:
        query = {
            'bool': {
                'must': [query],
                'must_not': [{'term': {'is_duplicate_primary': True}}],
            }
        }

    return {
        'size': 1000,
        'query': query,
        '_source': _MEMBER_SOURCE_FIELDS,
        'sort': [
            {'is_duplicate_primary': {'order': 'desc'}},
            {'duplicate_score': {'order': 'desc'}},
        ],
    }


GROUPED_COUNT_BODY: dict[str, Any] = {'query': GROUP_EXISTS_QUERY}
GROUP_CARDINALITY_BODY: dict[str, Any] = {
    'size': 0,
    'query': GROUP_EXISTS_QUERY,
    'aggs': {'group_count': {'cardinality': {'field': 'duplicate_group_id'}}},
}
