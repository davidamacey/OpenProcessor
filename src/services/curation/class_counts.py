"""Per-class item counts from one ``size: 0`` aggregation, shared by ``GET /classes`` and
``GET /stats/classes`` so both serve the same numbers.

Trainable = validated minus the frozen test holdout minus human-excluded items: a frozen holdout
item must never leak into training and an excluded item was deliberately removed from the dataset.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.config.curation_accessors import items_index


def _validated_and(flag: str) -> dict[str, Any]:
    return {
        'filter': {
            'bool': {'filter': [{'term': {'class_validated': True}}, {'term': {flag: True}}]}
        }
    }


def class_counts_body(
    *, query: dict[str, Any] | None = None, by_source: bool = False
) -> dict[str, Any]:
    """The by-class aggregation body. ``by_source`` adds the per-class ``label_source`` terms
    (mapped keyword directly on the live index, no ``.keyword`` subfield)."""
    per_class: dict[str, Any] = {
        'validated': {'filter': {'term': {'class_validated': True}}},
        'validated_test_holdout': _validated_and('test_holdout'),
        'validated_excluded': _validated_and('class_excluded'),
    }
    if by_source:
        per_class['by_source'] = {'terms': {'field': 'label_source', 'size': 16}}
    body: dict[str, Any] = {
        'size': 0,
        'aggs': {'by_class': {'terms': {'field': 'class_id', 'size': 1000}, 'aggs': per_class}},
    }
    if query is not None:
        body['query'] = query
    return body


@dataclass(frozen=True)
class ClassCounts:
    """Item counts keyed by ``class_id``; a class without items reads 0."""

    count: dict[int, int] = field(default_factory=dict)
    validated: dict[int, int] = field(default_factory=dict)
    validated_test_holdout: dict[int, int] = field(default_factory=dict)
    validated_excluded: dict[int, int] = field(default_factory=dict)

    def trainable(self, class_id: int) -> int:
        return (
            self.validated.get(class_id, 0)
            - self.validated_test_holdout.get(class_id, 0)
            - self.validated_excluded.get(class_id, 0)
        )


def parse_class_counts(aggs: dict[str, Any]) -> ClassCounts:
    counts = ClassCounts()
    for bucket in (aggs.get('by_class') or {}).get('buckets', []):
        cid = int(bucket['key'])
        counts.count[cid] = int(bucket.get('doc_count', 0))
        for name, target in (
            ('validated', counts.validated),
            ('validated_test_holdout', counts.validated_test_holdout),
            ('validated_excluded', counts.validated_excluded),
        ):
            target[cid] = int((bucket.get(name) or {}).get('doc_count', 0))
    return counts


async def fetch_class_counts(
    opensearch: Any, *, query: dict[str, Any] | None = None, by_source: bool = False
) -> tuple[ClassCounts, dict[str, Any]]:
    """The counts and the raw ``aggregations`` they were read from."""
    resp = await opensearch.search(
        index=items_index(), body=class_counts_body(query=query, by_source=by_source)
    )
    aggs = resp.get('aggregations') or {}
    return parse_class_counts(aggs), aggs
