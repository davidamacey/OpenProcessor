"""The shared by-class aggregation behind ``GET /classes`` and ``GET /stats/classes``."""

from __future__ import annotations

from src.services.curation.class_counts import class_counts_body, parse_class_counts


def test_trainable_is_validated_minus_holdout_minus_excluded() -> None:
    counts = parse_class_counts(
        {
            'by_class': {
                'buckets': [
                    {
                        'key': 7,
                        'doc_count': 30,
                        'validated': {'doc_count': 20},
                        'validated_test_holdout': {'doc_count': 5},
                        'validated_excluded': {'doc_count': 2},
                    },
                    {'key': 9, 'doc_count': 4},
                ]
            }
        }
    )
    assert counts.count == {7: 30, 9: 4}
    assert counts.trainable(7) == 13
    assert counts.trainable(9) == 0
    assert counts.trainable(404) == 0


def test_parse_of_an_empty_answer_is_empty() -> None:
    assert parse_class_counts({}).count == {}


def test_body_scopes_by_class_and_only_adds_source_and_query_on_request() -> None:
    plain = class_counts_body()
    assert plain['size'] == 0
    assert 'query' not in plain
    per_class = plain['aggs']['by_class']['aggs']
    assert set(per_class) == {'validated', 'validated_test_holdout', 'validated_excluded'}

    rich = class_counts_body(query={'match_all': {}}, by_source=True)
    assert rich['query'] == {'match_all': {}}
    assert rich['aggs']['by_class']['aggs']['by_source'] == {
        'terms': {'field': 'label_source', 'size': 16}
    }
