"""Moving the cluster and near-duplicate field groups out of ``bodies_core.py`` changed nothing
(2F-8): the items body carries exactly the fields the groups held, spec for spec.

The expected values are literals copied from the body before the move.
"""

from __future__ import annotations

from pathlib import Path

from src.clients.curation_opensearch.bodies_core import _items_body
from src.clients.curation_opensearch.cluster_fields import CLUSTER_ITEM_MAPPING
from src.clients.curation_opensearch.near_dup_fields import NEAR_DUP_ITEM_MAPPING


CLUSTER_BEFORE = {
    'cluster_id': {'type': 'integer'},
    'cluster_distance': {'type': 'float'},
    'cluster_subid': {'type': 'keyword'},
    'cluster_distance_cluster_id': {'type': 'integer'},
    'cluster_nearest_id': {'type': 'integer'},
    'cluster_auto_suggest': {'type': 'keyword'},
}
NEAR_DUP_BEFORE = {
    'dup_group_id': {'type': 'keyword'},
    'dup_group_size': {'type': 'integer'},
    'dup_is_representative': {'type': 'boolean'},
    'dup_threshold': {'type': 'float'},
    'dup_method': {'type': 'keyword'},
    'dup_scored_at': {'type': 'date'},
}


def test_the_cluster_module_holds_the_cluster_fields_unchanged() -> None:
    assert CLUSTER_ITEM_MAPPING == CLUSTER_BEFORE


def test_the_near_duplicate_module_holds_the_dup_fields_unchanged() -> None:
    assert NEAR_DUP_ITEM_MAPPING == NEAR_DUP_BEFORE


def test_the_items_body_carries_both_groups_field_for_field() -> None:
    properties = _items_body()['mappings']['properties']

    for field, spec in {**CLUSTER_BEFORE, **NEAR_DUP_BEFORE}.items():
        assert properties[field] == spec


def test_no_other_module_defines_those_fields_again() -> None:
    from src.clients.curation_opensearch import bodies_core

    source = Path(bodies_core.__file__).read_text(encoding='utf-8')

    for field in ('dup_group_id', 'dup_scored_at', 'cluster_nearest_id', 'cluster_auto_suggest'):
        assert field not in source
