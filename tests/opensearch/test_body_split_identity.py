"""The cluster and near-duplicate field groups live outside ``bodies_core.py`` (2F-8): the items
body carries exactly the fields the groups hold, spec for spec, and the images body the near-duplicate ones.

The expected values are literals copied from the body before the move.
"""

from __future__ import annotations

from pathlib import Path

from src.clients.curation_opensearch.bodies_core import _images_body, _items_body
from src.clients.curation_opensearch.cluster_fields import CLUSTER_ITEM_MAPPING
from src.clients.curation_opensearch.near_dup_fields import NEAR_DUP_MAPPING


CLUSTER_BEFORE = {
    'cluster_partition_id': {'type': 'keyword'},
    'cluster_uid': {'type': 'keyword'},
    'cluster_state': {'type': 'keyword'},
    'cluster_score': {'type': 'float'},
    'cluster_margin': {'type': 'float'},
    'cluster_assigned_by': {'type': 'keyword'},
    'cluster_distance': {'type': 'float'},
    'cluster_auto_suggest': {'type': 'keyword'},
}
NEAR_DUP_FIELDS = {
    'near_dup_group_id': {'type': 'keyword'},
    'near_dup_leak_group_id': {'type': 'keyword'},
    'near_dup_is_representative': {'type': 'boolean'},
    'near_dup_group_size': {'type': 'integer'},
    'near_dup_run_id': {'type': 'keyword'},
}


def test_the_cluster_module_holds_the_cluster_fields_unchanged() -> None:
    assert CLUSTER_ITEM_MAPPING == CLUSTER_BEFORE


def test_the_near_duplicate_module_holds_the_projection_fields() -> None:
    assert NEAR_DUP_MAPPING == NEAR_DUP_FIELDS


def test_the_items_and_images_bodies_carry_the_groups_field_for_field() -> None:
    items = _items_body()['mappings']['properties']
    images = _images_body()['mappings']['properties']

    for field, spec in {**CLUSTER_BEFORE, **NEAR_DUP_FIELDS}.items():
        assert items[field] == spec
    for field, spec in NEAR_DUP_FIELDS.items():
        assert images[field] == spec


def test_no_other_module_defines_those_fields_again() -> None:
    from src.clients.curation_opensearch import bodies_core

    source = Path(bodies_core.__file__).read_text(encoding='utf-8')

    for field in ('near_dup_group_id', 'cluster_auto_suggest'):
        assert field not in source
