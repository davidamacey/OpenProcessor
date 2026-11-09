"""The served cluster is one nested object on the item wire and survives a box round trip."""

from __future__ import annotations

from src.clients.curation_opensearch.bodies_core import _items_body, _region_boxes_mapping
from src.clients.curation_opensearch.cluster_fields import CLUSTER_PARTITION_MAPPING
from src.domain.region_boxes import RegionBox
from src.services.curation.wire import cluster_object


def test_an_item_with_no_projected_partition_has_no_cluster_object() -> None:
    assert cluster_object({'class_name': 'x'}) is None


def test_the_cluster_object_nests_the_served_fields() -> None:
    src = {
        'cluster_partition_id': 'p1',
        'cluster_uid': 'u1',
        'cluster_state': 'review',
        'cluster_score': 0.7,
        'cluster_margin': 0.1,
        'cluster_assigned_by': 'incremental',
    }
    assert cluster_object(src) == {
        'uid': 'u1',
        'partition_id': 'p1',
        'state': 'review',
        'score': 0.7,
        'margin': 0.1,
        'assigned_by': 'incremental',
    }


def test_the_mapping_carries_the_same_fields_on_items_and_boxes() -> None:
    items = _items_body()['mappings']['properties']
    assert set(CLUSTER_PARTITION_MAPPING) <= set(items)
    element = _region_boxes_mapping()['region_boxes']['properties']
    assert set(CLUSTER_PARTITION_MAPPING) <= set(element)


def test_a_box_element_keeps_its_cluster_keys_through_a_human_edit() -> None:
    doc = {
        'box_id': 'b1',
        'bbox_norm': [0.1, 0.1, 0.2, 0.2],
        'state': 'accepted',
        'cluster_partition_id': 'p1',
        'cluster_uid': 'u1',
        'cluster_state': 'assigned',
    }
    assert RegionBox.from_doc(doc).to_doc()['cluster_uid'] == 'u1'
