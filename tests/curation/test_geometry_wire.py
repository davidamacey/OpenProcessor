"""What the wire says about an item's box: its revision, who set it, whether an automated writer
may touch it, and the served text for that. The browser shows these and decides nothing; the lock
comes from the same predicate the writers use."""

from __future__ import annotations

from typing import Any

import pytest

from curation.query_fakes import matches
from src.clients.occ_locks import is_locked_geometry
from src.services.curation.image_boxes import item_box_views
from src.services.curation.item_filter import ItemFilter, item_filter_clauses
from src.services.curation.wire import serialize_item
from src.services.embedding_spaces.readers import WriteFields


def _wire(src: dict[str, Any]) -> dict[str, Any]:
    return serialize_item({'crop_id': 'c1', **src}, state_fields={}, api_prefix='/p')


CASES: list[tuple[str, dict[str, Any], str, bool, str | None]] = [
    ('machine', {}, 'detector', False, None),
    (
        'moved',
        {'geometry_source': 'human', 'geometry_revision': 2},
        'human',
        True,
        'Set by a person',
    ),
    (
        'deleted',
        {'class_excluded': True, 'excluded_by': 'human', 'excluded_reason': 'not_an_object'},
        'human',
        True,
        'Deleted by a person',
    ),
    (
        'imported',
        {'class_source': 'external_label', 'class_validated': True, 'import_ids': ['i']},
        'import',
        True,
        'Set by an import',
    ),
    ('held_out', {'test_holdout': True}, 'detector', True, 'In the test holdout'),
]


@pytest.mark.parametrize(('name', 'src', 'source', 'locked', 'label'), CASES)
def test_the_wire_serves_the_geometry_facts(
    name: str, src: dict[str, Any], source: str, locked: bool, label: str | None
) -> None:
    wire = _wire(src)
    assert wire['geometry_source'] == source, name
    assert wire['geometry_locked'] is locked
    assert wire['lock_label'] == label
    assert wire['geometry_locked'] is is_locked_geometry(src)  # one predicate for wire and writers
    assert wire['geometry_revision'] == src.get('geometry_revision', 0)


def test_thumbnail_url_carries_the_revision() -> None:
    assert _wire({'geometry_revision': 2})['thumbnail_url'] == '/p/crops/c1/thumbnail?v=2'
    assert _wire({})['thumbnail_url'] == '/p/crops/c1/thumbnail?v=0'


def test_the_item_box_view_carries_the_lock_and_the_revision() -> None:
    moved = {'bbox_norm': [0.1, 0.1, 0.5, 0.5], 'geometry_source': 'human', 'geometry_revision': 3}
    (view,) = item_box_views(moved, 'c1')
    assert (view.locked, view.lock_label, view.geometry_revision) == (True, 'Set by a person', 3)
    (plain,) = item_box_views({'bbox_norm': [0.1, 0.1, 0.5, 0.5]}, 'c2')
    assert (plain.locked, plain.lock_label, plain.geometry_revision) == (False, None, 0)


def test_a_human_region_box_view_is_locked() -> None:
    from curation.reprocess_fixtures import F, box

    src = {
        'bbox_norm': [0.1, 0.1, 0.9, 0.9],
        F.boxes: [
            box('b1', state='accepted', source='human', detector='human', reason=None),
            box('b2', state='accepted', source='detector'),
        ],
    }
    regions = {v.box_id: v for v in item_box_views(src, 'c1') if v.kind == 'region'}
    assert (regions['b1'].locked, regions['b1'].lock_label) == (True, 'Set by a person')
    assert (regions['b2'].locked, regions['b2'].lock_label) == (False, None)


def test_origin_drawn_means_a_person_drew_it() -> None:
    clauses = item_filter_clauses(
        ItemFilter(origin=['drawn']), embedding=WriteFields('pe_embedding', 'embedding_state')
    )
    drawn = {'geometry_source': 'human'}
    moved = {'geometry_source': 'human', 'detector_class_name': 'car'}
    detected = {'detector_class_name': 'car'}

    def kept(doc: dict[str, Any]) -> bool:
        return all(matches(doc, c) for c in clauses)

    assert kept(drawn)
    assert not kept(moved)  # a moved detector item is not a drawn one
    assert not kept(detected)
