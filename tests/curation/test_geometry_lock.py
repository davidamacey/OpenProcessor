"""The geometry lock: a box a person drew, moved or deleted is never moved, deleted or
re-created by an automated writer (re-detect, open-vocabulary run, dataset-import
reconcile, re-ingest). The predicate is :func:`src.clients.occ_locks.is_locked_geometry`;
every writer reaches it through ``is_locked_item``, so no writer can forget it."""

from __future__ import annotations

from typing import Any

import pytest

from src.clients.occ_locks import (
    is_human_owned_item,
    is_locked_geometry,
    is_locked_item,
    lock_includes,
)


MACHINE: dict[str, Any] = {'class_source': 'unlabeled_proposal', 'class_validated': False}
DELETED: dict[str, Any] = {
    'class_excluded': True,
    'excluded_by': 'human',
    'excluded_reason': 'not_an_object',
}

CASES: dict[str, tuple[dict[str, Any], bool]] = {
    'detector_box': ({**MACHINE, 'geometry_source': 'detector'}, False),
    'never_edited': (dict(MACHINE), False),
    'drawn_or_moved_by_a_person': ({**MACHINE, 'geometry_source': 'human'}, True),
    'human_deleted_as_not_an_object': ({**MACHINE, **DELETED}, True),
    'human_ignored_for_another_reason': (
        {**MACHINE, **DELETED, 'excluded_reason': 'blurry'},
        False,
    ),
    'machine_excluded_as_not_an_object': ({**MACHINE, **DELETED, 'excluded_by': 'vlm'}, False),
    'held_out': ({**MACHINE, 'test_holdout': True}, True),
    'validated_import': (
        {'class_source': 'external_label', 'class_validated': True, 'import_ids': ['i1']},
        True,
    ),
    'suggestion_import': (
        {'class_source': 'external_label', 'class_validated': False, 'import_ids': ['i1']},
        False,
    ),
    'import_proposal': ({**MACHINE, 'import_ids': ['i1'], 'proposed_by_import': 'i1'}, False),
}


@pytest.mark.parametrize('name', sorted(CASES))
def test_predicate_cases(name: str) -> None:
    doc, locked = CASES[name]
    assert is_locked_geometry(doc) is locked, name


@pytest.mark.parametrize('name', sorted(CASES))
def test_every_writer_that_asks_the_item_lock_gets_the_geometry_lock(name: str) -> None:
    doc, locked = CASES[name]
    if locked:
        assert is_locked_item(doc), name


@pytest.mark.parametrize('name', sorted(CASES))
def test_lock_includes_carries_every_field_the_predicates_read(name: str) -> None:
    """A query that fetches only ``lock_includes()`` must reach the same verdicts,
    or a writer that passes it would silently skip the lock."""
    doc, _ = CASES[name]
    wanted = set(lock_includes())
    fetched = {k: v for k, v in doc.items() if k in wanted}
    assert is_locked_geometry(fetched) is is_locked_geometry(doc)
    assert is_locked_item(fetched) is is_locked_item(doc)
    assert is_human_owned_item(fetched) is is_human_owned_item(doc)


@pytest.mark.parametrize(
    ('doc', 'owned'),
    [
        ({**MACHINE, 'geometry_source': 'human'}, True),
        ({**MACHINE, **DELETED}, True),
        ({**MACHINE, 'geometry_source': 'detector'}, False),
        ({'class_source': 'external_label', 'class_validated': True, 'import_ids': ['i1']}, False),
    ],
)
def test_a_person_owning_the_geometry_keeps_an_import_from_reconciling_it_away(
    doc: dict[str, Any], owned: bool
) -> None:
    assert is_human_owned_item(doc) is owned
