"""The pure helpers of the label-operation probe (``scripts/bench/label_op_probe.py``)."""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts' / 'bench'))

from label_op_probe import audit_state, expected_histogram, rss_mb, state_of


def _item(class_name: str | None = None, **extra: object) -> dict[str, object]:
    return state_of({'class_name': class_name, **extra})


def test_audit_finds_lost_extra_and_wrong_edits() -> None:
    before = {'a': _item('x'), 'b': _item('y'), 'c': _item('z'), 'd': _item('w')}
    after = {
        'a': _item('x'),  # restored
        'b': _item('probe_a', class_validated=True),  # written and not restored
        'c': _item('probe_b', class_validated=True, class_id=7),  # edited, keeps its edit
        'd': _item('w'),
    }
    edit = {'class_name': 'probe_b', 'class_id': 7, 'class_validated': True}
    report = audit_state(before, after, edited={'c'}, edit_state=edit)
    assert report['changed_or_not_restored'] == 1
    assert report['examples'] == ['b']
    assert report['wrong_edit'] == 0
    assert report['missing'] == 0

    lost_edit = {**after, 'c': _item('z')}
    assert audit_state(before, lost_edit, edited={'c'}, edit_state=edit)['wrong_edit'] == 1
    assert audit_state(before, {'a': after['a']}, edited=set(), edit_state=edit)['missing'] == 3


def test_a_validated_item_stored_outside_its_class_cluster_is_reported_apart() -> None:
    before = {'a': _item('x', class_id=3, class_validated=True, cluster_id=10_001)}
    after = {'a': _item('x', class_id=3, class_validated=True, cluster_id=3)}
    report = audit_state(before, after, edited=set(), edit_state={})
    assert report['normalised_cluster_only'] == 1
    assert report['changed_or_not_restored'] == 0


def test_expected_histogram_moves_the_edited_items() -> None:
    before = {'x': 5, 'y': 3, '<unset>': 2}
    edited = Counter({'x': 2, '<unset>': 1})
    assert expected_histogram(before, edited, 'probe_b') == {
        'x': 3,
        'y': 3,
        '<unset>': 1,
        'probe_b': 3,
    }


def test_rss_of_a_missing_process_is_none() -> None:
    assert rss_mb(2**22 + 12345) is None
