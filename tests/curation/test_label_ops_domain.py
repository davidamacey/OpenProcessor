"""The pure parts of a label operation: its id, the per-item undo decision, the
history stack, and the shared restore. No route, no index."""

from __future__ import annotations

import copy
from datetime import UTC, datetime
from typing import Any

import pytest

from curation.query_fakes import matches
from src.domain.audit_math import AUDIT_HUMAN, AUDIT_OUTCOME, AUDIT_SAMPLE
from src.domain.history import (
    CLASS_STATE_FIELDS,
    HUMAN_UNLABEL_WRITER,
    find_undo_snapshot,
    op_write_undone,
    record_class_snapshot,
    restore_class_state,
)
from src.domain.label_ops import (
    LABEL_OP_ID_RE,
    OpFingerprint,
    UndoOutcome,
    current_query,
    label_fingerprint,
    move_fingerprint,
    new_label_op_id,
    undo_outcome,
)


OP = 'lop_20260301T120000000000_abc123'
OTHER = 'lop_20260301T130000000000_def456'
AT = '2026-03-01T12:00:00.000000+00:00'
BEFORE = '2026-03-01T11:00:00.000000+00:00'
AFTER = '2026-03-01T13:00:00.000000+00:00'
WRITER = 'human:batch_label_crops'
NAMES = {1: 'widget', 2: 'gadget', 3: 'gizmo'}


def _registry_name(class_id: int) -> str | None:
    return NAMES.get(class_id)


def _fp(**kw: Any) -> OpFingerprint:
    return label_fingerprint(op_id=OP, writer=WRITER, at=AT, class_name='widget', **kw)


def _written(prior: dict[str, Any] | None = None, op_id: str = OP) -> dict[str, Any]:
    """An item as a bulk label of 'widget' left it, over ``prior`` (an unlabeled proposal)."""
    before = {'class_id': 2, 'class_name': 'gadget', 'class_source': 'vlm', **(prior or {})}
    history = record_class_snapshot(
        before, writer=WRITER, restorable=True, op_id=op_id, now='2026-03-01T12:00:01+00:00'
    )
    return {
        **before,
        'class_id': 1,
        'class_name': 'widget',
        'class_source': 'human',
        'class_validated': True,
        'cluster_id': 1,
        'label_op_id': op_id,
        'class_id_history': history,
    }


def test_op_id_format_and_regex() -> None:
    now = datetime(2026, 3, 1, 12, 0, 0, 123456, tzinfo=UTC)
    op_id = new_label_op_id(now)
    assert op_id.startswith('lop_20260301T120000123456_')
    assert LABEL_OP_ID_RE.fullmatch(op_id)
    later = new_label_op_id(datetime(2026, 3, 1, 12, 0, 1, tzinfo=UTC))
    assert later > op_id, 'lexicographic order is creation order'
    for bad in (
        '',
        '../x',
        'lop_20260301T120000123456_ABCDEF',
        f'{op_id}/..',
        f'{op_id}\n',
        'x' + op_id,
    ):
        assert LABEL_OP_ID_RE.fullmatch(bad) is None, bad


# (name, mutate(doc, fp-kwargs) -> doc, expected). One row per rule of the decision, in order.
def _rows() -> list[tuple[str, Any, UndoOutcome]]:
    def holdout(d: dict[str, Any]) -> dict[str, Any]:
        return {**d, 'test_holdout': True}

    def relabelled_by_person(d: dict[str, Any]) -> dict[str, Any]:
        d['class_id_history'] = record_class_snapshot(
            d, writer='human:label_crop', restorable=True, now=AFTER
        )
        return {**d, 'label_op_id': None, 'class_id': 3, 'class_name': 'gizmo'}

    def undone_one_by_one(d: dict[str, Any]) -> dict[str, Any]:
        d['class_id_history'] = record_class_snapshot(
            d, writer=HUMAN_UNLABEL_WRITER, restorable=False, op_id=OP, now=AFTER
        )
        return {**d, 'label_op_id': None}

    def later_op_on_top(d: dict[str, Any]) -> dict[str, Any]:
        d['class_id_history'] = record_class_snapshot(
            d, writer=WRITER, restorable=True, op_id=OTHER, now=AFTER
        )
        return {**d, 'label_op_id': OP}  # tag forced equal: only the stack can tell

    return [
        ('deleted', lambda _d: None, UndoOutcome.KEPT_CHANGED),
        ('other_tag', lambda d: {**d, 'label_op_id': OTHER}, UndoOutcome.KEPT_CHANGED),
        ('untagged_after_undo', undone_one_by_one, UndoOutcome.ALREADY_UNDONE),
        ('untagged_by_person', relabelled_by_person, UndoOutcome.KEPT_CHANGED),
        ('holdout', holdout, UndoOutcome.KEPT_HOLDOUT),
        ('later_op_on_top', later_op_on_top, UndoOutcome.KEPT_CHANGED),
        (
            'excluded_after',
            lambda d: {**d, 'class_excluded': True, 'excluded_at': AFTER},
            UndoOutcome.KEPT_CHANGED,
        ),
        ('excluded_no_stamp', lambda d: {**d, 'class_excluded': True}, UndoOutcome.KEPT_CHANGED),
        (
            'excluded_before',
            lambda d: {**d, 'class_excluded': True, 'excluded_at': BEFORE},
            UndoOutcome.RESTORE,
        ),
        ('merged_class', lambda d: {**d, 'class_name': 'gadget'}, UndoOutcome.KEPT_CHANGED),
        ('source_changed', lambda d: {**d, 'class_source': 'human_move'}, UndoOutcome.KEPT_CHANGED),
        ('unvalidated', lambda d: {**d, 'class_validated': False}, UndoOutcome.KEPT_CHANGED),
        ('renamed_restore_target', lambda d: {**d, '_rename': True}, UndoOutcome.KEPT_STALE_CLASS),
        ('untouched', lambda d: d, UndoOutcome.RESTORE),
    ]


@pytest.mark.parametrize(('name', 'mutate', 'expected'), _rows(), ids=[r[0] for r in _rows()])
def test_undo_outcome_decision_table(name: str, mutate: Any, expected: UndoOutcome) -> None:
    doc = mutate(_written())
    registry = _registry_name
    if doc is not None and doc.pop('_rename', False):
        registry = lambda cid: 'renamed' if cid == 2 else _registry_name(cid)  # noqa: E731
    assert undo_outcome(doc, _fp(), registry_name=registry) is expected


def test_undo_outcome_rule_order() -> None:
    # A holdout item carrying another op's tag is kept_changed (rule 2 before rule 3);
    # an excluded-after holdout item is kept_holdout (rule 3 before rule 5).
    doc = {**_written(), 'test_holdout': True}
    assert undo_outcome({**doc, 'label_op_id': OTHER}, _fp(), registry_name=_registry_name) is (
        UndoOutcome.KEPT_CHANGED
    )
    both = {**doc, 'class_excluded': True, 'excluded_at': AFTER}
    assert undo_outcome(both, _fp(), registry_name=_registry_name) is UndoOutcome.KEPT_HOLDOUT
    # A merged class on a stale restore target is kept_changed (rule 6 before rule 7).
    stale = {**_written(), 'class_name': 'gadget'}
    assert (
        undo_outcome(stale, _fp(), registry_name=lambda _cid: 'renamed') is UndoOutcome.KEPT_CHANGED
    )


def test_undo_outcome_class_names_compare_normalized() -> None:
    doc = {**_written(), 'class_name': 'Widget '}
    assert undo_outcome(doc, _fp(), registry_name=_registry_name) is UndoOutcome.RESTORE


def test_a_candidate_move_fingerprint_compares_the_cluster_not_the_class() -> None:
    fp = move_fingerprint(
        op_id=OP, writer='human:move_crops', at=AT, class_name=None, cluster_id=10_005
    )
    base = {'class_id': 3, 'class_name': 'gizmo', 'class_source': 'vlm'}
    history = record_class_snapshot(
        base, writer='human:move_crops', restorable=True, op_id=OP, now=AT
    )
    doc = {
        **base,
        'cluster_id': 10_005,
        'class_validated': False,
        'label_op_id': OP,
        'class_id_history': history,
    }
    assert undo_outcome(doc, fp, registry_name=_registry_name) is UndoOutcome.RESTORE
    assert undo_outcome({**doc, 'cluster_id': 10_006}, fp, registry_name=_registry_name) is (
        UndoOutcome.KEPT_CHANGED
    )
    assert undo_outcome({**doc, 'class_validated': True}, fp, registry_name=_registry_name) is (
        UndoOutcome.KEPT_CHANGED
    )


def test_op_write_undone_follows_the_stack() -> None:
    doc = _written()
    assert op_write_undone(doc['class_id_history'], OP) is False
    h = record_class_snapshot(
        doc, writer=HUMAN_UNLABEL_WRITER, restorable=False, op_id=OP, now=AFTER
    )
    assert op_write_undone(h, OP) is True
    assert op_write_undone(h, OTHER) is False
    # Redone by the same op id after an undo (A, B, undo B, undo A): A's unlabel entry counts.
    # An unlabel entry that belongs to another op does not.
    h2 = record_class_snapshot(
        doc, writer=HUMAN_UNLABEL_WRITER, restorable=False, op_id=OTHER, now=AFTER
    )
    assert op_write_undone(h2, OP) is False
    # A later write of the same op id after the unlabel makes it live again.
    h3 = record_class_snapshot(
        {**doc, 'class_id_history': h}, writer=WRITER, restorable=True, op_id=OP, now=AFTER
    )
    assert op_write_undone(h3, OP) is False
    assert op_write_undone(None, OP) is False


def test_class_undo_update_matches_per_item_undo(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.routers.curation import label_undo
    from src.services.curation import class_undo

    monkeypatch.setattr(class_undo, 'now_iso', lambda: AFTER)
    monkeypatch.setattr('src.domain.history.now_iso', lambda: AFTER)
    for excluded in (False, True):
        doc = {**_written(), 'class_excluded': excluded}
        shared = class_undo.class_undo_update(copy.deepcopy(doc))
        per_item = label_undo._undo_merger(require_history=True)(copy.deepcopy(doc))
        assert shared == per_item
        assert shared['class_id_history'][-1]['op_id'] == OP
        assert shared['class_id_history'][-1]['writer'] == HUMAN_UNLABEL_WRITER
        assert shared['label_op_id'] is None


def test_snapshot_records_and_restore_returns_the_tag_and_audit_verdict() -> None:
    current = {
        'class_id': 2,
        'class_name': 'gadget',
        'class_source': 'vlm',
        'label_op_id': OTHER,
        AUDIT_SAMPLE: True,
        AUDIT_OUTCOME: 'agree',
        AUDIT_HUMAN: 'gadget',
    }
    history = record_class_snapshot(current, writer=WRITER, restorable=True, op_id=OP, now=AT)
    entry = history[-1]
    assert entry['op_id'] == OP
    assert entry['label_op_id'] == OTHER
    assert (entry[AUDIT_OUTCOME], entry[AUDIT_HUMAN]) == ('agree', 'gadget')
    restored = restore_class_state(entry)
    assert restored['label_op_id'] == OTHER
    assert (restored[AUDIT_OUTCOME], restored[AUDIT_HUMAN]) == ('agree', 'gadget')
    assert 'label_op_id' in CLASS_STATE_FIELDS

    # No verdict on the item before the write: the restore clears the one the write stamped.
    sampled = {**current, AUDIT_OUTCOME: None, AUDIT_HUMAN: None}
    entry = record_class_snapshot(sampled, writer=WRITER, restorable=True, now=AT)[-1]
    assert restore_class_state(entry)[AUDIT_OUTCOME] is None

    # An item outside the audit records nothing and a restore never touches the verdict fields.
    plain = record_class_snapshot({'class_id': 2}, writer=WRITER, restorable=True, now=AT)[-1]
    assert AUDIT_OUTCOME not in plain
    assert AUDIT_OUTCOME not in restore_class_state(plain)
    assert find_undo_snapshot([plain]) is plain


def test_current_query_matches_undo_outcome_restore() -> None:
    fp = _fp()
    base = _written()
    docs: dict[str, dict[str, Any]] = {
        'ok1': copy.deepcopy(base),
        'ok2': copy.deepcopy(base),
        'excl_before': {**copy.deepcopy(base), 'class_excluded': True, 'excluded_at': BEFORE},
        'excl_after': {**copy.deepcopy(base), 'class_excluded': True, 'excluded_at': AFTER},
        'excl_unstamped': {**copy.deepcopy(base), 'class_excluded': True},
        'holdout': {**copy.deepcopy(base), 'test_holdout': True},
        'relabelled': {**copy.deepcopy(base), 'label_op_id': None, 'class_name': 'gizmo'},
        'other_op': {**copy.deepcopy(base), 'label_op_id': OTHER},
        'merged': {**copy.deepcopy(base), 'class_name': 'gadget'},
        'unvalidated': {**copy.deepcopy(base), 'class_validated': False},
    }
    for doc_id, doc in docs.items():
        doc['crop_id'] = doc_id
    expected = {
        i
        for i, d in docs.items()
        if undo_outcome(d, fp, registry_name=_registry_name) is UndoOutcome.RESTORE
    }
    assert expected == {'ok1', 'ok2', 'excl_before'}
    selected = {i for i, d in docs.items() if matches(d, current_query(fp))}
    assert selected == expected
    # The same holds for a move to a candidate cluster.
    cfp = move_fingerprint(
        op_id=OP, writer='human:move_crops', at=AT, class_name=None, cluster_id=9
    )
    cdocs = {
        'a': {'label_op_id': OP, 'cluster_id': 9, 'class_validated': False},
        'b': {'label_op_id': OP, 'cluster_id': 8, 'class_validated': False},
        'c': {'label_op_id': OP, 'cluster_id': 9, 'class_validated': True},
    }
    assert {i for i, d in cdocs.items() if matches(d, current_query(cfp))} == {'a'}
