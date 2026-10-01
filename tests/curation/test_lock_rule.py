"""W10.10 lock rule: ``is_locked_class`` / ``_is_locked_marker`` /
``is_locked_box`` / ``is_locked_item`` (``src/clients/occ.py``).

A human-set label or box, or a validated imported one, is never touched
by an automated writer. This module gates the table any_domain_plan.md
W10.10 specifies, plus the two live call sites the plan calls out
(re-ingest preserving an imported item; ``/vlm/label_batch`` skipping
one).
"""

from __future__ import annotations

from src.clients.occ import (
    _is_locked_marker,
    _merge_preserving_human,
    is_locked_box,
    is_locked_class,
    is_locked_item,
)
from src.config import get_region_fields
from src.services.curation.region_boxes import RegionBox


F = get_region_fields()


class TestIsLockedClass:
    def test_human_and_human_move_locked(self) -> None:
        assert is_locked_class({'class_source': 'human'}) is True
        assert is_locked_class({'class_source': 'human_move'}) is True
        assert is_locked_class({'class_source': 'vlm_human_confirmed'}) is True

    def test_machine_sources_unlocked(self) -> None:
        assert is_locked_class({'class_source': 'vlm'}) is False
        assert is_locked_class({'class_source': 'classifier_model'}) is False
        assert is_locked_class({}) is False

    def test_validated_import_locked(self) -> None:
        assert is_locked_class({'class_source': 'external_label', 'class_validated': True}) is True

    def test_suggestion_trust_import_unlocked(self) -> None:
        """``label_trust: suggestion`` imports are NOT locked — the
        machine pipeline may still relabel them (W10.10)."""
        assert (
            is_locked_class({'class_source': 'external_label', 'class_validated': False}) is False
        )

    def test_holdout_locked_regardless_of_class_source(self) -> None:
        assert is_locked_class({'class_source': 'vlm', 'test_holdout': True}) is True
        assert is_locked_class({'test_holdout': True}) is True


class TestIsLockedMarker:
    def test_human_markers_locked(self) -> None:
        assert _is_locked_marker('human') is True
        assert _is_locked_marker('human_move') is True

    def test_import_markers_locked(self) -> None:
        assert _is_locked_marker('import') is True
        assert _is_locked_marker('external_label') is True

    def test_machine_markers_unlocked(self) -> None:
        assert _is_locked_marker('ingest') is False
        assert _is_locked_marker('vlm') is False
        assert _is_locked_marker(None) is False


class TestIsLockedBox:
    def test_human_created_box_locked(self) -> None:
        box = RegionBox(box_id='b1', bbox_norm=(0, 0, 1, 1), state='accepted', source='human')
        assert is_locked_box(box) is True

    def test_import_box_locked(self) -> None:
        box = RegionBox(box_id='b1', bbox_norm=(0, 0, 1, 1), state='accepted', source='import')
        assert is_locked_box(box) is True

    def test_suggestion_import_box_is_not_locked(self) -> None:
        """``label_trust: suggestion`` writes ``proposed`` boxes with
        ``source: import``: the machine pipeline may still replace them."""
        box = RegionBox(box_id='b1', bbox_norm=(0, 0, 1, 1), state='proposed', source='import')
        assert is_locked_box(box) is False
        accepted = RegionBox(box_id='b1', bbox_norm=(0, 0, 1, 1), state='accepted', source='import')
        assert is_locked_box(accepted) is True

    def test_machine_box_unlocked(self) -> None:
        box = RegionBox(box_id='b1', bbox_norm=(0, 0, 1, 1), state='proposed', source='segmenter')
        assert is_locked_box(box) is False


class TestIsLockedItem:
    def test_locked_class_locks_item(self) -> None:
        assert is_locked_item({'class_source': 'human'}) is True

    def test_locked_box_locks_item(self) -> None:
        source = {
            'class_source': 'vlm',
            F.boxes: [
                {'box_id': 'b1', 'bbox_norm': [0, 0, 1, 1], 'state': 'accepted', 'source': 'human'}
            ],
        }
        assert is_locked_item(source) is True

    def test_validated_human_verifier_locks_item(self) -> None:
        source = {'class_source': 'vlm', F.validated: True, F.verifier: 'human'}
        assert is_locked_item(source) is True

    def test_validated_import_verifier_locks_item(self) -> None:
        source = {'class_source': 'vlm', F.validated: True, F.verifier: 'import'}
        assert is_locked_item(source) is True

    def test_unlocked_machine_item(self) -> None:
        source = {
            'class_source': 'vlm',
            F.validated: True,
            F.verifier: 'sam3',
            F.boxes: [
                {
                    'box_id': 'b1',
                    'bbox_norm': [0, 0, 1, 1],
                    'state': 'accepted',
                    'source': 'segmenter',
                }
            ],
        }
        assert is_locked_item(source) is False


class TestReingestPreservesImportedLabel:
    """A re-ingest merge must never clobber an imported item's guard
    fields — the OCC guard the ingest pipeline runs through
    ``occ_upsert_bulk`` (``ingest.py``'s ``_CROP_HUMAN_FIELD_GUARDS =
    ('label_source', 'class_source')``). Red before W10:
    ``_merge_preserving_human`` used ``is_human_marker``, which does not
    recognize import provenance.

    W10 fix-pass note (Opus review 2026-09-28, lock-rule call-site m4):
    the guard now fires on ``is_locked_class`` for these two fields, which
    requires ``class_validated=True`` for an import to lock — matching
    ``is_locked_class``'s own contract that an unvalidated ("suggestion")
    import is NOT locked. The prior version of this test asserted the
    opposite (preserved with no ``class_validated`` at all) and is now
    the ``test_unvalidated_import_not_preserved`` case below.
    """

    def test_validated_import_class_source_preserved(self) -> None:
        existing = {
            'class_source': 'external_label',
            'label_source': 'import',
            'class_id': 3,
            'class_validated': True,
        }
        new_doc = {'class_source': 'coco_yolo11', 'label_source': 'ingest', 'class_id': 7}
        merged, preserved = _merge_preserving_human(
            new_doc=new_doc,
            existing=existing,
            human_field_guards=['label_source', 'class_source'],
        )
        assert merged['class_source'] == 'external_label'
        assert merged['label_source'] == 'import'
        assert set(preserved) == {'label_source', 'class_source'}

    def test_unvalidated_suggestion_import_not_preserved(self) -> None:
        """is_locked_class explicitly does NOT lock an unvalidated
        (``label_trust: suggestion``) import — the merge guard must
        agree, so a fresh ingest pass is free to overwrite it."""
        existing = {
            'class_source': 'external_label',
            'label_source': 'import',
            'class_id': 3,
            'class_validated': False,
        }
        new_doc = {'class_source': 'coco_yolo11', 'label_source': 'ingest', 'class_id': 7}
        merged, preserved = _merge_preserving_human(
            new_doc=new_doc,
            existing=existing,
            human_field_guards=['label_source', 'class_source'],
        )
        assert merged['class_source'] == 'coco_yolo11'
        assert merged['label_source'] == 'ingest'
        assert preserved == []

    def test_machine_class_source_not_preserved(self) -> None:
        existing = {'class_source': 'vlm', 'label_source': 'vlm'}
        new_doc = {'class_source': 'coco_yolo11', 'label_source': 'ingest'}
        merged, preserved = _merge_preserving_human(
            new_doc=new_doc,
            existing=existing,
            human_field_guards=['label_source', 'class_source'],
        )
        assert merged['class_source'] == 'coco_yolo11'
        assert preserved == []


class TestVlmLabelBatchSkipsLockedItems:
    def test_class_write_locked_covers_validated_import(self) -> None:
        from src.services.curation.class_write_guard import class_write_locked

        assert class_write_locked({'class_source': 'external_label', 'class_validated': True})
        assert not class_write_locked({'class_source': 'external_label', 'class_validated': False})
