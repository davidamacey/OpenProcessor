"""_frame_row_for: one export row per source frame, import provenance and holdout sticky."""

from __future__ import annotations

from src.services.curation.export_single_class_rows import _frame_row_for


def test_missing_image_path_has_no_row() -> None:
    per_frame: dict = {}
    assert _frame_row_for(per_frame, {'image_id': 'i1'}) is None
    assert per_frame == {}


def test_items_of_one_frame_share_a_row_keyed_by_image_id() -> None:
    per_frame: dict = {}
    a = _frame_row_for(per_frame, {'image_id': 'i1', 'image_path': '/a.jpg'})
    b = _frame_row_for(
        per_frame,
        {
            'image_id': 'i1',
            'image_path': '/a.jpg',
            'test_holdout': True,
            'dataset_split': 'val',
            'import_hard_negative': True,
        },
    )
    assert a is not None
    assert a is b
    assert list(per_frame) == ['i1']
    assert (a.item_id, a.image_id, a.image_path) == ('i1', 'i1', '/a.jpg')
    assert a.has_test_crop is True
    assert a.dataset_split == 'val'
    assert a.is_hard_negative is True


def test_a_frame_without_image_id_is_keyed_by_its_path() -> None:
    per_frame: dict = {}
    row = _frame_row_for(per_frame, {'image_path': '/b.jpg'})
    assert row is not None
    assert list(per_frame) == ['/b.jpg']
