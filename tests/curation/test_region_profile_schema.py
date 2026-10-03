"""``GET /region_profiles/schema`` describes every profile field for real."""

from __future__ import annotations

from src.routers.curation._region_profile_models import RegionProfileBody, RegionProfileFieldSchema
from src.routers.curation._region_profile_schema import GROUPS, build_region_profile_schema


def _rows() -> dict[str, RegionProfileFieldSchema]:
    return {row.field: row for row in build_region_profile_schema().fields}


def test_one_row_per_body_field() -> None:
    assert set(_rows()) == set(RegionProfileBody.model_fields)


def test_every_row_is_in_a_declared_group() -> None:
    ids = {g.id for g in GROUPS}
    assert {row.group for row in _rows().values()} <= ids


def test_types_come_from_the_field_annotations() -> None:
    rows = _rows()
    assert rows['confidence_floor'].type == 'float'
    assert rows['batch_limit'].type == 'int'
    assert rows['text_uppercase'].type == 'bool'
    assert rows['letterbox_fill'].type == 'rgb'
    assert rows['auto_confirm_aspect'].type == 'float_pair'
    assert rows['parent_classes'].type == 'string_list'
    assert rows['detector_model'].type == 'string'


def test_ranges_defaults_and_choices_are_served() -> None:
    rows = _rows()
    assert (rows['confidence_floor'].min, rows['confidence_floor'].max) == (0.0, 1.0)
    assert rows['confidence_floor'].default == 0.4
    assert rows['parent_classes'].default == []
    assert rows['detector_model'].choices_from == 'detectors'
    assert rows['detector_model'].empty_choice is not None
    assert rows['detector_model'].empty_choice.id == ''


def test_fields_are_grouped_flagged_and_scoped_not_one_flat_advanced_list() -> None:
    rows = _rows()
    assert rows['detector_model'].group == 'detector'
    assert rows['text_reader'].group == 'text'
    assert rows['text_hint_enabled'].applies_when == 'text_hint'
    assert rows['segmenter_name'].applies_when == 'segmenter'
    assert rows['detector_model'].advanced is False
    assert rows['input_size'].advanced is True
    assert len({row.group for row in rows.values()}) > 3
