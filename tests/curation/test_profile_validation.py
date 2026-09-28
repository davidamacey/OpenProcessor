"""W4: region-profile validation (any_domain_plan.md §4.3) --
``src.services.config_store.profile_validation.validate_profile``.

Deviation note: the plan's §4.3 table names a ``segmenter_min_score``
field; ``DetectionProfile`` has no such field -- ``confidence_floor``
"doubles as the selection floor" for the segmenter leg (its own
docstring). The range test below exercises the real field name.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from _region_profile_fixture import EXAMPLE_LICENSE_PLATE_PROFILE

from src.config import DetectionProfile
from src.services.config_store.profile_validation import PROFILE_FIELD_RANGES, validate_profile


def _body(*, profile: DetectionProfile | None = None, **overrides: Any) -> dict[str, Any]:
    from dataclasses import asdict

    base = profile or DetectionProfile(name='p')
    raw = asdict(base)
    raw.pop('name')
    for key, value in raw.items():
        if isinstance(value, frozenset):
            raw[key] = sorted(value)
        elif isinstance(value, tuple):
            raw[key] = list(value)
    raw.update(overrides)
    return raw


async def _always_reachable() -> list[dict[str, Any]]:
    return [
        {'name': 'my_detector', 'state': 'READY', 'version': '1'},
        {'name': 'paddleocr_det_trt', 'state': 'READY', 'version': '1'},
        {'name': 'paddleocr_rec_trt', 'state': 'READY', 'version': '1'},
        {'name': 'ocr_pipeline', 'state': 'READY', 'version': '1'},
    ]


async def _always_unreachable() -> list[dict[str, Any]]:
    raise ConnectionError('triton down')


async def _segmenter_ready() -> tuple[str, str | None]:
    return 'ready', None


async def _segmenter_down() -> tuple[str, str | None]:
    return 'unavailable', 'connection refused'


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


def test_valid_segmenter_only_profile_has_zero_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('OP_SEGMENTER_URL', 'http://segmenter:8000')
    body = _body(
        segmenter_text_prompt='wheel',
        detector_model='',
        display_name='X',
        display_name_singular='x',
        text_reader='none',
    )
    report = _run(
        validate_profile(
            None, body, get_repository_index=_always_reachable, segmenter_health=_segmenter_ready
        )
    )
    assert report.errors == [], report.errors


def test_no_candidate_source_when_no_detector_and_no_prompt() -> None:
    body = _body(detector_model='', segmenter_text_prompt='')
    report = _run(validate_profile(None, body, get_repository_index=_always_reachable))
    assert any(e.code == 'no_candidate_source' for e in report.errors)


def test_profile_name_invalid() -> None:
    report = _run(validate_profile('Not Valid', _body()))
    assert any(e.code == 'profile_name_invalid' for e in report.errors)


def test_profile_name_reserved() -> None:
    report = _run(validate_profile('off', _body()))
    assert any(e.code == 'profile_name_reserved' for e in report.errors)


def test_name_conflict_is_report_issue() -> None:
    report = _run(validate_profile('taken', _body(), existing_names=frozenset({'taken'})))
    assert any(e.code == 'name_conflict' for e in report.errors)


def test_profile_field_unknown() -> None:
    body = _body()
    body['not_a_real_field'] = 1
    report = _run(validate_profile(None, body))
    assert any(e.code == 'profile_field_unknown' for e in report.errors)


def test_region_class_name_invalid() -> None:
    body = _body(region_class_name='Not Valid!')
    report = _run(validate_profile(None, body))
    assert any(e.code == 'region_class_name_invalid' for e in report.errors)


def test_text_reader_invalid() -> None:
    body = _body()
    body['text_reader'] = 'not_a_mode'
    report = _run(validate_profile(None, body))
    assert any(e.code == 'text_reader_invalid' for e in report.errors)


def test_text_regex_invalid() -> None:
    body = _body(text_charset='[invalid(')
    report = _run(validate_profile(None, body))
    assert any(e.code == 'text_regex_invalid' for e in report.errors)


@pytest.mark.parametrize(
    ('field', 'value'),
    [
        ('max_regions_per_item', 0),
        ('max_regions_per_item', 65),
        ('confidence_floor', 1.1),
        ('region_nms_iou', 0),
    ],
)
def test_field_ranges_out_of_bounds(field: str, value: float) -> None:
    body = _body(**{field: value})  # type: ignore[arg-type]
    report = _run(validate_profile(None, body))
    matches = [e for e in report.errors if e.code == 'profile_field_range' and e.field == field]
    assert matches, f'{field}={value} should be a profile_field_range error; got {report.errors}'


def test_schema_row_min_max_equal_validator_range() -> None:
    for lo, hi in PROFILE_FIELD_RANGES.values():
        assert lo is not None
        assert hi is not None
        assert lo <= hi


def test_input_size_must_be_multiple_of_32() -> None:
    body = _body(input_size=100)
    report = _run(validate_profile(None, body))
    assert any(e.code == 'profile_field_range' and e.field == 'input_size' for e in report.errors)


def test_auto_confirm_area_frac_range() -> None:
    body = _body(auto_confirm_area_frac=[0.5, 0.1])
    report = _run(validate_profile(None, body))
    assert any(
        e.code == 'profile_field_range' and e.field == 'auto_confirm_area_frac'
        for e in report.errors
    )


def test_letterbox_fill_range() -> None:
    body = _body(letterbox_fill=[114, 300, -1])
    report = _run(validate_profile(None, body))
    assert any(
        e.code == 'profile_field_range' and e.field == 'letterbox_fill' for e in report.errors
    )


def test_detector_model_not_found() -> None:
    body = _body(detector_model='nonexistent_model')
    report = _run(validate_profile(None, body, get_repository_index=_always_reachable))
    assert any(e.code == 'detector_model_not_found' for e in report.errors)


def test_detector_model_not_ready_warns_on_save() -> None:
    async def _index() -> list[dict[str, Any]]:
        return [{'name': 'my_detector', 'state': 'LOADING'}]

    body = _body(detector_model='my_detector', text_reader='none')
    report = _run(validate_profile(None, body, get_repository_index=_index))
    assert any(w.code == 'detector_model_not_ready' for w in report.warnings)
    assert report.ok  # a warning never blocks save


def test_detector_model_not_ready_errors_on_activation_unless_force() -> None:
    async def _index() -> list[dict[str, Any]]:
        return [{'name': 'my_detector', 'state': 'LOADING'}]

    body = _body(detector_model='my_detector', text_reader='none')
    report = _run(validate_profile(None, body, get_repository_index=_index, for_activation=True))
    assert not report.ok
    issue = next(e for e in report.errors if e.code == 'detector_model_not_ready')
    assert issue.bypassable is True
    assert report.force_allowed is True


def test_triton_unreachable_warns_on_save_errors_on_activation() -> None:
    body = _body(detector_model='my_detector')
    report = _run(validate_profile(None, body, get_repository_index=_always_unreachable))
    assert any(w.code == 'triton_unreachable' for w in report.warnings)
    report_act = _run(
        validate_profile(None, body, get_repository_index=_always_unreachable, for_activation=True)
    )
    assert any(e.code == 'triton_unreachable' for e in report_act.errors)
    assert report_act.force_allowed is True


def test_segmenter_prompt_empty_error_when_sole_leg() -> None:
    body = _body(detector_model='', segmenter_text_prompt='')
    report = _run(validate_profile(None, body))
    assert any(e.code == 'segmenter_prompt_empty' for e in report.errors)


def test_segmenter_prompt_too_long() -> None:
    body = _body(segmenter_text_prompt='x' * 201, detector_model='m')
    report = _run(validate_profile(None, body, get_repository_index=_always_reachable))
    assert any(e.code == 'segmenter_prompt_too_long' for e in report.errors)


def test_segmenter_prompt_too_many_phrases() -> None:
    body = _body(segmenter_text_prompt=','.join(['a'] * 9), detector_model='m')
    report = _run(validate_profile(None, body, get_repository_index=_always_reachable))
    assert any(e.code == 'segmenter_prompt_too_many_phrases' for e in report.errors)


def test_segmenter_prompt_multiline() -> None:
    body = _body(segmenter_text_prompt='a\nb', detector_model='m')
    report = _run(validate_profile(None, body, get_repository_index=_always_reachable))
    assert any(e.code == 'segmenter_prompt_multiline' for e in report.errors)


def test_segmenter_not_configured_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('OP_SEGMENTER_URL', raising=False)
    body = _body(segmenter_text_prompt='wheel', detector_model='m')
    report = _run(validate_profile(None, body, get_repository_index=_always_reachable))
    assert any(w.code == 'segmenter_not_configured' for w in report.warnings)


def test_segmenter_unreachable_warns_then_errors_on_activation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv('OP_SEGMENTER_URL', 'http://segmenter:8000')
    body = _body(segmenter_text_prompt='wheel', detector_model='m')
    report = _run(
        validate_profile(
            None, body, get_repository_index=_always_reachable, segmenter_health=_segmenter_down
        )
    )
    assert any(w.code == 'segmenter_unreachable' for w in report.warnings)
    report_act = _run(
        validate_profile(
            None,
            body,
            get_repository_index=_always_reachable,
            segmenter_health=_segmenter_down,
            for_activation=True,
        )
    )
    assert any(e.code == 'segmenter_unreachable' for e in report_act.errors)


def test_ocr_model_not_found() -> None:
    body = _body(
        text_reader='ocr', ocr_pipeline_model='missing_pipeline', ocr_det_model='', ocr_rec_model=''
    )
    report = _run(validate_profile(None, body, get_repository_index=_always_reachable))
    assert any(e.code == 'ocr_model_not_found' for e in report.errors)


def test_vlm_not_configured_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('OP_VLM_URL', raising=False)
    body = _body(text_reader='vlm', ocr_pipeline_model='', ocr_det_model='', ocr_rec_model='')
    report = _run(validate_profile(None, body, get_repository_index=_always_reachable))
    assert any(w.code == 'vlm_not_configured' for w in report.warnings)


def test_text_hint_inactive_no_ocr() -> None:
    body = _body(text_hint_enabled=True, ocr_pipeline_model='')
    report = _run(validate_profile(None, body, get_repository_index=_always_reachable))
    assert any(w.code == 'text_hint_inactive_no_ocr' for w in report.warnings)


def test_text_fields_ignored_when_text_reader_none() -> None:
    body = _body(text_reader='none', text_uppercase=True)
    report = _run(validate_profile(None, body))
    assert any(w.code == 'text_fields_ignored' for w in report.warnings)


def test_display_name_missing_warning() -> None:
    body = _body(display_name='', display_name_singular='')
    report = _run(validate_profile(None, body, get_repository_index=_always_reachable))
    assert any(w.code == 'display_name_missing' for w in report.warnings)


def test_parent_class_unknown_warning() -> None:
    body = _body(parent_classes=['not_a_class'])
    report = _run(
        validate_profile(
            None, body, get_repository_index=_always_reachable, class_names=frozenset({'car'})
        )
    )
    assert any(w.code == 'parent_class_unknown' for w in report.warnings)


def test_example_license_plate_profile_has_zero_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('OP_SEGMENTER_URL', 'http://segmenter:8000')
    body = _body(profile=EXAMPLE_LICENSE_PLATE_PROFILE, detector_model='my_detector')
    report = _run(
        validate_profile(
            None, body, get_repository_index=_always_reachable, segmenter_health=_segmenter_ready
        )
    )
    assert report.errors == [], report.errors


def test_activating_n4_profile_against_flat_active_pack_is_never_bypassable() -> None:
    from src.services.labeling.vlm_prompts import GENERIC_REGION_PACK, PromptPack

    flat_pack = PromptPack(
        **{
            **GENERIC_REGION_PACK.to_dict(),
            'combined_system': 'Return class_id, class_confidence, region_visible, '
            'region_bbox_correct, region_confidence.',
            'combined_user_template': '{class_block}{region_block}Answer with the fields above.',
        }
    )
    body = _body(max_regions_per_item=4, detector_model='m', segmenter_text_prompt='')
    report = _run(
        validate_profile(
            None,
            body,
            get_repository_index=_always_reachable,
            for_activation=True,
            active_pack=flat_pack,
        )
    )
    issue = next(e for e in report.errors if e.code == 'pack_multi_region_keys_missing')
    assert issue.bypassable is False
