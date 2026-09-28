"""W3: prompt-pack validation (any_domain_plan.md §3.3) --
``src.services.config_store.pack_validation.validate_pack``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.services.config_store.pack_validation import validate_pack
from src.services.labeling.vlm_prompts import GENERIC_ITEM_PACK, GENERIC_REGION_PACK


def _valid_body() -> dict[str, Any]:
    body = GENERIC_ITEM_PACK.to_dict()
    body.pop('name')
    return body


def test_generic_item_pack_has_zero_errors() -> None:
    report = validate_pack(None, _valid_body())
    assert report.errors == [], report.errors
    assert report.ok


def test_generic_region_pack_has_zero_errors() -> None:
    body = GENERIC_REGION_PACK.to_dict()
    body.pop('name')
    report = validate_pack(None, body)
    assert report.errors == [], report.errors


def test_example_prompt_pack_templates_have_zero_errors() -> None:
    root = Path(__file__).resolve().parents[2] / 'examples' / 'prompt_packs'
    files = list(root.glob('*.json'))
    assert files, 'expected at least one example prompt pack'
    for path in files:
        data = json.loads(path.read_text(encoding='utf-8'))
        data = {k: v for k, v in data.items() if not k.startswith('_')}
        data.pop('name', None)
        report = validate_pack(None, data)
        assert report.errors == [], f'{path}: {report.errors}'


def test_pack_field_missing_on_absent_field() -> None:
    body = _valid_body()
    del body['class_system']
    report = validate_pack(None, body)
    codes = {e.code for e in report.errors}
    assert 'pack_field_missing' in codes


def test_pack_field_empty_on_blank_string() -> None:
    body = _valid_body()
    body['class_system'] = '   '
    report = validate_pack(None, body)
    codes = {e.code for e in report.errors}
    assert 'pack_field_empty' in codes


def test_pack_field_too_long() -> None:
    body = _valid_body()
    body['class_system'] = 'x' * 20_001
    report = validate_pack(None, body)
    codes = {e.code for e in report.errors}
    assert 'pack_field_too_long' in codes


def test_pack_name_invalid() -> None:
    report = validate_pack('Not Valid!', _valid_body())
    assert any(e.code == 'pack_name_invalid' for e in report.errors)


def test_pack_name_reserved() -> None:
    report = validate_pack('default', _valid_body())
    assert any(e.code == 'pack_name_reserved' for e in report.errors)


def test_pack_name_conflict_is_report_issue_not_exception() -> None:
    report = validate_pack('taken', _valid_body(), existing_names=frozenset({'taken'}))
    assert any(e.code == 'name_conflict' for e in report.errors)


def test_null_name_skips_name_checks() -> None:
    report = validate_pack(None, _valid_body(), existing_names=frozenset({'default'}))
    codes = {e.code for e in report.errors}
    assert 'name_conflict' not in codes
    assert 'pack_name_reserved' not in codes


def test_pack_placeholder_missing() -> None:
    body = _valid_body()
    body['class_user_template'] = 'no placeholder here'
    report = validate_pack(None, body)
    assert any(e.code == 'pack_placeholder_missing' for e in report.errors)


def test_pack_placeholder_unknown() -> None:
    body = _valid_body()
    body['class_user_template'] = 'Classes: {class_names_csv} also {unknown_thing}'
    report = validate_pack(None, body)
    assert any(e.code == 'pack_placeholder_unknown' for e in report.errors)


def test_pack_template_format_error_on_unbalanced_braces() -> None:
    body = _valid_body()
    body['class_user_template'] = 'Classes: {class_names_csv} { unbalanced'
    report = validate_pack(None, body)
    assert any(e.code == 'pack_template_format_error' for e in report.errors)


def test_pack_placeholder_in_verbatim_field_is_a_warning() -> None:
    body = _valid_body()
    body['region_user'] += ' Also mention {class_names_csv} here.'
    report = validate_pack(None, body)
    assert any(w.code == 'pack_placeholder_in_verbatim_field' for w in report.warnings)
    assert report.ok  # warnings never block


def test_pack_reply_key_missing() -> None:
    body = _valid_body()
    body['region_visible_user'] = 'Respond with a JSON object naming each image.'
    body['region_visible_system'] = 'no meaningful content here'
    report = validate_pack(None, body)
    codes = {
        (e.code, e.detail.get('key')) for e in report.errors if e.code == 'pack_reply_key_missing'
    }
    assert ('pack_reply_key_missing', 'results') in codes
    assert ('pack_reply_key_missing', 'visible') in codes


def test_pack_synonym_target_unknown_is_a_warning() -> None:
    body = _valid_body()
    body['synonyms'] = {'foo': 'not_a_real_class'}
    report = validate_pack(None, body, class_names=frozenset({'car', 'truck'}))
    assert any(w.code == 'pack_synonym_target_unknown' for w in report.warnings)


def test_pack_description_class_unknown_is_a_warning() -> None:
    body = _valid_body()
    body['class_descriptions'] = {'not_a_real_class': 'a description'}
    report = validate_pack(None, body, class_names=frozenset({'car', 'truck'}))
    assert any(w.code == 'pack_description_class_unknown' for w in report.warnings)


def test_pack_example_values_info() -> None:
    body = _valid_body()
    body['region_batch_user'] = 'Reply like: [{"img": 1, "is_region": true, "text": "ABC 123"}]'
    report = validate_pack(None, body)
    assert any(w.code == 'pack_example_values' and w.severity == 'info' for w in report.warnings)


def test_text_mode_consistency_warnings() -> None:
    from src.config import DetectionProfile

    body = _valid_body()
    text_free = DetectionProfile(name='p', text_reader='none')
    report = validate_pack(None, body, profile=text_free)
    assert any(w.code == 'pack_asks_text_profile_text_free' for w in report.warnings)

    text_free_pack = GENERIC_REGION_PACK.to_dict()
    text_free_pack.pop('name')
    reading_profile = DetectionProfile(name='p', text_reader='vlm')
    report2 = validate_pack(None, text_free_pack, profile=reading_profile)
    assert any(w.code == 'pack_no_text_profile_reads_text' for w in report2.warnings)
