"""compare.py: what passes, what fails, and how a difference is reported."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

from . import compare as cmp


if TYPE_CHECKING:
    from pathlib import Path

DEFAULTS = cmp.load_normalisation()


def rule(kind: str, pointer: str, **extra: Any) -> dict[str, Any]:
    return {'kind': kind, 'pointer': pointer, 'reason': 'test', **extra}


def norm(*entries: dict[str, Any]) -> cmp.Normalisation:
    return cmp.Normalisation(cmp.parse_rules(list(entries), 'test'))


def answer(
    body: Any,
    status: int = 200,
    media: str = 'application/json',
    headers: dict[str, str] | None = None,
) -> cmp.Observed:
    raw = body if isinstance(body, bytes) else json.dumps(body).encode()
    return cmp.Observed.of(status, {'Content-Type': media, **(headers or {})}, raw)


def raw_json(text: str, status: int = 200) -> cmp.Observed:
    return answer(text.encode(), status)


def diffs(
    a: cmp.Observed, b: cmp.Observed, normalisation: cmp.Normalisation = DEFAULTS
) -> list[cmp.Difference]:
    return cmp.compare_observed(a, b, normalisation)


# --- what passes -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ('reference', 'candidate'),
    [
        ('{"a": 1.0}', '{"a": 1}'),
        ('{"a": 0.5}', '{"a": 5e-1}'),
        ('{"a": 100}', '{"a": 1.0E2}'),
        ('{"a": [1, 2.50]}', '{"a": [1.0, 2.5]}'),
        ('{"a": 1, "b": 2}', '{"b": 2, "a": 1}'),
        ('{"a": {"x": 1, "y": 2}}', '{"a": {"y": 2, "x": 1}}'),
    ],
)
def test_spelling_and_key_order_differences_pass(reference: str, candidate: str) -> None:
    assert diffs(raw_json(reference), raw_json(candidate)) == []


def test_header_case_and_unlisted_headers_do_not_matter() -> None:
    a = answer({}, headers={'Server': 'uvicorn', 'Allow': 'GET', 'X-Request-Id': '1'})
    b = answer({}, headers={'server': 'axum', 'ALLOW': 'GET', 'x-request-id': '2'})
    assert diffs(a, b) == []


def test_charset_parameter_of_the_content_type_is_not_compared() -> None:
    assert diffs(answer({}), answer({}, media='application/json; charset=utf-8')) == []


# --- what fails ------------------------------------------------------------------------------


def test_a_missing_key_fails_with_its_pointer_and_value() -> None:
    found = diffs(answer({'a': 1, 'b': {'c': 2}}), answer({'a': 1, 'b': {}}))
    assert [(d.pointer, d.kind, d.reference, d.candidate) for d in found] == [
        ('/b/c', 'missing_in_candidate', 2, None)
    ]


def test_an_extra_key_fails() -> None:
    found = diffs(answer({'a': 1}), answer({'a': 1, 'z': True}))
    assert [(d.pointer, d.kind) for d in found] == [('/z', 'extra_in_candidate')]


def test_a_changed_value_fails() -> None:
    found = diffs(answer({'a': [1, 2]}), answer({'a': [1, 3]}))
    assert [(d.pointer, d.reference, d.candidate) for d in found] == [('/a/1', 2, 3)]


def test_a_short_array_fails_with_the_lengths() -> None:
    found = diffs(answer({'a': [1, 2]}), answer({'a': [1]}))
    assert [(d.pointer, d.kind, d.reference, d.candidate) for d in found] == [
        ('/a', 'length', 2, 1)
    ]


@pytest.mark.parametrize(
    ('reference', 'candidate'),
    [(True, 1), (False, 0), (1, '1'), (None, 0), (None, ''), ('a', ['a'])],
)
def test_values_of_another_json_type_fail_even_when_python_would_call_them_equal(
    reference: Any, candidate: Any
) -> None:
    found = diffs(answer({'a': reference}), answer({'a': candidate}))
    assert [d.kind for d in found] == ['type']


def test_nested_pointers_are_escaped_as_rfc_6901() -> None:
    found = diffs(answer({'a/b': {'c~d': 1}}), answer({'a/b': {'c~d': 2}}))
    assert found[0].pointer == '/a~1b/c~0d'


def test_status_and_content_type_are_always_compared() -> None:
    assert [d.pointer for d in diffs(answer({}), answer({}, status=404))] == ['status']
    assert [d.pointer for d in diffs(answer({}), answer({}, media='text/plain'))] == [
        'content_type'
    ]


def test_a_listed_header_that_differs_fails() -> None:
    found = diffs(
        answer({}, headers={'Retry-After': '5'}), answer({}, headers={'Retry-After': '9'})
    )
    assert [(d.pointer, d.reference, d.candidate) for d in found] == [
        ('header:retry-after', '5', '9')
    ]


def test_non_json_bodies_compare_exactly_and_report_a_digest() -> None:
    a = answer(b'\xff\xd8abc', media='image/jpeg')
    assert diffs(a, answer(b'\xff\xd8abc', media='image/jpeg')) == []
    found = diffs(a, answer(b'\xff\xd8abd', media='image/jpeg'))
    assert [d.pointer for d in found] == ['body']
    assert 'sha256:' in str(found[0].reference)


def test_a_body_that_is_not_valid_json_is_compared_as_bytes() -> None:
    assert diffs(raw_json('{oops'), raw_json('{oops')) == []
    assert [d.pointer for d in diffs(raw_json('{oops'), raw_json('{oops!'))] == ['body']


# --- sets, ids, times: only when listed ------------------------------------------------------


def test_a_set_in_another_order_passes_only_when_it_is_listed() -> None:
    a, b = answer({'tags': ['x', 'y', 'z']}), answer({'tags': ['z', 'x', 'y']})
    assert diffs(a, b)
    assert diffs(a, b, norm(rule('unordered', '/tags'))) == []
    # a listed set still compares its members
    assert diffs(a, answer({'tags': ['z', 'x', 'q']}), norm(rule('unordered', '/tags')))


def test_an_unordered_array_of_objects_is_sorted_by_the_key() -> None:
    a = answer({'rows': [{'id': 'b', 'n': 1}, {'id': 'a', 'n': 2}]})
    b = answer({'rows': [{'id': 'a', 'n': 2}, {'id': 'b', 'n': 1}]})
    assert diffs(a, b, norm(rule('unordered', '/rows', key='/id'))) == []
    swapped = answer({'rows': [{'id': 'a', 'n': 1}, {'id': 'b', 'n': 2}]})
    assert diffs(a, swapped, norm(rule('unordered', '/rows', key='/id')))


def test_generated_ids_may_differ_but_the_pattern_of_reuse_may_not() -> None:
    rules = norm(rule('opaque_id', '/items/*/id'), rule('opaque_id', '/items/*/parent'))
    a = answer({'items': [{'id': 'p1', 'parent': None}, {'id': 'c1', 'parent': 'p1'}]})
    b = answer({'items': [{'id': 'X', 'parent': None}, {'id': 'Y', 'parent': 'X'}]})
    assert diffs(a, b, rules) == []
    broken = answer({'items': [{'id': 'X', 'parent': None}, {'id': 'Y', 'parent': 'Z'}]})
    assert [d.pointer for d in diffs(a, broken, rules)] == ['/items/1/parent']
    assert diffs(a, b) != []  # unlisted, the ids are compared


def test_an_id_that_is_missing_still_fails_when_ids_are_opaque() -> None:
    rules = norm(rule('opaque_id', '/id'))
    assert [d.kind for d in diffs(answer({'id': 'a'}), answer({}), rules)] == [
        'missing_in_candidate'
    ]


def test_timestamps_may_differ_but_must_still_be_timestamps() -> None:
    rules = norm(rule('timestamp', '/at'))
    assert (
        diffs(
            answer({'at': '2026-01-01T00:00:00Z'}),
            answer({'at': '2026-11-16T10:00:00+00:00'}),
            rules,
        )
        == []
    )
    assert diffs(answer({'at': '2026-01-01T00:00:00Z'}), answer({'at': 'yesterday'}), rules)
    assert diffs(answer({'at': 1.5}), answer({'at': 2}), rules) == []


def test_ignore_value_keeps_presence_and_type() -> None:
    rules = norm(rule('ignore_value', '/ms'))
    assert diffs(answer({'ms': 1.2}), answer({'ms': 99}), rules) == []
    assert diffs(answer({'ms': 1.2}), answer({'ms': 'slow'}), rules)
    assert diffs(answer({'ms': 1.2}), answer({}), rules)


def test_approx_accepts_a_small_difference_only() -> None:
    rules = norm(rule('approx', '/score', rel=1e-3))
    assert diffs(answer({'score': 0.9000}), answer({'score': 0.9005}), rules) == []
    assert diffs(answer({'score': 0.9}), answer({'score': 0.95}), rules)
    assert diffs(answer({'score': 0.9}), answer({'score': '0.9'}), rules)


def test_drop_removes_an_intended_change_from_both_sides() -> None:
    rules = norm(rule('drop', '/legacy'))
    assert diffs(answer({'a': 1, 'legacy': 5}), answer({'a': 1}), rules) == []
    assert diffs(answer({'a': 1}), answer({'a': 2}), rules)


def test_a_wildcard_matches_any_key_and_index() -> None:
    rules = norm(rule('ignore_value', '/*/n/*'))
    assert diffs(answer({'x': {'n': [1, 2]}}), answer({'x': {'n': [3, 4]}}), rules) == []


# --- the shared defaults ---------------------------------------------------------------------


def validation(msg: str, loc: str = 'query', status: int = 422) -> cmp.Observed:
    return answer({'detail': [{'loc': ['query', loc], 'msg': msg, 'type': 'x'}]}, status)


def test_the_message_text_of_a_validation_error_may_differ() -> None:
    assert diffs(validation('Field required'), validation('missing field')) == []


def test_its_loc_and_type_may_not() -> None:
    assert diffs(validation('m', loc='a'), validation('m', loc='b'))


def test_a_pattern_message_must_match_word_for_word() -> None:
    a = validation("String should match pattern '^[a-z]+$'")
    assert diffs(a, validation("String should match pattern '^[a-z]*$'"))
    assert diffs(a, validation('Does not match'))
    assert diffs(a, validation("String should match pattern '^[a-z]+$'")) == []


def test_the_message_rule_holds_only_for_the_status_it_names() -> None:
    body = {'detail': [{'loc': ['x'], 'msg': 'one', 'type': 't'}]}
    other = {'detail': [{'loc': ['x'], 'msg': 'two', 'type': 't'}]}
    assert diffs(answer(body, 400), answer(other, 400))


# --- loading and reporting -------------------------------------------------------------------


@pytest.mark.parametrize(
    'entry',
    [
        {'kind': 'drop', 'pointer': '/a'},
        {'kind': 'drop', 'pointer': '/a', 'reason': '  '},
        {'kind': 'whatever', 'pointer': '/a', 'reason': 'r'},
        {'kind': 'drop', 'reason': 'r'},
        {'kind': 'drop', 'pointer': 'a', 'reason': 'r'},
    ],
)
def test_a_rule_without_a_reason_a_known_kind_or_a_pointer_is_refused(
    entry: dict[str, Any],
) -> None:
    with pytest.raises(ValueError, match='normalise'):
        cmp.parse_rules([entry], 'normalise/x.toml')


def test_a_family_file_adds_to_the_defaults_and_cannot_remove_from_them(tmp_path: Path) -> None:
    (tmp_path / '_defaults.toml').write_text(
        'compare_headers = ["allow"]\n[[rule]]\nkind = "drop"\npointer = "/a"\nreason = "shared"\n'
    )
    (tmp_path / 'demo.toml').write_text(
        'compare_headers = ["etag"]\n[[rule]]\nkind = "drop"\npointer = "/b"\nreason = "demo"\n'
    )
    merged = cmp.load_normalisation('demo', tmp_path)
    assert merged.compare_headers == ('allow', 'etag')
    assert [r.pointer for r in merged.rules] == [('a',), ('b',)]
    assert cmp.load_normalisation('absent', tmp_path).compare_headers == ('allow',)


def test_the_committed_defaults_load_and_every_rule_has_a_reason() -> None:
    assert DEFAULTS.rules
    assert all(r.reason for r in DEFAULTS.rules)


def test_a_difference_reads_as_plain_data_and_long_values_are_shortened() -> None:
    long = 'x' * 1000
    entry = cmp.Difference('/a', 'changed', long, 'y').asdict()
    assert entry['pointer'] == '/a'
    assert entry['candidate'] == 'y'
    assert len(str(entry['reference'])) < 400
    json.dumps(entry)


def test_the_number_of_reported_differences_is_bounded() -> None:
    a = answer({'rows': list(range(500))})
    b = answer({'rows': list(range(1, 501))})
    assert len(diffs(a, b)) == cmp.MAX_DIFFERENCES
