"""cases.py: deterministic, complete, stack-independent request generation."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from functools import cache
from pathlib import Path

import httpx
import pytest

from . import cases, families as fam


REPO = Path(__file__).resolve().parents[2]
CURATION = fam.CONTRACT_DIR / 'curation.json'
SMALL = ['GET /curation/health', 'GET /curation/projects', 'POST /curation/projects']


@cache
def schema_operations() -> dict[str, object]:
    return {op.label: op for op in cases.operations_of(cases.load_schema(CURATION))}


def build(label: str, *, seed: int = 1, n: int = 6, **extra: object) -> cases.Generated:
    return cases.generate(
        schema_operations()[label],  # type: ignore[arg-type]
        'projects',
        seed_value=seed,
        max_examples=n,
        **extra,  # type: ignore[arg-type]
    )


def fingerprints(label: str, **kwargs: object) -> list[str]:
    return [c.fingerprint for c in build(label, **kwargs).cases]  # type: ignore[arg-type]


def test_every_operation_of_the_contract_loads_with_a_unique_operation_id() -> None:
    operations = cases.operations_of(cases.load_schema(CURATION))
    ids = [cases.operation_id_of(op) for op in operations]
    assert len(ids) == len([o for o in fam.load_operations() if o.contract == 'curation'])
    assert len(set(ids)) == len(ids)


@pytest.mark.parametrize('label', SMALL)
def test_an_operation_has_cases(label: str) -> None:
    assert build(label).cases


def test_the_same_seed_gives_the_same_cases_in_one_process() -> None:
    for label in SMALL:
        assert fingerprints(label) == fingerprints(label)
        assert fingerprints(label, modes=('negative',)) == fingerprints(label, modes=('negative',))


def test_the_same_seed_gives_the_same_cases_in_another_process_with_another_hash_seed() -> None:
    code = f"""
import json, pathlib, sys
sys.path.insert(0, {str(REPO / 'tests')!r})
from differential import cases
operations = {{
    o.label: o for o in cases.operations_of(cases.load_schema(pathlib.Path({str(CURATION)!r})))
}}
out = {{}}
for label in {SMALL!r}:
    generated = cases.generate(operations[label], 'x', seed_value=1, max_examples=6)
    out[label] = [c.fingerprint for c in generated.cases]
print(json.dumps(out))
"""
    answers = []
    for hash_seed in ('0', '4242'):
        done = subprocess.run(
            [sys.executable, '-c', code],
            capture_output=True,
            text=True,
            check=True,
            env={**os.environ, 'PYTHONHASHSEED': hash_seed},
            cwd=REPO,
        )
        answers.append(json.loads(done.stdout.splitlines()[-1]))
    assert answers[0] == answers[1]
    assert answers[0] == {label: fingerprints(label) for label in SMALL}


def test_another_seed_gives_other_cases() -> None:
    assert fingerprints('POST /curation/projects', seed=1) != fingerprints(
        'POST /curation/projects', seed=2
    )


def test_an_operation_draws_the_same_cases_whatever_was_generated_before_it() -> None:
    alone = fingerprints('POST /curation/projects')
    for label in SMALL:
        build(label, seed=9)
    assert fingerprints('POST /curation/projects') == alone


def test_nothing_random_of_schemathesis_reaches_the_request() -> None:
    for case in build('POST /curation/projects').cases:
        names = {name for name, _ in case.headers}
        assert not names & {
            'x-schemathesis-testcaseid',
            'user-agent',
            'accept-encoding',
            'connection',
        }
        assert case.request_kwargs()['headers'][cases.CASE_HEADER] == case.case_id
        assert case.case_id.startswith(f'{case.operation_id}:')


def test_case_ids_are_unique_within_an_operation() -> None:
    ids = [c.case_id for c in build('POST /curation/projects', n=12).cases]
    assert len(set(ids)) == len(ids)


def test_a_generated_request_builds_for_a_client_with_a_base_url() -> None:
    client = httpx.Client(base_url='http://stack.invalid:4603')
    for label in SMALL:
        for case in build(label).cases:
            request = client.build_request(case.method, case.path, **case.request_kwargs())
            assert str(request.url).startswith('http://stack.invalid:4603/curation/')


def test_negative_data_is_generated_where_the_schema_allows_it() -> None:
    generated = build('POST /curation/projects', modes=('negative',))
    assert generated.cases
    assert {c.mode for c in generated.cases} == {'negative'}


def test_an_operation_with_nothing_to_negate_says_so_instead_of_failing() -> None:
    generated = build('GET /curation/health', modes=('negative',))
    assert generated.cases == []
    assert any('negative' in note for note in generated.notes)


def test_fixture_values_are_kept_and_the_rest_is_generated() -> None:
    label = 'GET /curation/projects/{project}/images'
    generated = build(label, modes=(), bindings=[{'project': 'alpha'}, {'project': 'beta'}])
    assert [c.mode for c in generated.cases] == ['fixture', 'fixture']
    assert [c.path for c in generated.cases] == [
        '/curation/projects/alpha/images',
        '/curation/projects/beta/images',
    ]
    assert generated.cases[0].fingerprint != generated.cases[1].fingerprint


def test_a_binding_for_a_parameter_the_operation_lacks_adds_no_case() -> None:
    generated = build('GET /curation/health', modes=(), bindings=[{'project': 'alpha'}])
    assert generated.cases == []


def test_json_bodies_are_kept_as_data_and_summarised_for_a_report() -> None:
    case = build('POST /curation/projects').cases[0]
    assert case.body is not None
    assert case.body[0] in {'json', 'content'}
    json.dumps(case.summary(), default=str)


def test_an_unsendable_request_is_dropped_with_a_note_not_sent_half_built(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def refuse(*_a: object, **_k: object) -> None:
        raise UnicodeEncodeError('utf-8', '\ud800', 0, 1, 'surrogates not allowed')

    monkeypatch.setattr(cases.httpx, 'Request', refuse)
    generated = build('GET /curation/health')
    assert generated.cases == []
    assert any('cannot be put on the wire' in note for note in generated.notes)


def test_literals_of_other_local_modules_do_not_steer_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import importlib

    from hypothesis.internal.conjecture import providers

    marker = 'zq_marker_local_7'  # Hypothesis keeps string literals of up to 20 characters
    before = fingerprints('POST /curation/projects', n=30)
    (tmp_path / 'zq_local_module.py').write_text(f'VALUE = {marker!r}\n', encoding='utf-8')
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.import_module('zq_local_module')
    try:
        # the premise: Hypothesis's pool of local constants now holds the new module's literal
        assert marker in providers._get_local_constants().strings
        with cases.without_local_constants():
            assert not providers._get_local_constants().strings
        assert providers._get_local_constants().strings
        assert fingerprints('POST /curation/projects', n=30) == before
    finally:
        sys.modules.pop('zq_local_module', None)
