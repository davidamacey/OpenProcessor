"""runner.py end to end over fake stacks: what it sends, what it reports, when it fails."""

from __future__ import annotations

import io
import json
from typing import TYPE_CHECKING, Any

import httpx
import pytest

from . import _tiny_contract as tiny, cases, runner
from ._fake_stack import FakeStack, json_answer
from .providers import Fixtures


if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

REAL = runner.Paths()


def ok_answer(request: httpx.Request) -> httpx.Response:
    return json_answer({'id': 1, 'name': 'thing', 'path': request.url.path})


def tiny_paths(tmp_path: Path, sub: str = '', **kwargs: Any) -> runner.Paths:
    """The tiny contract written under ``tmp_path/sub``; one directory holds one contract."""
    directory = tmp_path / sub if sub else tmp_path
    directory.mkdir(exist_ok=True)
    contracts, families_file = tiny.write(directory, **kwargs)
    return runner.Paths(contracts, families_file)


def settings(**kwargs: Any) -> runner.Settings:
    return runner.Settings(**{'seed': 1, 'max_examples': 4, **kwargs})


def go(
    reference: FakeStack | None,
    candidate: FakeStack | None,
    *,
    paths: runner.Paths,
    options: runner.Settings | None = None,
    providers: dict[str, Any] | None = None,
) -> tuple[runner.Summary, list[dict[str, Any]]]:
    out = io.StringIO()
    summary = runner.run(
        options or settings(),
        reference.client() if reference else None,
        candidate.client() if candidate else None,
        out,
        paths=paths,
        providers=providers if providers is not None else {},
    )
    return summary, [json.loads(line) for line in out.getvalue().splitlines()]


def case_lines(lines: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [line for line in lines if line['type'] == 'case']


def test_two_identical_stacks_give_an_empty_diff_and_a_clean_summary(tmp_path: Path) -> None:
    summary, lines = go(
        FakeStack('reference', ok_answer),
        FakeStack('candidate', ok_answer),
        paths=tiny_paths(tmp_path),
    )
    assert summary.ok
    assert runner.exit_code(summary) == 0
    assert lines[-1]['type'] == 'summary'
    assert lines[-1]['ok'] is True
    assert lines[-1]['operations']['in_scope'] == 3
    assert lines[-1]['cases']['different'] == 0
    assert lines[-1]['cases']['same'] == lines[-1]['cases']['total'] == len(case_lines(lines))
    assert {line['label'] for line in lines if line['type'] == 'operation'} == {
        'GET /things',
        'POST /things',
        'GET /things/{thing_id}',
    }


def test_every_case_is_sent_to_both_stacks_identically(tmp_path: Path) -> None:
    reference, candidate = FakeStack('reference', ok_answer), FakeStack('candidate', ok_answer)
    go(reference, candidate, paths=tiny_paths(tmp_path))
    assert reference.requests()
    assert reference.requests() == candidate.requests()


def test_a_difference_is_reported_with_everything_a_person_needs_to_act(tmp_path: Path) -> None:
    def drops_the_name(request: httpx.Request) -> httpx.Response:
        return json_answer({'id': 1, 'path': request.url.path})

    summary, lines = go(
        FakeStack('reference', ok_answer),
        FakeStack('candidate', drops_the_name),
        paths=tiny_paths(tmp_path),
    )
    assert not summary.ok
    assert runner.exit_code(summary) == 1
    case = next(c for c in case_lines(lines) if c['label'] == 'POST /things')
    assert case['outcome'] == 'different'
    assert case['case_id'].startswith('create_thing:')
    assert case['operation_id'] == 'create_thing'
    assert case['request']['method'] == 'POST'
    assert case['request']['path'] == '/things'
    assert case['request']['body'] is not None
    assert case['reference']['status'] == case['candidate']['status'] == 200
    first = case['differences'][0]
    assert (first['pointer'], first['kind'], first['reference'], first['candidate']) == (
        '/name',
        'missing_in_candidate',
        'thing',
        None,
    )
    assert any('differs at /name' in f for f in summary.findings)


def test_a_float_spelling_difference_passes_end_to_end(tmp_path: Path) -> None:
    summary, _ = go(
        FakeStack('reference', lambda _r: json_answer(None, raw='{"id": 1.0, "name": "a"}')),
        FakeStack('candidate', lambda _r: json_answer(None, raw='{"name": "a", "id": 1}')),
        paths=tiny_paths(tmp_path),
    )
    assert summary.ok


def test_a_status_difference_is_reported(tmp_path: Path) -> None:
    summary, lines = go(
        FakeStack('reference', ok_answer),
        FakeStack('candidate', lambda _r: json_answer({'detail': 'gone'}, 404)),
        paths=tiny_paths(tmp_path),
    )
    assert not summary.ok
    pointers = {d['pointer'] for c in case_lines(lines) for d in c['differences']}
    assert 'status' in pointers


def test_an_operation_with_no_case_fails_the_run_it_never_passes_silently(
    tmp_path: Path,
) -> None:
    paths = tiny_paths(tmp_path, impossible_query=True)
    summary, lines = go(
        FakeStack('reference', ok_answer),
        FakeStack('candidate', ok_answer),
        paths=paths,
        options=settings(modes=('positive',)),
    )
    assert not summary.ok
    assert runner.exit_code(summary) == 1
    assert any('GET /things' in label for label in summary.zero_case_operations)
    assert lines[-1]['operations']['zero_cases']


def test_a_generation_that_fails_is_recorded_and_fails_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = cases.generate

    def broken(operation: Any, *args: Any, **kwargs: Any) -> cases.Generated:
        if operation.label == 'POST /things':
            raise ValueError('cannot build a strategy')
        return real(operation, *args, **kwargs)

    monkeypatch.setattr(cases, 'generate', broken)
    summary, lines = go(
        FakeStack('reference', ok_answer),
        FakeStack('candidate', ok_answer),
        paths=tiny_paths(tmp_path),
    )
    assert not summary.ok
    assert summary.generation_errors == ['POST /things: ValueError: cannot build a strategy']
    assert any(line.get('error') for line in lines if line['type'] == 'operation')
    assert summary.total('cases') > 0, 'the other operations still ran'


def test_a_transport_error_on_one_stack_fails_the_run_and_is_recorded(tmp_path: Path) -> None:
    def refuses(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError('refused')

    summary, lines = go(
        FakeStack('reference', ok_answer),
        FakeStack('candidate', refuses),
        paths=tiny_paths(tmp_path),
    )
    assert not summary.ok
    errors = [c for c in case_lines(lines) if c['outcome'] == 'error']
    assert errors
    assert 'ConnectError' in errors[0]['candidate']['error']
    assert errors[0]['reference']['status'] == 200
    assert summary.total('transport_errors') == len(errors)


def test_the_candidate_is_checked_against_the_schema_when_asked(tmp_path: Path) -> None:
    def wrong_type(_request: httpx.Request) -> httpx.Response:
        return json_answer({'id': 'not-an-integer'})

    paths = tiny_paths(tmp_path)
    summary, lines = go(
        FakeStack('reference', wrong_type),
        FakeStack('candidate', wrong_type),
        paths=paths,
        options=settings(check_schema=True, modes=('positive',)),
    )
    assert not summary.ok, 'two stacks that agree on a wrong answer still fail the schema check'
    flagged = [c for c in case_lines(lines) if c['schema_failures']]
    assert flagged
    failure = flagged[0]['schema_failures'][0]
    assert failure['check'] == 'response_schema_conformance'
    assert failure['operation'].endswith('/things') or '/things/' in failure['operation']
    assert failure['schema_path']
    assert failure['instance_path'] == ['id']
    assert all(c['outcome'] == 'same' for c in case_lines(lines))


def test_the_schema_check_is_off_unless_asked_and_never_applies_to_the_reference(
    tmp_path: Path,
) -> None:
    summary, lines = go(
        FakeStack('reference', lambda _r: json_answer({'id': 'bad'})),
        FakeStack('candidate', lambda _r: json_answer({'id': 'bad'})),
        paths=tiny_paths(tmp_path),
    )
    assert summary.ok
    assert all(c['schema_failures'] == [] for c in case_lines(lines))
    # a conforming candidate passes with the check on although the reference does not conform
    summary, _ = go(
        FakeStack('reference', lambda _r: json_answer({'id': 'bad'})),
        FakeStack('candidate', ok_answer),
        paths=tiny_paths(tmp_path, 'again'),
        options=settings(check_schema=True, modes=('positive',)),
    )
    assert summary.total('schema_failures') == 0
    assert summary.total('different') > 0


def test_a_dry_run_sends_nothing_and_still_demands_cases(tmp_path: Path) -> None:
    summary, lines = go(None, None, paths=tiny_paths(tmp_path), options=settings(dry_run=True))
    assert summary.ok
    assert summary.total('cases') > 0
    assert case_lines(lines) == []
    positive_only, _ = go(
        None,
        None,
        paths=tiny_paths(tmp_path, 'b'),
        options=settings(dry_run=True, modes=('positive',)),
    )
    assert positive_only.ok


def test_a_dry_run_fails_for_an_operation_with_no_case(tmp_path: Path) -> None:
    summary, _ = go(
        None,
        None,
        paths=tiny_paths(tmp_path, impossible_query=True),
        options=settings(dry_run=True, modes=('positive',)),
    )
    assert not summary.ok


def test_both_stacks_are_needed_unless_it_is_a_dry_run(tmp_path: Path) -> None:
    with pytest.raises(runner.ConfigError, match='both stacks'):
        go(FakeStack('reference', ok_answer), None, paths=tiny_paths(tmp_path))


def test_the_same_seed_sends_the_same_requests_and_another_seed_others(tmp_path: Path) -> None:
    def sent(seed: int, sub: str) -> list[Any]:
        stack = FakeStack('reference', ok_answer)
        go(
            stack,
            FakeStack('candidate', ok_answer),
            paths=tiny_paths(tmp_path, sub),
            options=settings(seed=seed),
        )
        return stack.requests()

    first, again, other = sent(1, 'a'), sent(1, 'b'), sent(2, 'c')
    assert first == again
    assert first != other


def test_provider_values_become_cases_with_that_parameter_fixed(tmp_path: Path) -> None:
    class Things:
        name = 'things'

        def collect(self, reference: httpx.Client, candidate: httpx.Client) -> Fixtures:
            return Fixtures(
                {'thing_id': ['abc', 'xyz']}, ["thing_id 'q' is only in the candidate stack"]
            )

    reference = FakeStack('reference', ok_answer)
    summary, lines = go(
        reference,
        FakeStack('candidate', ok_answer),
        paths=tiny_paths(tmp_path, families=tiny.WITH_FIXTURES),
        providers={'things': Things()},
        options=settings(modes=('positive',)),
    )
    assert summary.ok
    fixture_cases = [c for c in case_lines(lines) if c['mode'] == 'fixture']
    assert [c['request']['path'] for c in fixture_cases] == [
        '/things/abc',
        '/things/xyz',
    ]  # the third binding repeats the first, an identical request, sent once
    assert any('only in the candidate stack' in note for note in summary.fixture_notes)


def test_a_provider_that_fails_fails_the_run_instead_of_running_without_fixtures(
    tmp_path: Path,
) -> None:
    from .providers import ProviderError

    class Broken:
        name = 'things'

        def collect(self, reference: httpx.Client, candidate: httpx.Client) -> Fixtures:
            raise ProviderError('reference: GET /things answered 500')

    summary, _ = go(
        FakeStack('reference', ok_answer),
        FakeStack('candidate', ok_answer),
        paths=tiny_paths(tmp_path, families=tiny.WITH_FIXTURES),
        providers={'things': Broken()},
    )
    assert not summary.ok
    assert 'answered 500' in summary.fixture_errors[0]


def test_a_family_that_names_a_missing_provider_cannot_start(tmp_path: Path) -> None:
    paths = tiny_paths(tmp_path, families=tiny.WITH_FIXTURES)
    with pytest.raises(runner.ConfigError, match='no such provider'):
        go(FakeStack('reference', ok_answer), FakeStack('candidate', ok_answer), paths=paths)


def test_the_scope_can_be_narrowed_by_family_and_method(tmp_path: Path) -> None:
    paths = tiny_paths(tmp_path)
    summary, lines = go(
        FakeStack('reference', ok_answer),
        FakeStack('candidate', ok_answer),
        paths=paths,
        options=settings(families=('things',), methods=frozenset({'GET'})),
    )
    assert summary.ok
    assert summary.total('operations') == 2
    assert {c['request']['method'] for c in case_lines(lines)} == {'GET'}
    with pytest.raises(runner.ConfigError, match='unknown family'):
        go(None, None, paths=paths, options=settings(families=('nope',), dry_run=True))
    with pytest.raises(runner.ConfigError, match='nothing is in scope'):
        go(None, None, paths=paths, options=settings(methods=frozenset({'TRACE'}), dry_run=True))


def test_a_mapping_that_misses_an_operation_stops_the_run(tmp_path: Path) -> None:
    paths = tiny_paths(tmp_path, families=tiny.FAMILIES.replace('"/things/*"', '"/things"'))
    with pytest.raises(runner.ConfigError, match='does not cover the contracts'):
        go(None, None, paths=paths, options=settings(dry_run=True))


def test_writes_are_sent_after_reads_and_deletes_last() -> None:
    def op(method: str, path: str) -> runner.families.Operation:
        return runner.families.Operation('c', method, path, 'x')

    ordered = sorted(
        [op('DELETE', '/a'), op('POST', '/a'), op('GET', '/z'), op('PUT', '/b'), op('GET', '/a')],
        key=runner.order_key,
    )
    assert [(o.method, o.path) for o in ordered] == [
        ('GET', '/a'),
        ('GET', '/z'),
        ('POST', '/a'),
        ('PUT', '/b'),
        ('DELETE', '/a'),
    ]


class Endless(httpx.SyncByteStream):
    """An event stream that has no end: a heartbeat, then one event after another."""

    def __init__(self, produced: list[int]) -> None:
        self.produced = produced

    def __iter__(self) -> Iterator[bytes]:
        yield b': heartbeat\n\n'
        for n in range(10_000):
            self.produced.append(n)
            yield f'data: {n}\n\n'.encode()


def test_an_event_stream_is_read_up_to_its_first_event_not_to_its_end(tmp_path: Path) -> None:
    contract = tiny.contract()
    contract['paths']['/things']['get']['responses']['200']['content'] = {
        'text/event-stream': {'schema': {'type': 'string'}}
    }
    (tmp_path / 'contracts').mkdir()
    (tmp_path / 'contracts' / 'tiny.json').write_text(json.dumps(contract))
    (tmp_path / 'families.toml').write_text(tiny.FAMILIES)
    produced: list[int] = []

    def stream(request: httpx.Request) -> httpx.Response:
        if request.url.path == '/things' and request.method == 'GET':
            return httpx.Response(
                200, headers={'content-type': 'text/event-stream'}, stream=Endless(produced)
            )
        return ok_answer(request)

    summary, lines = go(
        FakeStack('reference', stream),
        FakeStack('candidate', stream),
        paths=runner.Paths(tmp_path / 'contracts', tmp_path / 'families.toml'),
        options=settings(modes=('positive',), methods=frozenset({'GET'}), families=('things',)),
    )
    events = [c for c in case_lines(lines) if c['label'] == 'GET /things']
    assert events
    assert all(c['reference']['media_type'] == 'text/event-stream' for c in events)
    assert all(0 < c['reference']['bytes'] < 100 for c in events), 'stopped at the first event'
    assert len(produced) < 200
    assert summary.ok


def test_the_summary_names_the_library_versions_the_cases_depend_on(tmp_path: Path) -> None:
    _, lines = go(None, None, paths=tiny_paths(tmp_path), options=settings(dry_run=True))
    versions = lines[-1]['versions']
    assert set(versions) == {'schemathesis', 'hypothesis', 'httpx'}
    assert versions['schemathesis'] == '4.33.4'


def test_the_real_contract_runs_end_to_end_for_one_family_over_fake_stacks() -> None:
    def not_found(_request: httpx.Request) -> httpx.Response:
        return json_answer({'detail': 'Not Found'}, 404)

    summary, lines = go(
        FakeStack('reference', not_found),
        FakeStack('candidate', not_found),
        paths=REAL,
        options=settings(families=('jobs',), max_examples=3),
    )
    assert summary.ok
    assert summary.total('operations') == 3
    assert len(case_lines(lines)) == summary.total('cases') >= 3


def test_the_command_line_writes_json_lines_and_a_short_summary(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / 'run.jsonl'
    code = runner.main(
        ['--dry-run', '--families', 'jobs', '--max-examples', '2', '--out', str(out)]
    )
    assert code == 0
    lines = [json.loads(line) for line in out.read_text().splitlines()]
    assert lines[-1]['type'] == 'summary'
    assert lines[-1]['operations']['in_scope'] == 3
    shown = capsys.readouterr().out
    assert 'RESULT: OK' in shown
    assert str(out) in shown


def test_the_command_line_exits_two_when_the_run_cannot_start(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert runner.main(['--dry-run', '--families', 'nope', '--out', str(tmp_path / 'x.jsonl')]) == 2
    assert 'unknown family' in capsys.readouterr().err
    assert runner.main(['--out', str(tmp_path / 'y.jsonl')]) == 2
    assert runner.main(['--dry-run', '--modes', 'weird', '--out', str(tmp_path / 'z.jsonl')]) == 2
