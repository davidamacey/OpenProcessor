"""Send the same generated requests to the reference stack and the candidate stack, and report.

    .venv/bin/python -m tests.differential.runner --reference http://127.0.0.1:4603 \\
        --candidate http://127.0.0.1:4613 [--families projects,jobs] [--check-schema]

Every operation of the selected families gets cases (``cases.py``); each case is sent to both stacks and
the two answers are compared by parsed JSON under the family's normalisation list (``compare.py``). With
``--check-schema`` the candidate's answers are also checked against the committed OpenAPI schema
(Schemathesis's ``response_schema_conformance``); the reference's never are, since it generated the schema.

Results go to a JSON-lines file, one line per operation, per case and a last ``summary`` line. The exit
code is 0 only when nothing differed and nothing is missing: 1 on any difference, any operation with no
case, any transport error, generation error, fixture error or schema failure; 2 when the run cannot start
(a family name that does not exist, ``families.toml`` not covering the contracts, nothing in scope).

The stacks must be disposable: a write operation is sent like any other, and ``DELETE`` last in each
family, so a project seen by the fixtures provider can be gone afterwards. ``--dry-run`` generates the
cases and sends nothing; it is how "every operation has cases" is checked without a stack.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, TextIO

import httpx
import hypothesis
import schemathesis
from hypothesis.errors import HypothesisException
from schemathesis.core.errors import SchemathesisError

from . import cases, compare, families
from .providers import PROVIDERS, ProviderError
from .send import Sent, schema_failures, send


if TYPE_CHECKING:
    from collections.abc import Mapping

    from .providers import Provider

READS = frozenset({'GET', 'HEAD', 'OPTIONS'})
FIRST_DIFFERENCES = 5
MAX_FINDINGS = 20


class ConfigError(Exception):
    """The run cannot start as asked."""


@dataclass(frozen=True)
class Settings:
    seed: int = 0
    max_examples: int = 8
    modes: tuple[str, ...] = cases.MODES
    families: tuple[str, ...] | None = None
    methods: frozenset[str] | None = None
    check_schema: bool = False
    dry_run: bool = False
    fixture_cases: int = 3
    stream_limit: int = 16384
    stream_seconds: float = 3.0


@dataclass(frozen=True)
class Paths:
    contract_dir: Path = families.CONTRACT_DIR
    families_file: Path = families.FAMILIES_FILE
    normalise_dir: Path = compare.NORMALISE_DIR


@dataclass
class Tally:
    operations: int = 0
    cases: int = 0
    same: int = 0
    different: int = 0
    transport_errors: int = 0
    schema_failures: int = 0


@dataclass
class Summary:
    settings: Settings
    tallies: dict[str, Tally] = field(default_factory=dict)
    zero_case_operations: list[str] = field(default_factory=list)
    generation_errors: list[str] = field(default_factory=list)
    fixture_errors: list[str] = field(default_factory=list)
    fixture_notes: list[str] = field(default_factory=list)
    findings: list[str] = field(default_factory=list)

    def total(self, name: str) -> int:
        return sum(getattr(t, name) for t in self.tallies.values())

    @property
    def ok(self) -> bool:
        return (
            self.total('operations') > 0
            and not self.zero_case_operations
            and not self.generation_errors
            and not self.fixture_errors
            and not self.total('different')
            and not self.total('transport_errors')
            and not self.total('schema_failures')
        )

    def asdict(self) -> dict[str, Any]:
        s = self.settings
        return {
            'scope': {
                'families': sorted(self.tallies),
                'methods': sorted(s.methods) if s.methods else None,
                'seed': s.seed,
                'max_examples': s.max_examples,
                'modes': list(s.modes),
                'fixture_cases': s.fixture_cases,
                'dry_run': s.dry_run,
                'check_schema': s.check_schema,
            },
            'versions': {
                'schemathesis': schemathesis.__version__,
                'hypothesis': hypothesis.__version__,
                'httpx': httpx.__version__,
            },
            'operations': {
                'in_scope': self.total('operations'),
                'zero_cases': self.zero_case_operations,
                'generation_errors': self.generation_errors,
            },
            'cases': {
                'total': self.total('cases'),
                'same': self.total('same'),
                'different': self.total('different'),
                'transport_errors': self.total('transport_errors'),
                'schema_failures': self.total('schema_failures'),
            },
            'families': {name: vars(t) for name, t in sorted(self.tallies.items())},
            'fixtures': {'errors': self.fixture_errors, 'notes': self.fixture_notes},
            'ok': self.ok,
        }


@dataclass
class Context:
    settings: Settings
    paths: Paths
    reference: httpx.Client | None
    candidate: httpx.Client | None
    providers: Mapping[str, Provider]
    out: TextIO
    summary: Summary

    def emit(self, record: dict[str, Any]) -> None:
        self.out.write(json.dumps(record, sort_keys=True, default=str) + '\n')


def plan(
    settings: Settings, paths: Paths
) -> list[tuple[families.Family, list[tuple[families.Operation, Any]]]]:
    """The families in scope with their operations, each paired with its Schemathesis operation."""
    operations = families.load_operations(paths.contract_dir)
    known = families.load_families(paths.families_file)
    problems = families.coverage_problems(operations, known)
    if problems:
        raise ConfigError('families.toml does not cover the contracts: ' + '; '.join(problems[:5]))
    names = {f.name for f in known}
    unknown = set(settings.families or ()) - names
    if unknown:
        raise ConfigError(f'unknown family: {", ".join(sorted(unknown))}')
    schemas = {
        c: cases.load_schema(paths.contract_dir / f'{c}.json')
        for c in {o.contract for o in operations}
    }
    lookup = {
        (c, op.method.upper(), op.path): op
        for c, s in schemas.items()
        for op in cases.operations_of(s)
    }
    grouped = families.by_family(operations, known)
    chosen = []
    for family in known:
        if settings.families and family.name not in settings.families:
            continue
        paired = [
            (o, lookup[(o.contract, o.method, o.path)])
            for o in sorted(grouped[family.name], key=order_key)
            if not settings.methods or o.method in settings.methods
        ]
        if paired:
            chosen.append((family, paired))
    if not chosen:
        raise ConfigError(
            'nothing is in scope: no operation matches the families and methods asked for'
        )
    return chosen


def order_key(operation: families.Operation) -> tuple[int, str, str]:
    rank = 0 if operation.method in READS else (2 if operation.method == 'DELETE' else 1)
    return (rank, operation.path, operation.method)


def run(
    settings: Settings,
    reference: httpx.Client | None,
    candidate: httpx.Client | None,
    out: TextIO,
    *,
    paths: Paths | None = None,
    providers: Mapping[str, Provider] | None = None,
) -> Summary:
    """Run the selected families; the clients may be None only for a dry run."""
    paths = paths or Paths()
    if not settings.dry_run and (reference is None or candidate is None):
        raise ConfigError('both stacks are needed unless --dry-run')
    chosen = plan(settings, paths)
    ctx = Context(
        settings,
        paths,
        reference,
        candidate,
        PROVIDERS if providers is None else providers,
        out,
        Summary(settings),
    )
    for family, operations in chosen:
        run_family(ctx, family, operations)
    ctx.emit({'type': 'summary', **ctx.summary.asdict()})
    return ctx.summary


def run_family(
    ctx: Context, family: families.Family, operations: list[tuple[families.Operation, Any]]
) -> None:
    tally = ctx.summary.tallies.setdefault(family.name, Tally())
    values = fixture_values(ctx, family)
    bindings = [
        {name: found[k % len(found)] for name, found in values.items() if found}
        for k in range(ctx.settings.fixture_cases)
    ]
    bindings = [b for b in bindings if b]
    norm = compare.load_normalisation(family.name, ctx.paths.normalise_dir)
    for operation, api_operation in operations:
        tally.operations += 1
        generated = generate_for(ctx, family, api_operation, bindings)
        if generated is None:
            continue
        ctx.emit(operation_record(generated))
        if not generated.cases:
            ctx.summary.zero_case_operations.append(f'{operation.label} ({operation.operation_id})')
            continue
        tally.cases += len(generated.cases)
        if not ctx.settings.dry_run:
            for case in generated.cases:
                run_case(ctx, tally, norm, case, api_operation)


def fixture_values(ctx: Context, family: families.Family) -> dict[str, list[str]]:
    if ctx.settings.dry_run or not family.fixtures:
        return {}
    assert ctx.reference is not None
    assert ctx.candidate is not None
    values: dict[str, list[str]] = {}
    for name in family.fixtures:
        if name not in ctx.providers:
            raise ConfigError(f'family {family.name} lists fixtures {name!r}: no such provider')
        try:
            found = ctx.providers[name].collect(ctx.reference, ctx.candidate)
        except ProviderError as error:
            ctx.summary.fixture_errors.append(f'{family.name}/{name}: {error}')
            continue
        values.update(found.values)
        ctx.summary.fixture_notes += [f'{family.name}/{name}: {note}' for note in found.notes]
    return values


def generate_for(
    ctx: Context, family: families.Family, api_operation: Any, bindings: list[dict[str, str]]
) -> cases.Generated | None:
    try:
        return cases.generate(
            api_operation,
            family.name,
            seed_value=ctx.settings.seed,
            max_examples=ctx.settings.max_examples,
            modes=ctx.settings.modes,
            bindings=bindings,
        )
    except (HypothesisException, SchemathesisError, ValueError, TypeError) as error:
        message = f'{api_operation.label}: {type(error).__name__}: {error}'
        ctx.summary.generation_errors.append(message)
        ctx.emit(
            {
                'type': 'operation',
                'label': api_operation.label,
                'family': family.name,
                'error': message,
            }
        )
        return None


def operation_record(generated: cases.Generated) -> dict[str, Any]:
    by_mode: dict[str, int] = {}
    for case in generated.cases:
        by_mode[case.mode] = by_mode.get(case.mode, 0) + 1
    return {
        'type': 'operation',
        'operation_id': generated.operation_id,
        'label': generated.label,
        'family': generated.family,
        'cases': len(generated.cases),
        'by_mode': by_mode,
        'fingerprints': [case.fingerprint for case in generated.cases],
        'notes': generated.notes,
    }


def run_case(
    ctx: Context,
    tally: Tally,
    norm: compare.Normalisation,
    case: cases.DiffCase,
    api_operation: Any,
) -> None:
    assert ctx.reference is not None
    assert ctx.candidate is not None
    stream = cases.declares_event_stream(api_operation)

    def ask(client: httpx.Client) -> Sent:
        return send(
            client,
            case,
            event_stream=stream,
            stream_limit=ctx.settings.stream_limit,
            stream_seconds=ctx.settings.stream_seconds,
        )

    answers = ask(ctx.reference), ask(ctx.candidate)
    record: dict[str, Any] = {
        'type': 'case',
        'case_id': case.case_id,
        'operation_id': case.operation_id,
        'label': case.label,
        'family': case.family,
        'mode': case.mode,
        'fingerprint': case.fingerprint,
        'request': case.summary(),
        'reference': answers[0].asdict(),
        'candidate': answers[1].asdict(),
    }
    differences: list[compare.Difference] = []
    if answers[0].observed is None or answers[1].observed is None:
        record['outcome'] = 'error'
        tally.transport_errors += 1
    else:
        differences = compare.compare_observed(answers[0].observed, answers[1].observed, norm)
        record['outcome'] = 'different' if differences else 'same'
        tally.different += bool(differences)
        tally.same += not differences
    record['difference_count'] = len(differences)
    record['differences'] = [d.asdict() for d in differences[:FIRST_DIFFERENCES]]
    failures = candidate_schema_failures(ctx, case, answers[1].response)
    tally.schema_failures += bool(failures)
    record['schema_failures'] = failures
    ctx.emit(record)
    note_finding(ctx, case, record)


def candidate_schema_failures(
    ctx: Context, case: cases.DiffCase, response: httpx.Response | None
) -> list[dict[str, Any]]:
    if not ctx.settings.check_schema or response is None:
        return []
    return schema_failures(case, response)


def note_finding(ctx: Context, case: cases.DiffCase, record: dict[str, Any]) -> None:
    if len(ctx.summary.findings) >= MAX_FINDINGS:
        return
    if record['outcome'] == 'error':
        what = f'transport error: {record["reference"].get("error") or record["candidate"].get("error")}'
    elif record['differences']:
        first = record['differences'][0]
        what = f'differs at {first["pointer"]}: {first["reference"]!r} vs {first["candidate"]!r}'
    elif record['schema_failures']:
        what = f'candidate breaks the schema at {"/".join(record["schema_failures"][0]["schema_path"])}'
    else:
        return
    ctx.summary.findings.append(f'{case.case_id} ({case.label}): {what}')


def exit_code(summary: Summary) -> int:
    return 0 if summary.ok else 1


def format_summary(summary: Summary, out_path: Path | None) -> str:
    data = summary.asdict()
    c, o = data['cases'], data['operations']
    mode = 'dry run' if summary.settings.dry_run else 'sent to both stacks'
    lines = [
        f'families {", ".join(data["scope"]["families"])}; seed {data["scope"]["seed"]}; {mode}',
        (
            f'operations {o["in_scope"]}, cases {c["total"]}: same {c["same"]}, '
            f'different {c["different"]}, transport errors {c["transport_errors"]}, '
            f'schema failures {c["schema_failures"]}'
        ),
    ]
    lines += [f'NO CASES: {label}' for label in o['zero_cases']]
    lines += [f'GENERATION ERROR: {message}' for message in o['generation_errors']]
    lines += [f'FIXTURE ERROR: {message}' for message in data['fixtures']['errors']]
    lines += [f'  {finding}' for finding in summary.findings]
    if out_path:
        lines.append(f'results: {out_path}')
    lines.append('RESULT: ' + ('OK, nothing differs' if summary.ok else 'FAILED'))
    return '\n'.join(lines)


def make_client(url: str, timeout: float) -> httpx.Client:
    return httpx.Client(
        base_url=url,
        timeout=httpx.Timeout(timeout),
        follow_redirects=False,
        headers={'user-agent': 'op-differential/1'},
    )


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--reference', default=os.environ.get('OP_DIFF_REFERENCE_URL'))
    parser.add_argument('--candidate', default=os.environ.get('OP_DIFF_CANDIDATE_URL'))
    parser.add_argument('--families', help='comma-separated family names; default all')
    parser.add_argument(
        '--methods', help='comma-separated HTTP methods, narrows the scope; default all'
    )
    parser.add_argument('--modes', default=','.join(cases.MODES))
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument(
        '--max-examples', type=int, default=8, help='generated cases per operation and mode'
    )
    parser.add_argument(
        '--fixture-cases', type=int, default=3, help='cases per operation with provider values'
    )
    parser.add_argument(
        '--check-schema', action='store_true', help='check the candidate against the schema'
    )
    parser.add_argument('--dry-run', action='store_true', help='generate cases, send nothing')
    parser.add_argument('--timeout', type=float, default=30.0)
    parser.add_argument('--stream-seconds', type=float, default=3.0)
    parser.add_argument(
        '--out', type=Path, help='JSON-lines file; default under artifacts_local/differential/'
    )
    return parser.parse_args(argv)


def settings_from(args: argparse.Namespace) -> Settings:
    modes = tuple(m for m in args.modes.split(',') if m)
    if not set(modes) <= set(cases.MODES):
        raise ConfigError(f'--modes takes {" and ".join(cases.MODES)}')
    return Settings(
        seed=args.seed,
        max_examples=args.max_examples,
        modes=modes,
        families=tuple(args.families.split(',')) if args.families else None,
        methods=frozenset(m.strip().upper() for m in args.methods.split(','))
        if args.methods
        else None,
        check_schema=args.check_schema,
        dry_run=args.dry_run,
        fixture_cases=args.fixture_cases,
        stream_seconds=args.stream_seconds,
    )


def default_out(seed: int) -> Path:
    stamp = datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')
    return families.REPO_ROOT / 'artifacts_local' / 'differential' / f'run-{stamp}-seed{seed}.jsonl'


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        settings = settings_from(args)
        if not settings.dry_run and not (args.reference and args.candidate):
            raise ConfigError(
                '--reference and --candidate (or OP_DIFF_REFERENCE_URL, OP_DIFF_CANDIDATE_URL) are required'
            )
        out_path: Path = args.out or default_out(settings.seed)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        clients = (
            (None, None)
            if settings.dry_run
            else (
                make_client(args.reference, args.timeout),
                make_client(args.candidate, args.timeout),
            )
        )
        with out_path.open('w', encoding='utf-8') as out:
            summary = run(settings, clients[0], clients[1], out)
    except ConfigError as error:
        print(f'differential runner: {error}', file=sys.stderr)
        return 2
    print(format_summary(summary, out_path))
    return exit_code(summary)


if __name__ == '__main__':
    sys.exit(main())
