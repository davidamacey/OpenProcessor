"""Deterministic request generation from the committed OpenAPI files, through Schemathesis.

Documentation read (Schemathesis 4.33.4, https://schemathesis.readthedocs.io/en/stable/):

* loading a schema from a file: ``schemathesis.openapi.from_path`` (reference/python);
* one Hypothesis strategy per operation, with fixed path, query, header or body parts and the
  positive or negative generation mode: ``APIOperation.as_strategy`` (reference/python);
* preparing a case for a transport: ``BaseSchema.as_transport_kwargs`` (reference/python); the runner
  sends the result with httpx, so a test can put ``httpx.MockTransport`` underneath;
* data generation modes: explanations/data-generation.

Determinism. Cases are generated once per operation and the same objects are sent to both stacks, so the
two stacks cannot see different requests. Between runs, the same ``seed`` gives the same cases because
each operation's draws run under ``hypothesis.seed(<seed mixed with the operation label>)`` with the
example database off: Hypothesis documents that a seeded test replays the same examples
(https://hypothesis.readthedocs.io/en/latest/reference/api.html#hypothesis.seed), and Schemathesis's own
pytest and CLI paths derive a per-operation seed the same way. The only random part of a Schemathesis
case is its ``id`` (a header for log correlation); it is replaced here by ``<operation id>:<mode>:<n>``.
"""

from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import httpx
import schemathesis
from hypothesis import HealthCheck, Phase, given, seed, settings
from hypothesis.errors import Unsatisfiable
from schemathesis.core.result import Err
from schemathesis.generation import GenerationMode


if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence
    from pathlib import Path

    from schemathesis.generation.case import Case
    from schemathesis.schemas import APIOperation, BaseSchema

PLACEHOLDER_BASE = 'http://op-diff.invalid'
CASE_HEADER = 'x-op-diff-case'
# Headers Schemathesis's transport adds to every request; none is part of the generated case.
_TRANSPORT_DEFAULTS = frozenset(
    {'user-agent', 'accept-encoding', 'accept', 'connection', 'x-schemathesis-testcaseid'}
)
MODES = ('positive', 'negative')


@dataclass(frozen=True)
class DiffCase:
    """One request, independent of the stack it is sent to."""

    case_id: str
    operation_id: str
    label: str
    family: str
    mode: str
    method: str
    path: str
    headers: tuple[tuple[str, str], ...]
    params: Any
    body: tuple[str, Any] | None
    fingerprint: str
    source: Case = field(compare=False, repr=False)

    def request_kwargs(self) -> dict[str, Any]:
        """Arguments for ``httpx.Client.build_request`` (the path is relative to the client's base)."""
        kwargs: dict[str, Any] = {
            'headers': {**dict(self.headers), CASE_HEADER: self.case_id},
            'params': self.params,
        }
        if self.body is not None:
            kwargs[self.body[0]] = self.body[1]
        return kwargs

    def summary(self) -> dict[str, Any]:
        """What a person needs to replay the request."""
        body = None if self.body is None else _printable_body(self.body)
        return {
            'method': self.method,
            'path': self.path,
            'query': self.params,
            'headers': dict(self.headers),
            'body': body,
        }


def _printable_body(body: tuple[str, Any]) -> Any:
    kind, value = body
    if kind == 'content':
        return {'bytes_hex': value.hex()[:200], 'length': len(value)}
    if kind == 'files':
        return [_file_summary(item) for item in value]
    return value


def _file_summary(item: Any) -> Any:
    name, spec = item
    content = spec[1] if isinstance(spec, tuple) and len(spec) > 1 else b''
    raw = content.encode() if isinstance(content, str) else bytes(content)
    return {'field': name, 'sha256': hashlib.sha256(raw).hexdigest()[:16], 'length': len(raw)}


def load_schema(contract: Path) -> BaseSchema:
    return schemathesis.openapi.from_path(contract)


def operations_of(schema: BaseSchema) -> list[APIOperation]:
    """Every operation of the schema; an operation the loader rejects is an error, not a skip."""
    operations: list[APIOperation] = []
    for result in schema.get_all_operations():
        if isinstance(result, Err):
            raise result.err()
        operations.append(result.ok())
    return operations


def operation_id_of(operation: APIOperation) -> str:
    return str(operation.definition.raw.get('operationId') or operation.label)


def declares_event_stream(operation: APIOperation) -> bool:
    return 'text/event-stream' in json.dumps(operation.definition.raw.get('responses', {}))


def mixed_seed(seed_value: int, label: str) -> int:
    """A stable per-operation seed: operations with the same parameters must not draw the same inputs."""
    digest = hashlib.blake2b(f'{seed_value}:{label}'.encode(), digest_size=8).digest()
    return int.from_bytes(digest, 'big')


@contextmanager
def without_local_constants() -> Iterator[None]:
    """Generate without Hypothesis drawing from the literals of the local modules loaded in the process.

    Hypothesis mixes the string, number and bytes constants of every non-stdlib, non-site-packages
    module in ``sys.modules`` into 5 to 15 percent of its draws (``hypothesis/internal/conjecture/
    providers.py``, ``_maybe_draw_constant``; modules under a ``tests`` directory are skipped). So the same
    seed gives other cases in a process that has imported other project modules, or when this script's own
    literals change. The pool is made empty for the length of a draw; Hypothesis's own global constants,
    which depend on its version only, stay. The version of both libraries is written to every summary.
    """
    from hypothesis.internal.conjecture import providers
    from hypothesis.internal.constants_ast import Constants

    saved = providers._get_local_constants
    providers._get_local_constants = Constants
    providers.CONSTANTS_CACHE.cache.clear()
    try:
        yield
    finally:
        providers._get_local_constants = saved
        providers.CONSTANTS_CACHE.cache.clear()


def _collect(strategy: Any, seed_value: int, count: int) -> list[Case]:
    found: list[Case] = []

    @seed(seed_value)
    @settings(
        max_examples=count,
        database=None,
        deadline=None,
        phases=[Phase.generate],
        suppress_health_check=list(HealthCheck),
        derandomize=False,
    )
    @given(strategy)
    def draw(case: Case) -> None:
        found.append(case)

    with without_local_constants():
        draw()
    return found


def _fingerprint(parts: dict[str, Any]) -> str:
    text = json.dumps(parts, sort_keys=True, ensure_ascii=True, default=repr)
    return hashlib.sha256(text.encode()).hexdigest()[:20]


def _body_of(kwargs: Mapping[str, Any]) -> tuple[str, Any] | None:
    if 'json' in kwargs:
        return ('json', kwargs['json'])
    if kwargs.get('files'):
        return ('files', kwargs['files'])
    data = kwargs.get('data')
    if isinstance(data, bytes | str):
        return ('content', data.encode() if isinstance(data, str) else data)
    if data:
        return ('data', data)
    return None


def _body_fingerprint(body: tuple[str, Any] | None) -> Any:
    if body is None:
        return None
    kind, value = body
    if kind == 'content':
        return [kind, hashlib.sha256(value).hexdigest()]
    if kind == 'files':
        return [kind, [_file_summary(item) for item in value]]
    return [kind, value]


def to_diff_case(
    case: Case, *, operation_id: str, family: str, mode: str, number: int
) -> DiffCase | None:
    """The stack-independent form of ``case``; ``None`` when it cannot be put on the wire at all."""
    kwargs = case.as_transport_kwargs(base_url=PLACEHOLDER_BASE)
    path = kwargs['url'][len(PLACEHOLDER_BASE) :]
    generated = {name.lower() for name in case.headers}
    headers = {
        name: str(value)
        for name, value in kwargs['headers'].items()
        if name.lower() not in _TRANSPORT_DEFAULTS or name.lower() in generated
    }
    if kwargs.get('cookies'):
        headers['cookie'] = '; '.join(f'{k}={v}' for k, v in sorted(kwargs['cookies'].items()))
    body = _body_of(kwargs)
    params = kwargs.get('params') or {}
    try:
        httpx.Request(
            kwargs['method'],
            PLACEHOLDER_BASE + path,
            headers=headers,
            params=params,
            **_as_httpx(body),
        )
    except (UnicodeError, ValueError, TypeError):
        return None
    case_id = f'{operation_id}:{mode}:{number}'
    ordered = tuple(sorted((k.lower(), v) for k, v in headers.items()))
    fingerprint = _fingerprint(
        {
            'method': kwargs['method'],
            'path': path,
            'headers': ordered,
            'params': params,
            'body': _body_fingerprint(body),
        }
    )
    return DiffCase(
        case_id=case_id,
        operation_id=operation_id,
        label=case.operation.label,
        family=family,
        mode=mode,
        method=kwargs['method'],
        path=path,
        headers=ordered,
        params=params,
        body=body,
        fingerprint=fingerprint,
        source=case,
    )


def _as_httpx(body: tuple[str, Any] | None) -> dict[str, Any]:
    return {} if body is None else {body[0]: body[1]}


@dataclass
class Generated:
    operation_id: str
    label: str
    family: str
    cases: list[DiffCase] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def generate(
    operation: APIOperation,
    family: str,
    *,
    seed_value: int,
    max_examples: int,
    modes: Sequence[str] = MODES,
    bindings: Sequence[Mapping[str, Any]] = (),
) -> Generated:
    """The cases of one operation: generated ones per mode, then one per fixture binding.

    ``bindings`` come from a fixtures provider: each maps parameter names to values present in both
    stacks; the case keeps them and generates everything else.
    """
    operation_id = operation_id_of(operation)
    result = Generated(operation_id, operation.label, family)
    seen: set[str] = set()

    def add(drawn: list[Case], mode: str) -> None:
        for case in drawn:
            built = to_diff_case(
                case, operation_id=operation_id, family=family, mode=mode, number=len(result.cases)
            )
            if built is None:
                result.notes.append(
                    f'{mode}: a generated request cannot be put on the wire; dropped'
                )
            elif built.fingerprint not in seen:
                seen.add(built.fingerprint)
                result.cases.append(built)

    for mode in modes:
        generation = GenerationMode(mode)
        label_seed = mixed_seed(seed_value, f'{operation.label}:{mode}')
        try:
            add(_collect(operation.as_strategy(generation), label_seed, max_examples), mode)
        except Unsatisfiable:
            result.notes.append(
                f'{mode}: Schemathesis can generate no {mode} data for this operation'
            )
    path_names = {p.name for p in operation.path_parameters}
    query_names = {p.name for p in operation.query}
    for index, binding in enumerate(bindings):
        fixed_path = {k: v for k, v in binding.items() if k in path_names}
        fixed_query = {k: v for k, v in binding.items() if k in query_names}
        if not (fixed_path or fixed_query):
            continue
        strategy = operation.as_strategy(
            GenerationMode.POSITIVE, path_parameters=fixed_path or None, query=fixed_query or None
        )
        label_seed = mixed_seed(seed_value, f'{operation.label}:fixture:{index}')
        add(_collect(strategy, label_seed, 1), 'fixture')
    return result
