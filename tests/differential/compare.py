"""Parsed-JSON comparison of two answers under an explicit, reviewed normalisation list.

What may differ between the reference stack and the Rust stack is written down, never blanket-ignored:
``normalise/_defaults.toml`` holds what holds for every family (18a section 3.2: the ``msg`` text of
validation errors, float spelling, key order, header case, the ``server`` header) and
``normalise/<family>.toml`` what that family's slice adds (generated ids, timestamps, durations, the
order of sets, its intended changes). Every rule carries a ``reason``; a rule without one is refused.

Status code and content type are always compared. The body is compared as parsed JSON: a number is
equal to the same number however it is spelled, a ``true`` never equals ``1``, key order is not
compared, and the order of an array only when no ``unordered`` rule names it.

Rule kinds (``pointer`` is a JSON pointer whose segments may be ``*`` for any key or index):

* ``opaque_id``     a generated id: replaced per document by a token numbered in order of first
                    appearance, so the ids may differ but the pattern of reuse may not
* ``timestamp``     both sides are ISO-8601 strings or numbers; their value is not compared
* ``ignore_value``  only presence and JSON type are compared (durations, counters); on an object or
                    array the whole subtree is left out
* ``approx``        numbers within ``abs`` or ``rel`` tolerance
* ``unordered``     an array compared as a set, sorted by the element's ``key`` (a relative pointer,
                    default the whole element); the key must not itself be an ``opaque_id``
* ``message_text``  a string compared only unless one side matches ``except_pattern`` (kept exact)
* ``drop``          the member is removed from both sides: an intended change of the slice
"""

from __future__ import annotations

import hashlib
import json
import re
import tomllib
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


NORMALISE_DIR = Path(__file__).resolve().parent / 'normalise'
KINDS = ('opaque_id', 'timestamp', 'ignore_value', 'approx', 'unordered', 'message_text', 'drop')
MAX_DIFFERENCES = 50


@dataclass(frozen=True)
class Rule:
    kind: str
    pointer: tuple[str, ...]
    reason: str
    statuses: frozenset[int] = frozenset()
    options: tuple[tuple[str, Any], ...] = ()

    def option(self, name: str, default: Any = None) -> Any:
        return dict(self.options).get(name, default)

    def applies(self, segments: tuple[str, ...]) -> bool:
        return len(segments) == len(self.pointer) and all(
            want in ('*', got) for want, got in zip(self.pointer, segments, strict=True)
        )


@dataclass(frozen=True)
class Normalisation:
    rules: tuple[Rule, ...] = ()
    compare_headers: tuple[str, ...] = ()

    def merged(self, other: Normalisation) -> Normalisation:
        headers = tuple(dict.fromkeys(self.compare_headers + other.compare_headers))
        return Normalisation(self.rules + other.rules, headers)

    def for_status(self, status: int) -> tuple[Rule, ...]:
        return tuple(r for r in self.rules if not r.statuses or status in r.statuses)


def split_pointer(pointer: str) -> tuple[str, ...]:
    if pointer == '':
        return ()
    if not pointer.startswith('/'):
        raise ValueError(f'normalise: {pointer!r} is not a JSON pointer')
    return tuple(s.replace('~1', '/').replace('~0', '~') for s in pointer[1:].split('/'))


def join_pointer(segments: tuple[str, ...]) -> str:
    return ''.join('/' + s.replace('~', '~0').replace('/', '~1') for s in segments)


def parse_rules(entries: list[dict[str, Any]], source: str) -> tuple[Rule, ...]:
    rules = []
    for entry in entries:
        kind = entry.get('kind')
        reason = str(entry.get('reason', '')).strip()
        if kind not in KINDS:
            raise ValueError(f'{source}: unknown rule kind {kind!r}')
        if not reason:
            raise ValueError(f'{source}: rule {kind} {entry.get("pointer")!r} has no reason')
        if 'pointer' not in entry:
            raise ValueError(f'{source}: rule {kind} has no pointer')
        known = {'kind', 'pointer', 'reason', 'status'}
        options = tuple(sorted((k, v) for k, v in entry.items() if k not in known))
        rules.append(
            Rule(
                kind,
                split_pointer(entry['pointer']),
                reason,
                frozenset(entry.get('status', ())),
                options,
            )
        )
    return tuple(rules)


def load_normalisation(family: str | None = None, directory: Path = NORMALISE_DIR) -> Normalisation:
    """The shared defaults, plus ``<family>.toml`` when the family has one."""
    names = ['_defaults'] + ([family] if family else [])
    result = Normalisation()
    for name in names:
        path = directory / f'{name}.toml'
        if not path.is_file():
            if name == '_defaults':
                raise FileNotFoundError(path)
            continue
        data = tomllib.loads(path.read_text(encoding='utf-8'))
        result = result.merged(
            Normalisation(
                parse_rules(data.get('rule', []), str(path)),
                tuple(h.lower() for h in data.get('compare_headers', ())),
            )
        )
    return result


@dataclass(frozen=True)
class Observed:
    """One stack's answer: what is compared, nothing else."""

    status: int
    headers: dict[str, str]
    body: bytes

    @classmethod
    def of(cls, status: int, headers: Any, body: bytes) -> Observed:
        return cls(status, {k.lower(): v for k, v in dict(headers).items()}, body)

    @property
    def media_type(self) -> str:
        return self.headers.get('content-type', '').split(';')[0].strip().lower()


@dataclass(frozen=True)
class Difference:
    pointer: str
    kind: str
    reference: Any
    candidate: Any

    def asdict(self) -> dict[str, Any]:
        return {
            'pointer': self.pointer,
            'kind': self.kind,
            'reference': shown(self.reference),
            'candidate': shown(self.candidate),
        }


def shown(value: Any, limit: int = 300) -> Any:
    text = json.dumps(value, sort_keys=True, ensure_ascii=False, default=repr)
    return value if len(text) <= limit else text[:limit] + f'... ({len(text)} chars)'


def is_json_media(media_type: str) -> bool:
    return media_type == 'application/json' or media_type.endswith('+json')


def compare_observed(
    reference: Observed, candidate: Observed, norm: Normalisation
) -> list[Difference]:
    """Every difference between two answers; an empty list means they agree."""
    found: list[Difference] = []
    if reference.status != candidate.status:
        found.append(Difference('status', 'changed', reference.status, candidate.status))
    if reference.media_type != candidate.media_type:
        found.append(
            Difference('content_type', 'changed', reference.media_type, candidate.media_type)
        )
    for name in norm.compare_headers:
        a, b = reference.headers.get(name), candidate.headers.get(name)
        if a != b:
            found.append(Difference(f'header:{name}', 'changed', a, b))
    found += compare_bodies(reference, candidate, norm)
    return found


def compare_bodies(
    reference: Observed, candidate: Observed, norm: Normalisation
) -> list[Difference]:
    if is_json_media(reference.media_type) and is_json_media(candidate.media_type):
        try:
            a, b = json.loads(reference.body or b'null'), json.loads(candidate.body or b'null')
        except ValueError:
            return exact_body(reference, candidate)
        return compare_json(a, b, norm.for_status(reference.status))
    return exact_body(reference, candidate)


def exact_body(reference: Observed, candidate: Observed) -> list[Difference]:
    if reference.body == candidate.body:
        return []
    return [Difference('body', 'bytes', digest(reference.body), digest(candidate.body))]


def digest(body: bytes) -> str:
    return f'sha256:{hashlib.sha256(body).hexdigest()[:16]} ({len(body)} bytes)'


def compare_json(reference: Any, candidate: Any, rules: tuple[Rule, ...]) -> list[Difference]:
    prepared = (prepare(reference, rules), prepare(candidate, rules))
    out: list[Difference] = []
    walk(prepared[0], prepared[1], (), rules, out)
    return out


# ---- phase 1: per document, drop members, order sets, number opaque ids -------------------------


@dataclass
class _Tokens:
    seen: dict[str, str] = field(default_factory=dict)

    def token(self, value: Any) -> str:
        key = json.dumps(value, sort_keys=True)
        return self.seen.setdefault(key, f'<id:{len(self.seen) + 1}>')


def prepare(node: Any, rules: tuple[Rule, ...]) -> Any:
    tokens = _Tokens()
    return _prepare(node, (), rules, tokens)


def _matching(rules: tuple[Rule, ...], kind: str, segments: tuple[str, ...]) -> Rule | None:
    return next((r for r in rules if r.kind == kind and r.applies(segments)), None)


def _prepare(node: Any, segments: tuple[str, ...], rules: tuple[Rule, ...], tokens: _Tokens) -> Any:
    if _matching(rules, 'opaque_id', segments) and node is not None:
        return tokens.token(node)
    if isinstance(node, dict):
        return {
            k: _prepare(node[k], (*segments, k), rules, tokens)
            for k in sorted(node)
            if not _matching(rules, 'drop', (*segments, k))
        }
    if isinstance(node, list):
        ordered = _ordered(node, segments, rules)
        return [_prepare(v, (*segments, str(i)), rules, tokens) for i, v in enumerate(ordered)]
    return node


def _ordered(items: list[Any], segments: tuple[str, ...], rules: tuple[Rule, ...]) -> list[Any]:
    rule = _matching(rules, 'unordered', segments)
    if rule is None:
        return items
    key = rule.option('key')
    return sorted(items, key=lambda item: _sort_key(item, key))


def _sort_key(item: Any, key: str | None) -> str:
    value = item
    for segment in split_pointer(key) if key else ():
        value = value.get(segment) if isinstance(value, dict) else None
    return json.dumps(canonical(value), sort_keys=True, ensure_ascii=False)


def canonical(value: Any) -> Any:
    """A value with integral floats made ints, so ``1.0`` and ``1`` sort and compare alike."""
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, dict):
        return {k: canonical(v) for k, v in value.items()}
    if isinstance(value, list):
        return [canonical(v) for v in value]
    return value


# ---- phase 2: walk both documents together ------------------------------------------------------


def walk(
    a: Any, b: Any, segments: tuple[str, ...], rules: tuple[Rule, ...], out: list[Difference]
) -> None:
    if len(out) >= MAX_DIFFERENCES:
        return
    if pairwise_rule_holds(a, b, segments, rules):
        return
    here = join_pointer(segments)
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(a.keys() | b.keys()):
            if key not in b:
                out.append(
                    Difference(join_pointer((*segments, key)), 'missing_in_candidate', a[key], None)
                )
            elif key not in a:
                out.append(
                    Difference(join_pointer((*segments, key)), 'extra_in_candidate', None, b[key])
                )
            else:
                walk(a[key], b[key], (*segments, key), rules, out)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(Difference(here, 'length', len(a), len(b)))
        for index in range(min(len(a), len(b))):
            walk(a[index], b[index], (*segments, str(index)), rules, out)
    elif not scalar_equal(a, b):
        kind = 'type' if json_type(a) != json_type(b) else 'changed'
        out.append(Difference(here, kind, a, b))


def json_type(value: Any) -> str:
    if value is None:
        return 'null'
    if isinstance(value, bool):
        return 'boolean'
    if isinstance(value, int | float):
        return 'number'
    if isinstance(value, str):
        return 'string'
    return 'array' if isinstance(value, list) else 'object'


def scalar_equal(a: Any, b: Any) -> bool:
    if json_type(a) != json_type(b):
        return False
    return a == b


def pairwise_rule_holds(a: Any, b: Any, segments: tuple[str, ...], rules: tuple[Rule, ...]) -> bool:
    """True when a rule at this path says ``a`` and ``b`` agree."""
    for rule in rules:
        if not rule.applies(segments):
            continue
        if rule.kind == 'ignore_value' and json_type(a) == json_type(b):
            return True
        if rule.kind == 'timestamp' and is_timestamp(a) and is_timestamp(b):
            return True
        if rule.kind == 'approx' and within(a, b, rule):
            return True
        if rule.kind == 'message_text' and message_text_agrees(a, b, rule):
            return True
    return False


def is_timestamp(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, int | float):
        return True
    if not isinstance(value, str):
        return False
    try:
        datetime.fromisoformat(value)
    except ValueError:
        return False
    return True


def within(a: Any, b: Any, rule: Rule) -> bool:
    if json_type(a) != 'number' or json_type(b) != 'number':
        return False
    delta = abs(a - b)
    return delta <= float(rule.option('abs', 0.0)) or delta <= float(rule.option('rel', 0.0)) * max(
        abs(a), abs(b)
    )


def message_text_agrees(a: Any, b: Any, rule: Rule) -> bool:
    if not (isinstance(a, str) and isinstance(b, str)):
        return False
    keep = rule.option('except_pattern')
    return not (keep and (re.search(keep, a) or re.search(keep, b)))
