"""Operation to family assignment: which route family owns each operation of the contracts.

``families.toml`` lists, per family, the path patterns it owns. Every operation of every
``contracts/openapi/*.json`` file present must land in exactly one family, so no operation of a route being
deleted can fall outside a differential run (docs/design/v070_rebuild/18a_backend_rust_port_slices.md,
section 3.2). The module reads the contracts as plain JSON: it has no Schemathesis dependency.

A pattern is ``[METHOD[,METHOD] ]<path>``. A path with no ``*`` matches that path exactly; ``<prefix>/*``
matches the prefix itself and everything below it. The most specific pattern wins (exact before prefix,
a method list before none, a longer prefix before a shorter one); two families holding equally specific
patterns for one operation is an error, so the order of the file never matters.
"""

from __future__ import annotations

import json
import tomllib
from dataclasses import dataclass
from pathlib import Path


DIFFERENTIAL_DIR = Path(__file__).resolve().parent
REPO_ROOT = DIFFERENTIAL_DIR.parents[1]
FAMILIES_FILE = DIFFERENTIAL_DIR / 'families.toml'
CONTRACT_DIR = REPO_ROOT / 'contracts' / 'openapi'

HTTP_METHODS = ('get', 'put', 'post', 'delete', 'options', 'head', 'patch', 'trace')


@dataclass(frozen=True)
class Operation:
    contract: str
    method: str
    path: str
    operation_id: str

    @property
    def label(self) -> str:
        return f'{self.method} {self.path}'


@dataclass(frozen=True)
class Pattern:
    methods: frozenset[str]
    path: str
    prefix: bool

    @classmethod
    def parse(cls, text: str) -> Pattern:
        head, _, tail = text.strip().rpartition(' ')
        methods = (
            frozenset(m.strip().upper() for m in head.split(',') if m.strip())
            if head
            else frozenset()
        )
        unknown = methods - {m.upper() for m in HTTP_METHODS}
        if unknown or not tail.startswith('/'):
            raise ValueError(f'families.toml: bad pattern {text!r}')
        if tail.endswith('/*'):
            return cls(methods, tail[:-2], prefix=True)
        if '*' in tail:
            raise ValueError(f'families.toml: "*" is allowed only as a final "/*": {text!r}')
        return cls(methods, tail, prefix=False)

    def matches(self, operation: Operation) -> bool:
        if self.methods and operation.method not in self.methods:
            return False
        if self.prefix:
            return operation.path == self.path or operation.path.startswith(self.path + '/')
        return operation.path == self.path

    @property
    def specificity(self) -> tuple[int, int, int]:
        return (0 if self.prefix else 1, 1 if self.methods else 0, len(self.path.split('/')))


@dataclass(frozen=True)
class Family:
    name: str
    contract: str
    slice: str
    patterns: tuple[Pattern, ...]
    fixtures: tuple[str, ...]

    def best_match(self, operation: Operation) -> tuple[int, int, int] | None:
        found = [p.specificity for p in self.patterns if p.matches(operation)]
        return max(found) if found else None


def load_operations(contract_dir: Path = CONTRACT_DIR) -> list[Operation]:
    """Every operation of every contract file in the directory, in file, path, method order."""
    operations: list[Operation] = []
    for contract in sorted(contract_dir.glob('*.json')):
        document = json.loads(contract.read_text(encoding='utf-8'))
        for path in sorted(document.get('paths', {})):
            item = document['paths'][path]
            for method in HTTP_METHODS:
                if method in item:
                    operation_id = item[method].get('operationId', '')
                    operations.append(Operation(contract.stem, method.upper(), path, operation_id))
    return operations


def load_families(path: Path = FAMILIES_FILE) -> list[Family]:
    data = tomllib.loads(path.read_text(encoding='utf-8'))
    return [
        Family(
            name=entry['name'],
            contract=entry['contract'],
            slice=entry['slice'],
            patterns=tuple(Pattern.parse(text) for text in entry['paths']),
            fixtures=tuple(entry.get('fixtures', ())),
        )
        for entry in data['family']
    ]


def assign(operation: Operation, families: list[Family]) -> Family:
    """The one family that owns ``operation``; raises when none or two equally specific ones do."""
    scored = [(f.best_match(operation), f) for f in families]
    hits = [(score, f) for score, f in scored if score is not None]
    if not hits:
        raise LookupError(f'{operation.label} ({operation.operation_id}) is in no family')
    best = max(score for score, _ in hits)
    winners = [f for score, f in hits if score == best]
    if len(winners) > 1:
        names = ', '.join(sorted(f.name for f in winners))
        raise LookupError(f'{operation.label} matches equally specific patterns of: {names}')
    return winners[0]


def coverage_problems(operations: list[Operation], families: list[Family]) -> list[str]:
    """Why the mapping is not a total, unambiguous one over ``operations``; empty when it is."""
    problems: list[str] = []
    names = [f.name for f in families]
    problems += [
        f'family {n} is listed twice' for n in sorted({n for n in names if names.count(n) > 1})
    ]
    present = {op.contract for op in operations}
    used: set[tuple[str, int]] = set()
    for operation in operations:
        try:
            family = assign(operation, families)
        except LookupError as error:
            problems.append(str(error))
            continue
        if family.contract != operation.contract:
            problems.append(
                f'{operation.label} is in family {family.name}, a {family.contract} family'
            )
        for index, pattern in enumerate(family.patterns):
            if pattern.matches(operation) and pattern.specificity == family.best_match(operation):
                used.add((family.name, index))
    for family in families:
        if family.contract not in present:
            continue
        problems += [
            f'family {family.name}: pattern {family.patterns[i].path!r} owns no operation'
            for i in range(len(family.patterns))
            if (family.name, i) not in used
        ]
    return problems


def by_family(operations: list[Operation], families: list[Family]) -> dict[str, list[Operation]]:
    grouped: dict[str, list[Operation]] = {f.name: [] for f in families}
    for operation in operations:
        grouped[assign(operation, families).name].append(operation)
    return grouped
