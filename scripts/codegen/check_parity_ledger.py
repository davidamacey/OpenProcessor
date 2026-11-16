#!/usr/bin/env python3
"""Pre-commit guard: a deleted Python test has a row in the parity ledger.

The Rust port deletes Python tests with the Python routes they covered (docs/design/v070_rebuild/
18a_backend_rust_port_slices.md, section 3.2). Each deleted test id needs one row in
``tests/parity/ledger/<slice>.tsv``: ``<test id><TAB><replacement>``. The id is
``<path>::<function>`` or ``<path>::<Class>::<function>`` (a parametrized test is one row).
The replacement says what now proves the behaviour, in one of four forms:

* ``rust: <path>::<name>``  a Rust test; the file must exist and contain ``fn <name>``
* ``diff: <family>/<glob>``  differential cases: a family of ``tests/differential/families.toml``
  and a glob that matches at least one operation id of ``contracts/openapi/*.json``
* ``contract: <file> <what is checked>``  a check against a committed contract file
* ``gone: <reason>``  a behaviour removed on purpose, with the reason

Which tests were deleted comes from git: the staged change by default, or two commits
(``--between A B``, for a sweep over a merged range). A test counts as deleted when its
``(class, function)`` is gone from its file; moving a file with the names unchanged is not a deletion.
Rows are also checked on their own: well-formed, resolvable, unique, and never for a test that still
exists (``--ledger-only`` checks nothing else). A failure prints the rows still to fill in.
"""

from __future__ import annotations

import argparse
import ast
import fnmatch
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
LEDGER_DIR = 'tests/parity/ledger'
TEST_FILE = re.compile(r'^tests/(?:.*/)?(?:test_[^/]*|[^/]*_test)\.py$')
FORMS = ('rust:', 'diff:', 'contract:', 'gone:')


@dataclass(frozen=True)
class Row:
    test_id: str
    replacement: str
    where: str


def git(root: Path, *args: str) -> str:
    done = subprocess.run(  # nosec B603 B607  fixed argv, git from PATH
        ['git', *args], cwd=root, capture_output=True, text=True, check=False
    )
    if done.returncode != 0:
        raise RuntimeError(f'git {" ".join(args)}: {done.stderr.strip()}')
    return done.stdout


def test_ids_in(source: str, path: str) -> set[tuple[str, str]]:
    """``(class, function)`` of every test function in a module; class is '' at module level."""
    found: set[tuple[str, str]] = set()
    tree = ast.parse(source, filename=path)

    def walk(body: list[ast.stmt], owner: str) -> None:
        for node in body:
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name.startswith(
                'test'
            ):
                found.add((owner, node.name))
            elif isinstance(node, ast.ClassDef) and node.name.startswith('Test'):
                walk(node.body, f'{owner}::{node.name}'.lstrip(':'))

    walk(tree.body, '')
    return found


def node_id(path: str, owner: str, function: str) -> str:
    return '::'.join(part for part in (path, owner, function) if part)


def changes(root: Path, between: tuple[str, str] | None) -> list[tuple[str, str, str]]:
    """``(status, old path, new path)`` of every removed, modified or renamed file under tests/."""
    if between is None:
        try:
            git(root, 'rev-parse', '--verify', '-q', 'HEAD')
        except RuntimeError:
            return []
        args = [
            'diff',
            '--cached',
            '--name-status',
            '-M',
            '--diff-filter=DMR',
            'HEAD',
            '--',
            'tests/',
        ]
    else:
        args = ['diff', '--name-status', '-M', '--diff-filter=DMR', *between, '--', 'tests/']
    rows = []
    for line in git(root, *args).splitlines():
        status, *paths = line.split('\t')
        rows.append((status[0], paths[0], paths[-1]))
    return rows


def read_at(root: Path, revision: str, path: str) -> str:
    return git(root, 'show', f'{revision}:{path}')


def deleted_ids(root: Path, between: tuple[str, str] | None) -> list[str]:
    old_rev, new_rev = (between[0], between[1]) if between else ('HEAD', '')
    deleted: list[str] = []
    for status, old_path, new_path in changes(root, between):
        if not TEST_FILE.match(old_path):
            continue
        old = test_ids_in(read_at(root, old_rev, old_path), old_path)
        new: set[tuple[str, str]] = set()
        if status != 'D' and TEST_FILE.match(new_path):
            new = test_ids_in(read_at(root, new_rev, new_path), new_path)
        deleted += [node_id(old_path, owner, name) for owner, name in sorted(old - new)]
    return deleted


def read_rows(root: Path) -> tuple[list[Row], list[str]]:
    rows: list[Row] = []
    problems: list[str] = []
    for ledger in sorted((root / LEDGER_DIR).glob('*.tsv')):
        for number, raw in enumerate(ledger.read_text(encoding='utf-8').splitlines(), start=1):
            if not raw.strip() or raw.startswith('#'):
                continue
            where = f'{ledger.relative_to(root).as_posix()}:{number}'
            test_id, tab, replacement = raw.partition('\t')
            if not tab or not test_id.strip() or not replacement.strip():
                problems.append(f'{where}: want "<test id><TAB><replacement>"')
                continue
            rows.append(Row(test_id.strip(), replacement.strip(), where))
    return rows, problems


def defined(root: Path, test_id: str) -> bool:
    path, _, rest = test_id.partition('::')
    file = root / path
    if not file.is_file() or not rest:
        return False
    *owners, function = rest.split('::')
    try:
        return ('::'.join(owners), function) in test_ids_in(file.read_text(encoding='utf-8'), path)
    except SyntaxError:
        return False


def check_replacement(root: Path, row: Row) -> str | None:
    text = row.replacement
    if not text.startswith(FORMS):
        return f'{row.where}: replacement must start with one of {", ".join(FORMS)}'
    form, _, body = text.partition(':')
    body = body.strip()
    if not body:
        return f'{row.where}: "{form}:" needs content'
    if form == 'rust':
        path, _, name = body.partition('::')
        file = root / path
        if not name or not file.is_file() or f'fn {name}' not in file.read_text(encoding='utf-8'):
            return f'{row.where}: no Rust test {body} (want <path>::<name> of a file holding `fn <name>`)'
    elif form == 'contract':
        if not (root / body.split()[0]).is_file():
            return f'{row.where}: {body.split()[0]} is not a contract file in the tree'
    elif form == 'diff':
        return check_diff(root, row, body)
    return None


def check_diff(root: Path, row: Row, body: str) -> str | None:
    family, _, pattern = body.partition('/')
    if not pattern:
        return f'{row.where}: "diff:" wants <family>/<operation id glob>'
    sys.path.insert(0, str(REPO_ROOT / 'tests' / 'differential'))
    import families as fam

    known = {f.name for f in fam.load_families(root / 'tests/differential/families.toml')}
    if family not in known:
        return f'{row.where}: {family} is not a family of tests/differential/families.toml'
    ids = [op.operation_id for op in fam.load_operations(root / 'contracts/openapi')]
    if not any(fnmatch.fnmatchcase(operation_id, pattern) for operation_id in ids):
        return f'{row.where}: no operation id matches {pattern}'
    return None


def ledger_problems(root: Path) -> tuple[list[Row], list[str]]:
    rows, problems = read_rows(root)
    seen: dict[str, str] = {}
    for row in rows:
        if row.test_id in seen:
            problems.append(f'{row.where}: {row.test_id} already has a row at {seen[row.test_id]}')
        seen[row.test_id] = row.where
        if defined(root, row.test_id):
            problems.append(f'{row.where}: {row.test_id} still exists, so it needs no row')
        replacement_problem = check_replacement(root, row)
        if replacement_problem:
            problems.append(replacement_problem)
    return rows, problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--root', type=Path, default=REPO_ROOT)
    parser.add_argument(
        '--between', nargs=2, metavar=('OLD', 'NEW'), help='two commits instead of the index'
    )
    parser.add_argument(
        '--ledger-only', action='store_true', help='check the ledger files, not git'
    )
    args = parser.parse_args(argv)
    root: Path = args.root.resolve()

    rows, problems = ledger_problems(root)
    missing: list[str] = []
    if not args.ledger_only:
        try:
            have = {row.test_id for row in rows}
            missing = [
                t
                for t in deleted_ids(root, tuple(args.between) if args.between else None)
                if t not in have
            ]
        except (RuntimeError, SyntaxError) as error:
            print(f'check_parity_ledger: cannot read the change: {error}', file=sys.stderr)
            return 2
    for problem in problems:
        print(problem)
    if missing:
        print('Deleted tests with no row in tests/parity/ledger/<slice>.tsv; add one row each:')
        for test_id in missing:
            print(
                f'{test_id}\t<rust: path::name | diff: family/glob | contract: file what | gone: reason>'
            )
    return 1 if problems or missing else 0


if __name__ == '__main__':
    sys.exit(main())
