#!/usr/bin/env python3
"""Write ``native/op_db/schema/head.sql``: the schema the Rust database tests load.

Alembic owns the schema until the Rust server takes the migrations over (plan 18, section 3.2). The
Rust database tier (``op_db::testing``) starts a bare PostgreSQL, loads this dump, and applies every
``src/db/migrations/sql/00NN_*.sql`` newer than the revision on the dump's first line, so a Rust
test needs no Python install and no migrated database of its own.

The dump is ``pg_dump --schema-only --no-owner --no-privileges`` of a database that ``db-migrate``
has brought to head (the live harness's, by default), preceded by one line:

    -- alembic revision: <version_num of alembic_version>

The loader creates the roles and re-applies the grants (``src/db/roles.py``), which is why the dump
carries neither owners nor privileges. Only the coordinator runs this, in the sweep, against the
harness stack; a lane commits no hand-written dump.

Usage:

    python3 scripts/codegen/dump_schema.py --compose-project <name>           # write
    python3 scripts/codegen/dump_schema.py --compose-project <name> --check   # exit 1 on a diff

Each database call is one short statement of its own, with its own ``statement_timeout``.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO_ROOT / 'native' / 'op_db' / 'schema' / 'head.sql'
REVISION_PREFIX = '-- alembic revision: '
DUMP_TIMEOUT_S = 300
REVISION_TIMEOUT_S = 30
# pg_dump 18 brackets a dump in `\restrict <key>` / `\unrestrict <key>` with a random key; a fixed
# (alphanumeric) one keeps two dumps of one schema byte-equal, which `--check` compares.
RESTRICT_KEY = 'opschemahead'


class DumpError(RuntimeError):
    """The dump could not be made; the message says what to fix."""


def _run(argv: Sequence[str], timeout_s: int) -> str:
    try:
        done = subprocess.run(  # nosec B603 - fixed argv built below, no shell
            list(argv), capture_output=True, text=True, check=False, timeout=timeout_s
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise DumpError(f'{argv[0]} did not run: {exc}') from exc
    if done.returncode != 0:
        raise DumpError(
            f'{" ".join(argv[:6])} ... failed ({done.returncode}): {done.stderr.strip()}'
        )
    return done.stdout


def _exec(project: str, service: str, *command: str) -> list[str]:
    # The statement timeout rides in PGOPTIONS: a `SET ...; SELECT ...` in one `-c` makes psql
    # print the `SET` command tag as a second line of output.
    return [
        'docker',
        'compose',
        '-p',
        project,
        'exec',
        '-T',
        '-e',
        'PGOPTIONS=-c statement_timeout=5s',
        service,
        *command,
    ]


def read_revision(
    run: Callable[[Sequence[str], int], str], project: str, args: argparse.Namespace
) -> str:
    """The revision the database is at, from ``alembic_version``."""
    out = run(
        _exec(
            project,
            args.service,
            'psql',
            '-X',
            '-A',
            '-t',
            '-U',
            args.user,
            '-d',
            args.database,
            '-c',
            'SELECT version_num FROM alembic_version',
        ),
        REVISION_TIMEOUT_S,
    )
    revisions = [line.strip() for line in out.splitlines() if line.strip()]
    if len(revisions) != 1:
        raise DumpError(
            f'alembic_version holds {len(revisions)} revisions; run db-migrate on the harness first'
        )
    return revisions[0]


def render(run: Callable[[Sequence[str], int], str], args: argparse.Namespace) -> str:
    """The text of ``head.sql``."""
    project: str = args.compose_project
    revision = read_revision(run, project, args)
    dump = run(
        _exec(
            project,
            args.service,
            'pg_dump',
            '--schema-only',
            '--no-owner',
            '--no-privileges',
            f'--restrict-key={RESTRICT_KEY}',
            '-U',
            args.user,
            '-d',
            args.database,
        ),
        DUMP_TIMEOUT_S,
    )
    if 'CREATE TABLE public.jobs' not in dump:
        raise DumpError('the dump has no jobs table; is this the OpenProcessor database?')
    return f'{REVISION_PREFIX}{revision}\n{dump.rstrip()}\n'


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--compose-project', required=True, help='the harness stack (`-p <name>`)')
    parser.add_argument('--service', default='postgres')
    parser.add_argument('--user', default='op_owner')
    parser.add_argument('--database', default='openprocessor')
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--check', action='store_true', help='Do not write; exit 1 on a diff.')
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None, run: Callable[[Sequence[str], int], str] = _run) -> int:
    args = parse_args(argv)
    try:
        text = render(run, args)
    except DumpError as exc:
        sys.stderr.write(f'dump_schema: {exc}\n')
        return 2
    out: Path = args.out
    if args.check:
        if not out.is_file() or out.read_text(encoding='utf-8') != text:
            sys.stderr.write(
                f'{out} is out of date; re-run without --check and commit the result\n'
            )
            return 1
        sys.stdout.write(f'{out} is in sync\n')
        return 0
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding='utf-8')
    sys.stdout.write(f'wrote {out}\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
