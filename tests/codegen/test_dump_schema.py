"""``dump_schema.py`` writes the schema dump the Rust database tests load (no database here: the
commands are faked and their argument lists asserted)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import dump_schema
import pytest


if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

DUMP = 'SET x = 1;\nCREATE TABLE public.jobs (\n    id uuid\n);\n'


class Fake:
    def __init__(self, revision: str = '0009_ingest_retry_at\n', dump: str = DUMP) -> None:
        self.revision = revision
        self.dump = dump
        self.calls: list[tuple[list[str], int]] = []

    def __call__(self, argv: Sequence[str], timeout_s: int) -> str:
        self.calls.append((list(argv), timeout_s))
        return self.dump if 'pg_dump' in argv else self.revision


def _main(tmp_path: Path, fake: Fake, *extra: str) -> int:
    out = tmp_path / 'schema' / 'head.sql'
    return dump_schema.main(['--compose-project', 'harness', '--out', str(out), *extra], run=fake)


def test_the_first_line_names_the_revision_and_the_rest_is_the_dump(tmp_path: Path) -> None:
    assert _main(tmp_path, Fake()) == 0
    text = (tmp_path / 'schema' / 'head.sql').read_text(encoding='utf-8')
    assert text.splitlines()[0] == '-- alembic revision: 0009_ingest_retry_at'
    assert text.endswith(DUMP)


def test_each_database_call_is_one_short_statement_in_the_named_project(tmp_path: Path) -> None:
    fake = Fake()
    _main(tmp_path, fake)
    revision_call, dump_call = fake.calls
    assert revision_call[0][:6] == ['docker', 'compose', '-p', 'harness', 'exec', '-T']
    assert "SET statement_timeout='5s'" in revision_call[0][-1]
    assert revision_call[0][-1].count(';') == 1
    assert {'--schema-only', '--no-owner', '--no-privileges'} <= set(dump_call[0])
    assert dump_call[1] > revision_call[1]


@pytest.mark.parametrize('revision', ['', '0008_a\n0009_b\n'])
def test_zero_or_many_revisions_refuse(tmp_path: Path, revision: str) -> None:
    assert _main(tmp_path, Fake(revision=revision)) == 2
    assert not (tmp_path / 'schema' / 'head.sql').exists()


def test_a_dump_of_another_database_refuses(tmp_path: Path) -> None:
    assert _main(tmp_path, Fake(dump='CREATE TABLE public.other ();')) == 2


def test_check_passes_on_an_equal_file_and_fails_on_a_stale_or_missing_one(tmp_path: Path) -> None:
    assert _main(tmp_path, Fake(), '--check') == 1  # missing
    _main(tmp_path, Fake())
    assert _main(tmp_path, Fake(), '--check') == 0
    assert _main(tmp_path, Fake(revision='0010_next\n'), '--check') == 1  # stale
