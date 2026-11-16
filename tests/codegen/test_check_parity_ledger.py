"""check_parity_ledger.py: a deleted Python test needs a row saying what replaces it."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import check_parity_ledger as guard
import pytest


if TYPE_CHECKING:
    from conftest import ScratchRepo

OLD = 'def test_a():\n    pass\n\n\ndef test_b():\n    pass\n\n\nclass TestK:\n    def test_m(self):\n        pass\n'
LEDGER = 'tests/parity/ledger/r5.tsv'


def git(repo: ScratchRepo, *args: str) -> None:
    subprocess.run(
        ['git', '-c', 'user.name=t', '-c', 'user.email=t@example.com', *args],
        cwd=repo.root,
        check=True,
        capture_output=True,
    )


def committed(repo: ScratchRepo, files: dict[str, str]) -> None:
    for rel, text in files.items():
        repo.write(rel, text)
    git(repo, 'commit', '-q', '-m', 'base')


def run(repo: ScratchRepo, *extra: str) -> int:
    return guard.main(['--root', str(repo.root), *extra])


def test_deleting_a_test_file_without_rows_fails_and_prints_the_rows_to_fill(
    scratch_repo: ScratchRepo, capsys: pytest.CaptureFixture[str]
) -> None:
    committed(scratch_repo, {'tests/test_old.py': OLD})
    git(scratch_repo, 'rm', '-q', 'tests/test_old.py')

    assert run(scratch_repo) == 1

    out = capsys.readouterr().out
    for test_id in (
        'tests/test_old.py::test_a',
        'tests/test_old.py::test_b',
        'tests/test_old.py::TestK::test_m',
    ):
        assert test_id in out


def test_every_deleted_test_with_a_row_passes(scratch_repo: ScratchRepo) -> None:
    committed(scratch_repo, {'tests/test_old.py': OLD})
    git(scratch_repo, 'rm', '-q', 'tests/test_old.py')
    scratch_repo.write(
        LEDGER,
        'tests/test_old.py::test_a\tgone: the route is removed on purpose\n'
        'tests/test_old.py::test_b\tcontract: contracts/openapi/curation.json the shape of the answer\n'
        'tests/test_old.py::TestK::test_m\tgone: covered by the schema\n',
    )
    scratch_repo.write('contracts/openapi/curation.json', '{"paths": {}}')

    assert run(scratch_repo) == 0


def test_removing_one_function_from_a_file_needs_only_that_row(
    scratch_repo: ScratchRepo, capsys: pytest.CaptureFixture[str]
) -> None:
    committed(scratch_repo, {'tests/test_old.py': OLD})
    scratch_repo.write('tests/test_old.py', 'def test_a():\n    pass\n')

    assert run(scratch_repo) == 1

    out = capsys.readouterr().out
    assert 'test_old.py::test_b' in out
    assert 'test_old.py::TestK::test_m' in out
    assert 'test_old.py::test_a\t' not in out


def test_a_file_move_that_keeps_the_names_needs_no_row(scratch_repo: ScratchRepo) -> None:
    committed(scratch_repo, {'tests/test_old.py': OLD})
    git(scratch_repo, 'mv', 'tests/test_old.py', 'tests/test_new.py')

    assert run(scratch_repo) == 0


def test_a_renamed_function_needs_a_row(scratch_repo: ScratchRepo) -> None:
    committed(scratch_repo, {'tests/test_old.py': OLD})
    scratch_repo.write('tests/test_old.py', OLD.replace('def test_b', 'def test_b_renamed'))

    assert run(scratch_repo) == 1


def test_a_commit_that_deletes_no_test_passes(scratch_repo: ScratchRepo) -> None:
    committed(scratch_repo, {'tests/test_old.py': OLD, 'tests/helpers.py': 'X = 1\n'})
    scratch_repo.write('tests/test_old.py', OLD + '\n\ndef test_c():\n    pass\n')
    git(scratch_repo, 'rm', '-q', 'tests/helpers.py')

    assert run(scratch_repo) == 0


def test_a_commit_with_no_head_yet_passes(scratch_repo: ScratchRepo) -> None:
    scratch_repo.write('tests/test_old.py', OLD)

    assert run(scratch_repo) == 0


@pytest.mark.parametrize(
    'row',
    [
        'tests/test_old.py::test_a',
        'tests/test_old.py::test_a\t',
        'tests/test_old.py::test_a\tsomething else entirely',
        'tests/test_old.py::test_a\tgone:',
        'tests/test_old.py::test_a\tgone:   ',
        'tests/test_old.py::test_a\trust: native/x/tests/none.rs::test_a',
        'tests/test_old.py::test_a\tcontract: contracts/openapi/none.json nothing here',
        'tests/test_old.py::test_a\tdiff: nofamily/some_operation_id',
        'tests/test_old.py::test_a\tdiff: projects/matches_no_operation_*',
    ],
)
def test_a_malformed_or_unverifiable_row_fails(scratch_repo: ScratchRepo, row: str) -> None:
    committed(scratch_repo, {'tests/test_other.py': 'def test_z():\n    pass\n'})
    scratch_repo.write(LEDGER, row + '\n')
    scratch_repo.write(
        'contracts/openapi/curation.json', '{"paths": {"/x": {"get": {"operationId": "op_x"}}}}'
    )
    scratch_repo.write(
        'tests/differential/families.toml',
        '[[family]]\nname = "projects"\ncontract = "curation"\nslice = "R11a"\npaths = ["/x"]\n',
    )

    assert run(scratch_repo) == 1


def test_rows_that_point_at_something_real_pass(scratch_repo: ScratchRepo) -> None:
    committed(scratch_repo, {'tests/test_old.py': OLD})
    git(scratch_repo, 'rm', '-q', 'tests/test_old.py')
    scratch_repo.write('native/op_domain/tests/fold.rs', '#[test]\nfn folds_a_claim() {}\n')
    scratch_repo.write(
        'contracts/openapi/curation.json',
        json.dumps({'paths': {'/x': {'get': {'operationId': 'list_x_get'}}}}),
    )
    scratch_repo.write(
        'tests/differential/families.toml',
        '[[family]]\nname = "projects"\ncontract = "curation"\nslice = "R11a"\npaths = ["/x"]\n',
    )
    scratch_repo.write(
        LEDGER,
        'tests/test_old.py::test_a\trust: native/op_domain/tests/fold.rs::folds_a_claim\n'
        'tests/test_old.py::test_b\tdiff: projects/list_x_*\n'
        'tests/test_old.py::TestK::test_m\tgone: removed with the route\n',
    )

    assert run(scratch_repo) == 0


def test_a_row_for_a_test_that_still_exists_fails(scratch_repo: ScratchRepo) -> None:
    committed(scratch_repo, {'tests/test_old.py': OLD})
    scratch_repo.write(LEDGER, 'tests/test_old.py::test_a\tgone: but it is still here\n')

    assert run(scratch_repo) == 1


def test_the_same_test_in_two_ledger_files_fails(scratch_repo: ScratchRepo) -> None:
    committed(scratch_repo, {'tests/test_other.py': 'def test_z():\n    pass\n'})
    scratch_repo.write('tests/parity/ledger/r4.tsv', 'tests/test_gone.py::test_q\tgone: one\n')
    scratch_repo.write('tests/parity/ledger/r5.tsv', 'tests/test_gone.py::test_q\tgone: two\n')

    assert run(scratch_repo) == 1


def test_comments_and_blank_lines_in_a_ledger_are_skipped(scratch_repo: ScratchRepo) -> None:
    committed(scratch_repo, {'tests/test_other.py': 'def test_z():\n    pass\n'})
    scratch_repo.write(LEDGER, '# slice R5\n\ntests/test_gone.py::test_q\tgone: removed\n')

    assert run(scratch_repo) == 0


def test_between_two_commits_finds_what_a_range_deleted(scratch_repo: ScratchRepo) -> None:
    committed(scratch_repo, {'tests/test_old.py': OLD})
    git(scratch_repo, 'rm', '-q', 'tests/test_old.py')
    git(scratch_repo, 'commit', '-q', '-m', 'delete')

    assert run(scratch_repo, '--between', 'HEAD~1', 'HEAD') == 1
    assert run(scratch_repo) == 0


def test_the_committed_ledger_directory_is_valid() -> None:
    root = Path(__file__).resolve().parents[2]
    assert guard.main(['--root', str(root), '--ledger-only']) == 0
