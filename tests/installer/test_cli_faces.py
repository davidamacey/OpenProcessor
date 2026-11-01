"""`./openprocessor faces purge`: it forwards exactly what the user asked for and refuses a real
purge while faces are on."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import test_cli_project
from test_cli_project import cli


if TYPE_CHECKING:
    from pathlib import Path

    from installer_harness import Shimmed

inst = test_cli_project.inst


def _faces_off(inst: Path) -> None:
    with (inst / '.env').open('a') as fh:
        fh.write('\nOP_FACES_ENABLED=false\n')


def _run_lines(shimmed: Shimmed) -> list[str]:
    return [ln for ln in shimmed.mutating_docker_calls() if ' run ' in ln]


def test_purge_without_flags_forwards_none(shimmed: Shimmed, inst: Path) -> None:
    shimmed.flag('allow_mutations')
    done = cli(shimmed, inst, 'faces', 'purge')
    assert done.returncode == 0, done.stderr
    runs = _run_lines(shimmed)
    assert runs, shimmed.mutating_docker_calls()
    assert 'scripts.purge_face_data' in runs[-1]
    assert '--yes' not in runs[-1]
    assert '--dry-run' not in runs[-1]


def test_dry_run_is_forwarded(shimmed: Shimmed, inst: Path) -> None:
    shimmed.flag('allow_mutations')
    assert cli(shimmed, inst, 'faces', 'purge', '--dry-run').returncode == 0
    assert _run_lines(shimmed)[-1].rstrip().endswith('--dry-run')


def test_yes_is_refused_while_faces_are_on(shimmed: Shimmed, inst: Path) -> None:
    shimmed.flag('allow_mutations')
    done = cli(shimmed, inst, 'faces', 'purge', '--yes')
    assert done.returncode == 2
    assert 'OP_FACES_ENABLED' in done.stderr + done.stdout
    assert _run_lines(shimmed) == []


def test_yes_runs_once_faces_are_off(shimmed: Shimmed, inst: Path) -> None:
    shimmed.flag('allow_mutations')
    _faces_off(inst)
    done = cli(shimmed, inst, 'faces', 'purge', '--yes')
    assert done.returncode == 0, done.stderr
    assert _run_lines(shimmed)[-1].rstrip().endswith('--yes')


@pytest.mark.parametrize('args', [['--bogus'], ['wipe'], []])
def test_unknown_usage_exits_2_without_running(
    shimmed: Shimmed, inst: Path, args: list[str]
) -> None:
    shimmed.flag('allow_mutations')
    done = cli(shimmed, inst, 'faces', *(['purge', *args] if args == ['--bogus'] else args))
    assert done.returncode == 2
    assert _run_lines(shimmed) == []
