"""`./openprocessor ocr purge`: it forwards exactly what the user asked for."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import test_cli_project
from test_cli_project import cli


if TYPE_CHECKING:
    from pathlib import Path

    from installer_harness import Shimmed

inst = test_cli_project.inst


def _run_lines(shimmed: Shimmed) -> list[str]:
    return [ln for ln in shimmed.mutating_docker_calls() if ' run ' in ln]


def test_purge_without_flags_forwards_none(shimmed: Shimmed, inst: Path) -> None:
    shimmed.flag('allow_mutations')
    done = cli(shimmed, inst, 'ocr', 'purge')
    assert done.returncode == 0, done.stderr
    runs = _run_lines(shimmed)
    assert runs, shimmed.mutating_docker_calls()
    assert 'scripts.purge_ocr_data' in runs[-1]
    assert '--yes' not in runs[-1]
    assert '--dry-run' not in runs[-1]


@pytest.mark.parametrize('flag', ['--dry-run', '--yes'])
def test_a_flag_is_forwarded(shimmed: Shimmed, inst: Path, flag: str) -> None:
    shimmed.flag('allow_mutations')
    assert cli(shimmed, inst, 'ocr', 'purge', flag).returncode == 0
    assert _run_lines(shimmed)[-1].rstrip().endswith(flag)


@pytest.mark.parametrize('args', [['purge', '--bogus'], ['wipe'], []])
def test_unknown_usage_exits_2_without_running(
    shimmed: Shimmed, inst: Path, args: list[str]
) -> None:
    shimmed.flag('allow_mutations')
    done = cli(shimmed, inst, 'ocr', *args)
    assert done.returncode == 2
    assert _run_lines(shimmed) == []
