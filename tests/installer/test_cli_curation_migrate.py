"""`./openprocessor curation migrate-registry`: it forwards exactly the flags it was given to
the migration module, needs OpenSearch up, and refuses anything else with exit 2."""

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


def test_all_dry_run_runs_the_module_with_exactly_those_flags(shimmed: Shimmed, inst: Path) -> None:
    shimmed.flag('allow_mutations')
    shimmed.flag('opensearch_container')
    done = cli(shimmed, inst, 'curation', 'migrate-registry', '--all', '--dry-run')
    assert done.returncode == 0, done.stderr
    runs = _run_lines(shimmed)
    assert runs, shimmed.mutating_docker_calls()
    assert 'scripts.curation.migrate_class_registry' in runs[-1]
    assert runs[-1].rstrip().endswith('--all --dry-run')


def test_project_and_reverse_are_forwarded(shimmed: Shimmed, inst: Path) -> None:
    shimmed.flag('allow_mutations')
    shimmed.flag('opensearch_container')
    done = cli(shimmed, inst, 'curation', 'migrate-registry', '--project', 'demo', '--reverse')
    assert done.returncode == 0, done.stderr
    assert _run_lines(shimmed)[-1].rstrip().endswith('--project demo --reverse')


@pytest.mark.parametrize('args', [['--bogus'], ['--all', 'extra'], ['--project']])
def test_unknown_usage_exits_2_without_running(
    shimmed: Shimmed, inst: Path, args: list[str]
) -> None:
    shimmed.flag('allow_mutations')
    shimmed.flag('opensearch_container')
    done = cli(shimmed, inst, 'curation', 'migrate-registry', *args)
    assert done.returncode == 2
    assert _run_lines(shimmed) == []


def test_it_says_so_when_opensearch_is_down(shimmed: Shimmed, inst: Path) -> None:
    shimmed.flag('allow_mutations')
    done = cli(shimmed, inst, 'curation', 'migrate-registry', '--all')
    assert done.returncode == 1
    assert 'OpenSearch is not running' in done.stderr + done.stdout
    assert _run_lines(shimmed) == []
