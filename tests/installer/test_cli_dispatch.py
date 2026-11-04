"""The management CLI names only subcommands it can run."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
CLI = REPO / 'openprocessor'
PROJECT_PATH = re.compile(
    r'\$\{?PROJECT_DIR\}?/((?:benchmarks|scripts|tests|export|docker|examples|models)/[\w./-]+)'
)
GUARD = re.compile(r'\[\[? +!? *-[fdxe] ')


def _cli_sources() -> list[Path]:
    return [CLI, *sorted((REPO / 'scripts' / 'lib').glob('*.sh'))]


def test_every_repo_path_a_subcommand_runs_exists() -> None:
    missing = []
    for source in _cli_sources():
        lines = source.read_text().splitlines()
        for number, line in enumerate(lines, 1):
            guarded = GUARD.search(line) or (number > 1 and GUARD.search(lines[number - 2]))
            if line.lstrip().startswith('#') or guarded:
                continue
            missing.extend(
                f'{source.name}:{number} {rel}'
                for rel in PROJECT_PATH.findall(line)
                if not (REPO / rel).exists()
            )

    assert not missing, missing


def test_bench_is_not_a_subcommand() -> None:
    dispatch = CLI.read_text()
    assert not re.search(r'^\s*bench\)', dispatch, re.MULTILINE)
    assert 'cmd_bench' not in dispatch


def test_other_subcommands_still_dispatch() -> None:
    dispatch = CLI.read_text()
    for name in ('profile', 'test', 'clean', 'curation', 'faces'):
        assert re.search(rf'^\s*{name}\)', dispatch, re.MULTILINE), name


def test_unknown_subcommand_is_refused() -> None:
    result = subprocess.run(
        [str(CLI), 'bench'], check=False, capture_output=True, text=True, timeout=60
    )
    assert result.returncode != 0
