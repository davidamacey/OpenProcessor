"""Shared fixtures for the repo-guard unit tests: a throwaway git repo to plant violations in."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest


if TYPE_CHECKING:
    from collections.abc import Callable


CODEGEN_DIR = Path(__file__).resolve().parents[2] / 'scripts' / 'codegen'
sys.path.insert(0, str(CODEGEN_DIR))


class ScratchRepo:
    """A git repository in a temp dir; ``write`` stages the file so ``git ls-files`` sees it."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self._git('init', '-q')

    def _git(self, *args: str) -> None:
        subprocess.run(
            ['git', '-c', 'user.name=t', '-c', 'user.email=t@example.com', *args],
            cwd=self.root,
            check=True,
            capture_output=True,
        )

    def write(self, rel: str, text: str) -> Path:
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding='utf-8')
        self._git('add', '-f', rel)
        return path

    def write_lines(self, rel: str, n: int) -> Path:
        return self.write(rel, ''.join(f'x{i} = {i}\n' for i in range(n)))


@pytest.fixture
def scratch_repo(tmp_path: Path) -> ScratchRepo:
    return ScratchRepo(tmp_path)


@pytest.fixture
def baseline_file(tmp_path: Path) -> Callable[[str], Path]:
    def make(text: str) -> Path:
        path = tmp_path / 'baseline.txt'
        path.write_text(text, encoding='utf-8')
        return path

    return make
