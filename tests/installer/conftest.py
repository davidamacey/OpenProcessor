"""Fixtures for the installer tests; the harness lives in installer_harness.py."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from installer_harness import REPO_ROOT, Shimmed, build_fake_release, run_bash


if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(scope='session')
def fake_release(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return build_fake_release(tmp_path_factory.mktemp('release'))


@pytest.fixture
def shimmed(tmp_path: Path, fake_release: Path) -> Shimmed:
    return Shimmed(root=tmp_path, release=fake_release)


@pytest.fixture
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture
def bash():
    return run_bash
