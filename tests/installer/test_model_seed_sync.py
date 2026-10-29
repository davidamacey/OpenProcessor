"""The upgrade copies changed model files into an installed model repository (#377).

``cp -rn`` kept the old ``gpu_image_prep`` files, so the GPU embed route stayed off after an
upgrade. A model directory that carries a ``MODEL_VERSION`` file is replaced when the seed's version
is newer; everything else is still only added when missing.
"""

from __future__ import annotations

import importlib.util
import subprocess
from typing import TYPE_CHECKING

import pytest
from installer_harness import REPO_ROOT


if TYPE_CHECKING:
    from pathlib import Path
    from types import ModuleType

SCRIPT = REPO_ROOT / 'export' / 'sync_model_seed.py'


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location('sync_model_seed', SCRIPT)
    assert spec
    assert spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write(root: Path, rel: str, text: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


@pytest.fixture
def repos(tmp_path: Path) -> tuple[Path, Path]:
    seed, dest = tmp_path / 'seed', tmp_path / 'models'
    dest.mkdir()
    return seed, dest


def test_a_newer_seed_version_replaces_the_model_files(repos: tuple[Path, Path]) -> None:
    seed, dest = repos
    _write(seed, 'gpu_image_prep/MODEL_VERSION', '2\n')
    _write(seed, 'gpu_image_prep/1/model.py', 'new')
    _write(seed, 'gpu_image_prep/config.pbtxt', 'new config')
    _write(dest, 'gpu_image_prep/1/model.py', 'old')
    _write(dest, 'gpu_image_prep/config.pbtxt', 'old config')
    updated = _load().sync(seed, dest)
    assert updated == ['gpu_image_prep']
    assert (dest / 'gpu_image_prep/1/model.py').read_text() == 'new'
    assert (dest / 'gpu_image_prep/config.pbtxt').read_text() == 'new config'
    assert (dest / 'gpu_image_prep/MODEL_VERSION').read_text().strip() == '2'


def test_an_install_without_a_version_file_is_version_zero(repos: tuple[Path, Path]) -> None:
    seed, dest = repos
    _write(seed, 'm/MODEL_VERSION', '1')
    _write(seed, 'm/1/model.py', 'new')
    _write(dest, 'm/1/model.py', 'old')
    assert _load().sync(seed, dest) == ['m']
    assert (dest / 'm/1/model.py').read_text() == 'new'


def test_the_same_or_a_newer_installed_version_is_left_alone(repos: tuple[Path, Path]) -> None:
    seed, dest = repos
    _write(seed, 'm/MODEL_VERSION', '2')
    _write(seed, 'm/1/model.py', 'seed')
    _write(dest, 'm/MODEL_VERSION', '2')
    _write(dest, 'm/1/model.py', 'edited')
    assert _load().sync(seed, dest) == []
    assert (dest / 'm/1/model.py').read_text() == 'edited'


def test_unversioned_files_are_only_added_never_replaced(repos: tuple[Path, Path]) -> None:
    seed, dest = repos
    _write(seed, 'plain/config.pbtxt', 'seed')
    _write(seed, 'plain/labels.txt', 'seed labels')
    _write(dest, 'plain/config.pbtxt', 'profile-edited')
    assert _load().sync(seed, dest) == []
    assert (dest / 'plain/config.pbtxt').read_text() == 'profile-edited'
    assert (dest / 'plain/labels.txt').read_text() == 'seed labels'


def test_built_engines_beside_replaced_files_survive(repos: tuple[Path, Path]) -> None:
    seed, dest = repos
    _write(seed, 'm/MODEL_VERSION', '3')
    _write(seed, 'm/1/model.py', 'new')
    _write(dest, 'm/MODEL_VERSION', '1')
    _write(dest, 'm/1/model.plan', 'engine')
    _load().sync(seed, dest)
    assert (dest / 'm/1/model.plan').read_text() == 'engine'


def test_the_shipped_gpu_image_prep_carries_a_version() -> None:
    text = (REPO_ROOT / 'models' / 'gpu_image_prep' / 'MODEL_VERSION').read_text()
    assert int(text.strip()) >= 1


def test_the_command_line_entry_point_reports_updated_models(repos: tuple[Path, Path]) -> None:
    seed, dest = repos
    _write(seed, 'm/MODEL_VERSION', '1')
    _write(seed, 'm/1/model.py', 'new')
    done = subprocess.run(
        ['python3', str(SCRIPT), str(seed), str(dest)],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert done.returncode == 0, done.stderr
    assert 'updated model files: m' in done.stdout


def test_the_preflight_group_syncs_instead_of_cp_rn() -> None:
    script = (
        f'source {REPO_ROOT}/scripts/lib/model_setup.sh; '
        'runner() { local n=$1; shift; echo "STEP $n $*"; }; '
        '_ms_run_group_steps preflight runner'
    )
    done = subprocess.run(
        ['bash', '-c', script], capture_output=True, text=True, check=False, timeout=30
    )
    assert done.returncode == 0, done.stderr
    seed = next(ln for ln in done.stdout.splitlines() if ln.startswith('STEP preflight_seed'))
    assert '/app/export/sync_model_seed.py /opt/openprocessor/model_repo_seed /app/models' in seed
    assert '-rn' not in seed
