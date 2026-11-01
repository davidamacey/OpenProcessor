"""docker/buildlib/*.sh: the build-time helpers the multi-stage Dockerfile uses to ship an extra
dependency set as a file-level delta and to share the identical packages of the export venv."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
DELTA = REPO_ROOT / 'docker' / 'buildlib' / 'site_delta.sh'
DEDUPE = REPO_ROOT / 'docker' / 'buildlib' / 'dedupe_site_packages.sh'


def _run(script: Path, *args: object) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # nosec B603 B607 - fixed repo script, tmp_path arguments
        ['bash', str(script), *map(str, args)], capture_output=True, text=True, check=False
    )


def _write(root: Path, rel: str, text: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def test_delta_copies_only_files_added_after_the_snapshot(tmp_path: Path) -> None:
    site = tmp_path / 'local'
    _write(site, 'lib/python3.13/site-packages/nvidia/cublas/lib.so', 'runtime')
    _write(site, 'lib/python3.13/site-packages/torch/__init__.py', 'torch')
    assert _run(DELTA, 'snapshot', site, tmp_path / 'pre.sha').returncode == 0

    _write(site, 'lib/python3.13/site-packages/nvidia/tensorrt/lib.so', 'export')
    _write(site, 'lib/python3.13/site-packages/modelopt/__init__.py', 'modelopt')
    (site / 'bin').mkdir()
    (site / 'bin' / 'polygraphy').symlink_to('../lib/python3.13/site-packages/modelopt')

    out = tmp_path / 'delta'
    result = _run(DELTA, 'delta', site, tmp_path / 'pre.sha', out)
    assert result.returncode == 0, result.stderr
    copied = sorted(
        str(p.relative_to(out)) for p in out.rglob('*') if p.is_file() or p.is_symlink()
    )
    assert copied == [
        'bin/polygraphy',
        'lib/python3.13/site-packages/modelopt/__init__.py',
        'lib/python3.13/site-packages/nvidia/tensorrt/lib.so',
    ]
    assert (out / 'bin' / 'polygraphy').is_symlink()


def test_delta_fails_when_the_extra_install_rewrote_a_runtime_file(tmp_path: Path) -> None:
    site = tmp_path / 'local'
    _write(site, 'lib/python3.13/site-packages/numpy/__init__.py', 'v1')
    assert _run(DELTA, 'snapshot', site, tmp_path / 'pre.sha').returncode == 0
    _write(site, 'lib/python3.13/site-packages/numpy/__init__.py', 'v2')

    result = _run(DELTA, 'delta', site, tmp_path / 'pre.sha', tmp_path / 'delta')
    assert result.returncode != 0
    assert 'numpy/__init__.py' in result.stderr


def test_delta_fails_when_the_extra_install_removed_a_runtime_file(tmp_path: Path) -> None:
    site = tmp_path / 'local'
    _write(site, 'lib/python3.13/site-packages/six.py', 'six')
    assert _run(DELTA, 'snapshot', site, tmp_path / 'pre.sha').returncode == 0
    (site / 'lib/python3.13/site-packages/six.py').unlink()

    result = _run(DELTA, 'delta', site, tmp_path / 'pre.sha', tmp_path / 'delta')
    assert result.returncode != 0
    assert 'six.py' in result.stderr


def test_delta_ignores_bytecode_caches(tmp_path: Path) -> None:
    site = tmp_path / 'local'
    _write(site, 'lib/python3.13/site-packages/a/__init__.py', 'a')
    assert _run(DELTA, 'snapshot', site, tmp_path / 'pre.sha').returncode == 0
    _write(site, 'lib/python3.13/site-packages/a/__pycache__/__init__.cpython-313.pyc', 'x')

    out = tmp_path / 'delta'
    assert _run(DELTA, 'delta', site, tmp_path / 'pre.sha', out).returncode == 0
    assert not list(out.rglob('*.pyc'))


def test_delta_rejects_unknown_subcommand(tmp_path: Path) -> None:
    assert _run(DELTA, 'frobnicate', tmp_path).returncode != 0


def _tree(root: Path, rel: str, files: dict[str, str]) -> None:
    for name, text in files.items():
        _write(root, f'{rel}/{name}', text)


def test_dedupe_symlinks_identical_entries_and_keeps_differing_ones(tmp_path: Path) -> None:
    canon_build = tmp_path / 'canon-build'
    dup = tmp_path / 'venv-site'
    runtime = '/home/appuser/.local/lib/python3.13/site-packages'
    for site in (canon_build, dup):
        _tree(site, 'onnx', {'__init__.py': 'same', 'sub/mod.py': 'same'})
        _write(site, 'six.py', 'six')
    _tree(canon_build, 'torch', {'__init__.py': 'cuda'})
    _tree(dup, 'torch', {'__init__.py': 'cpu'})
    _write(dup, 'only_in_venv.py', 'x')
    _write(dup, 'pkg-1.0.dist-info/METADATA', 'meta')
    _write(canon_build, 'pkg-1.0.dist-info/METADATA', 'meta')

    result = _run(DEDUPE, dup, canon_build, runtime)
    assert result.returncode == 0, result.stderr

    assert (dup / 'onnx').is_symlink()
    assert (dup / 'onnx').readlink() == Path(runtime) / 'onnx'
    assert (dup / 'six.py').is_symlink()
    assert (dup / 'six.py').readlink() == Path(runtime) / 'six.py'
    # a differing package, an entry only the venv has, and dist-info stay real
    assert not (dup / 'torch').is_symlink()
    assert (dup / 'torch' / '__init__.py').read_text() == 'cpu'
    assert not (dup / 'only_in_venv.py').is_symlink()
    assert not (dup / 'pkg-1.0.dist-info').is_symlink()


def test_dedupe_leaves_an_entry_that_differs_deep_inside(tmp_path: Path) -> None:
    canon_build = tmp_path / 'canon-build'
    dup = tmp_path / 'venv-site'
    _tree(canon_build, 'pkg', {'a.py': 'a', 'deep/b.py': 'b1'})
    _tree(dup, 'pkg', {'a.py': 'a', 'deep/b.py': 'b2'})

    assert _run(DEDUPE, dup, canon_build, '/rt').returncode == 0
    assert not (dup / 'pkg').is_symlink()


def test_dedupe_is_idempotent_and_skips_pycache(tmp_path: Path) -> None:
    canon_build = tmp_path / 'canon-build'
    dup = tmp_path / 'venv-site'
    _tree(canon_build, 'pkg', {'a.py': 'a'})
    _tree(dup, 'pkg', {'a.py': 'a'})
    _write(dup, '__pycache__/x.pyc', 'x')
    _write(canon_build, '__pycache__/x.pyc', 'x')

    assert _run(DEDUPE, dup, canon_build, '/rt').returncode == 0
    assert _run(DEDUPE, dup, canon_build, '/rt').returncode == 0
    assert (dup / 'pkg').is_symlink()
    assert not (dup / '__pycache__').is_symlink()


@pytest.mark.parametrize('script', [DELTA, DEDUPE])
def test_helpers_are_strict_bash(script: Path) -> None:
    text = script.read_text()
    assert text.startswith('#!/bin/bash')
    assert 'set -euo pipefail' in text
