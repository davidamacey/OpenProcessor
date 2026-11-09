"""docker/buildlib/site_delta.sh: the build-time helper the multi-stage Dockerfile uses to ship an
extra dependency set as a file-level delta."""

from __future__ import annotations

import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DELTA = REPO_ROOT / 'docker' / 'buildlib' / 'site_delta.sh'


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


def test_the_helper_is_strict_bash() -> None:
    text = DELTA.read_text()
    assert text.startswith('#!/bin/bash')
    assert 'set -euo pipefail' in text
