"""Tests for scripts/release/build_and_publish.sh (Wave 4, installer plan §11.1).

No real docker/trivy work happens here: this builds a throwaway sandbox repo
(VERSION + a git init + stub Dockerfiles) and puts fake `docker`, `trivy`, and
`git` shims first on PATH so the script's control flow — gating, tagging,
images.lock / release-manifest.txt generation — is exercised deterministically
and fast. Nothing here builds, scans, or pushes a real image.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / 'scripts' / 'release' / 'build_and_publish.sh'


def _write_shim(path: Path, body: str) -> None:
    path.write_text(f'#!/bin/bash\n{body}\n')
    path.chmod(path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)


@pytest.fixture
def sandbox(tmp_path: Path) -> Path:
    """A minimal repo: VERSION, one Dockerfile per image spec, git-initted."""
    repo = tmp_path / 'repo'
    (repo / 'docker' / 'evaluator').mkdir(parents=True)
    (repo / 'docker' / 'segmenter').mkdir(parents=True)
    (repo / 'docker' / 'trainer').mkdir(parents=True)
    for df in [
        'Dockerfile',
        'Dockerfile.triton',
        'docker/evaluator/Dockerfile',
        'docker/segmenter/Dockerfile',
        'docker/trainer/Dockerfile',
    ]:
        (repo / df).write_text('FROM scratch\n')
    (repo / 'VERSION').write_text('1.2.3\n')

    subprocess.run(['git', 'init', '-q'], cwd=repo, check=True)
    subprocess.run(['git', 'config', 'user.email', 'test@example.com'], cwd=repo, check=True)
    subprocess.run(['git', 'config', 'user.name', 'test'], cwd=repo, check=True)
    subprocess.run(['git', 'add', '-A'], cwd=repo, check=True)
    subprocess.run(['git', 'commit', '-q', '-m', 'init'], cwd=repo, check=True)

    release_dir = repo / 'scripts' / 'release'
    release_dir.mkdir(parents=True)
    shutil.copy(SCRIPT, release_dir / 'build_and_publish.sh')
    (release_dir / 'build_and_publish.sh').chmod(0o755)
    (release_dir / 'trivy-allowlist.txt').write_text('# no waivers\n')

    subprocess.run(['git', 'add', '-A'], cwd=repo, check=True)
    subprocess.run(['git', 'commit', '-q', '-m', 'add release script'], cwd=repo, check=True)
    return repo


@pytest.fixture
def fake_bin(tmp_path: Path) -> Path:
    """PATH dir with shims. docker/trivy default to trivially succeeding."""
    bindir = tmp_path / 'bin'
    bindir.mkdir()

    calls_log = tmp_path / 'docker_calls.log'

    _write_shim(
        bindir / 'docker',
        f"""
echo "$@" >> "{calls_log}"
case "$1" in
    build) exit 0 ;;
    push) exit 0 ;;
    image)
        case "$2" in
            inspect)
                # last arg is the tag; fabricate a stable digest from it
                tag="${{@: -1}}"
                digest=$(printf '%s' "$tag" | sha256sum | cut -d' ' -f1)
                echo "someregistry/repo@sha256:${{digest}}"
                exit 0
                ;;
        esac
        ;;
esac
exit 0
""",
    )
    _write_shim(bindir / 'trivy', 'exit 0')

    return bindir


def _run(sandbox: Path, fake_bin: Path, args: list[str], env_extra: dict[str, str] | None = None):
    env = os.environ.copy()
    env['PATH'] = f'{fake_bin}:{env["PATH"]}'
    env.pop('OP_IMAGE_NAMESPACE', None)
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        [str(sandbox / 'scripts' / 'release' / 'build_and_publish.sh'), *args],
        check=False,
        cwd=sandbox,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_no_mode_flag_is_misuse(sandbox: Path, fake_bin: Path) -> None:
    result = _run(sandbox, fake_bin, [])
    assert result.returncode == 2
    assert 'usage' in result.stderr.lower() or '--dry-run' in result.stderr


def test_unknown_only_service_is_misuse(sandbox: Path, fake_bin: Path) -> None:
    result = _run(sandbox, fake_bin, ['--dry-run', '--only', 'bogus'])
    assert result.returncode == 2


def test_dry_run_builds_and_scans_never_pushes(sandbox: Path, fake_bin: Path) -> None:
    result = _run(sandbox, fake_bin, ['--dry-run', '--only', 'api,triton'])
    assert result.returncode == 0, result.stderr
    calls = (fake_bin.parent / 'docker_calls.log').read_text()
    assert 'build' in calls
    assert 'push' not in calls
    assert not (sandbox / 'images.lock').exists()
    assert not (sandbox / 'release-manifest.txt').exists()


def test_dry_run_refuses_push_without_explicit_flag(sandbox: Path, fake_bin: Path) -> None:
    # --push is required to push; a bare invocation with an invalid flag
    # combination should not silently push.
    result = _run(sandbox, fake_bin, ['--only', 'api'])
    assert result.returncode == 2
    assert not (sandbox / 'images.lock').exists()


def test_push_writes_images_lock_and_manifest(sandbox: Path, fake_bin: Path) -> None:
    result = _run(sandbox, fake_bin, ['--push', '--only', 'api,triton'])
    assert result.returncode == 0, result.stderr

    lock = sandbox / 'images.lock'
    manifest = sandbox / 'release-manifest.txt'
    assert lock.exists()
    assert manifest.exists()

    lock_lines = sorted(lock.read_text().strip().splitlines())
    assert len(lock_lines) == 2
    for line in lock_lines:
        key, rest = line.split('=', 1)
        assert key in {'api', 'triton'}
        assert '@sha256:' in rest
        assert rest.startswith('davidamacey/openprocessor')  # default namespace

    # release-manifest.txt round trip: the recorded sha256 matches the file.
    manifest_lines = manifest.read_text().strip().splitlines()
    assert len(manifest_lines) == 1
    name, recorded_sha = manifest_lines[0].split('\t')
    assert name == 'images.lock'
    actual_sha = hashlib.sha256(lock.read_bytes()).hexdigest()
    assert recorded_sha == actual_sha


def test_namespace_override_applies_to_lock(sandbox: Path, fake_bin: Path) -> None:
    result = _run(sandbox, fake_bin, ['--push', '--only', 'api', '--namespace', 'acmeco'])
    assert result.returncode == 0, result.stderr
    lock = (sandbox / 'images.lock').read_text()
    assert 'acmeco/openprocessor:' in lock
    assert 'davidamacey' not in lock


def test_namespace_env_var_default(sandbox: Path, fake_bin: Path) -> None:
    result = _run(
        sandbox, fake_bin, ['--push', '--only', 'api'], env_extra={'OP_IMAGE_NAMESPACE': 'envns'}
    )
    assert result.returncode == 0, result.stderr
    lock = (sandbox / 'images.lock').read_text()
    assert 'envns/openprocessor:' in lock


def test_refuses_dirty_worktree_on_push(sandbox: Path, fake_bin: Path) -> None:
    (sandbox / 'VERSION').write_text('1.2.3\n#dirty\n')
    result = _run(sandbox, fake_bin, ['--push', '--only', 'api'])
    assert result.returncode == 1
    assert 'dirty' in result.stderr.lower()
    assert not (sandbox / 'images.lock').exists()


def test_dirty_worktree_allowed_for_dry_run_with_allow_dirty(sandbox: Path, fake_bin: Path) -> None:
    (sandbox / 'README.md').write_text('scratch\n')
    result = _run(sandbox, fake_bin, ['--dry-run', '--only', 'api', '--allow-dirty'])
    assert result.returncode == 0, result.stderr


def test_version_tag_mismatch_is_a_gate_failure(sandbox: Path, fake_bin: Path) -> None:
    result = _run(sandbox, fake_bin, ['--push', '--only', 'api', '--version', 'v9.9.9'])
    assert result.returncode == 1
    assert '9.9.9' in result.stderr


def test_version_mismatch_allowed_with_allow_dirty(sandbox: Path, fake_bin: Path) -> None:
    result = _run(
        sandbox,
        fake_bin,
        ['--dry-run', '--only', 'api', '--version', 'v9.9.9', '--allow-dirty'],
    )
    assert result.returncode == 0, result.stderr


def test_trivy_critical_finding_fails_the_gate(sandbox: Path, fake_bin: Path) -> None:
    _write_shim(fake_bin / 'trivy', 'exit 1')
    result = _run(sandbox, fake_bin, ['--dry-run', '--only', 'api'])
    assert result.returncode == 1
    assert not (sandbox / 'images.lock').exists()


def test_allowlist_entry_without_reason_is_rejected(sandbox: Path, fake_bin: Path) -> None:
    (sandbox / 'scripts' / 'release' / 'trivy-allowlist.txt').write_text('CVE-2024-00000\n')
    result = _run(sandbox, fake_bin, ['--dry-run', '--only', 'api'])
    assert result.returncode == 1
    assert 'reason' in result.stderr.lower()


def test_missing_trivy_binary_warns_but_does_not_block_dry_run(
    sandbox: Path, tmp_path: Path
) -> None:
    bindir = tmp_path / 'bin_no_trivy'
    bindir.mkdir()
    _write_shim(
        bindir / 'docker',
        """
case "$1" in
    build) exit 0 ;;
    *) exit 0 ;;
esac
""",
    )
    orig_dirs = [
        d for d in os.environ['PATH'].split(os.pathsep) if not (Path(d) / 'trivy').exists()
    ]
    env = os.environ.copy()
    env['PATH'] = os.pathsep.join([str(bindir), *orig_dirs])
    result = subprocess.run(
        [
            str(sandbox / 'scripts' / 'release' / 'build_and_publish.sh'),
            '--dry-run',
            '--only',
            'api',
        ],
        check=False,
        cwd=sandbox,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert 'not found' in result.stderr.lower()
