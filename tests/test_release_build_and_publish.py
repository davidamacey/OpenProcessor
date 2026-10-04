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
    (repo / 'scripts' / 'lib').mkdir(parents=True)
    shutil.copy(
        REPO_ROOT / 'scripts' / 'lib' / 'image_keys.sh', repo / 'scripts' / 'lib' / 'image_keys.sh'
    )
    # The installer's file list: the release script must never touch it (K-2).
    (repo / 'release-manifest.txt').write_text('docker-compose.yml\nimages.lock\n')
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
    assert (sandbox / 'release-manifest.txt').read_text() == 'docker-compose.yml\nimages.lock\n'


def test_dry_run_refuses_push_without_explicit_flag(sandbox: Path, fake_bin: Path) -> None:
    # --push is required to push; a bare invocation with an invalid flag
    # combination should not silently push.
    result = _run(sandbox, fake_bin, ['--only', 'api'])
    assert result.returncode == 2
    assert not (sandbox / 'images.lock').exists()


def test_push_writes_images_lock_with_every_key_and_leaves_the_manifest(
    sandbox: Path, fake_bin: Path
) -> None:
    manifest_before = (sandbox / 'release-manifest.txt').read_text()
    result = _run(sandbox, fake_bin, ['--push', '--only', 'api,triton'])
    assert result.returncode == 0, result.stderr

    lock = sandbox / 'images.lock'
    entries = dict(line.split('=', 1) for line in lock.read_text().strip().splitlines())
    keys = subprocess.run(
        ['bash', '-c', f'source "{REPO_ROOT}/scripts/lib/image_keys.sh"; image_keys third'],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.split()
    assert set(entries) == {'api', 'triton', *keys}
    for key in ('api', 'triton'):
        assert entries[key].startswith('davidamacey/openprocessor')  # default namespace
    assert entries['opensearch'].startswith('opensearchproject/opensearch:3.6.0@sha256:')
    for ref in entries.values():
        assert '@sha256:' in ref
        assert ':latest' not in ref

    # K-2: the installer's file list is untouched; the lock checksum has its own file.
    assert (sandbox / 'release-manifest.txt').read_text() == manifest_before
    sums = (sandbox / 'images.lock.sha256').read_text().split()
    assert sums == [hashlib.sha256(lock.read_bytes()).hexdigest(), 'images.lock']
    calls = (fake_bin.parent / 'docker_calls.log').read_text()
    assert 'pull opensearchproject/opensearch:3.6.0' in calls


def test_refuses_to_write_the_installer_manifest(sandbox: Path, fake_bin: Path) -> None:
    result = _run(
        sandbox,
        fake_bin,
        ['--push', '--only', 'api'],
        env_extra={'IMAGES_LOCK_SUMS_FILE': str(sandbox / 'release-manifest.txt')},
    )
    assert result.returncode == 2
    assert not (sandbox / 'images.lock').exists()


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


def test_failed_docker_build_is_reported_not_logged_as_built(sandbox: Path, fake_bin: Path) -> None:
    _write_shim(fake_bin / 'docker', 'case "$1" in build) exit 1 ;; esac; exit 0')
    result = _run(sandbox, fake_bin, ['--dry-run', '--only', 'api'])
    assert result.returncode == 1
    assert 'builds failed' in result.stderr
    assert 'built ' not in result.stderr.replace('builds failed', '')


def test_segmenter_builds_with_its_own_directory_as_context(
    sandbox: Path, fake_bin: Path, tmp_path: Path
) -> None:
    result = _run(sandbox, fake_bin, ['--dry-run', '--only', 'segmenter,api'])
    assert result.returncode == 0, result.stderr
    builds = [
        ln
        for ln in (tmp_path / 'docker_calls.log').read_text().splitlines()
        if ln.startswith('build ')
    ]
    seg = next(ln for ln in builds if 'docker/segmenter/Dockerfile' in ln)
    api = next(ln for ln in builds if '--file Dockerfile ' in ln)
    assert seg.endswith(' docker/segmenter')
    assert api.endswith(' .')


def test_trivy_critical_finding_fails_the_gate(sandbox: Path, fake_bin: Path) -> None:
    _write_shim(fake_bin / 'trivy', 'exit 10')
    result = _run(sandbox, fake_bin, ['--dry-run', '--only', 'api'])
    assert result.returncode == 1
    assert 'CRITICAL finding' in result.stderr
    assert not (sandbox / 'images.lock').exists()


def test_trivy_scanner_error_is_reported_as_error_not_a_finding(
    sandbox: Path, fake_bin: Path
) -> None:
    _write_shim(fake_bin / 'trivy', 'echo "context deadline exceeded" >&2; exit 1')
    result = _run(sandbox, fake_bin, ['--dry-run', '--only', 'api'])
    assert result.returncode == 1
    assert 'scan ERROR' in result.stderr
    assert 'CRITICAL finding' not in result.stderr


def test_trivy_gets_timeout_and_vuln_only_scanners(sandbox: Path, fake_bin: Path) -> None:
    log = sandbox / 'trivy.args'
    _write_shim(fake_bin / 'trivy', f'echo "$@" > {log}; exit 0')
    result = _run(sandbox, fake_bin, ['--dry-run', '--only', 'api'], {'TRIVY_TIMEOUT': '45m'})
    assert result.returncode == 0, result.stderr
    args = log.read_text()
    assert '--timeout 45m' in args
    assert '--scanners vuln' in args
    assert '--exit-code 10' in args


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


def test_release_lock_round_trips_through_the_installer_parser(
    sandbox: Path, fake_bin: Path
) -> None:
    """K-1: the lock the release script writes is exactly what the installer
    reads: same key names (both come from scripts/lib/image_keys.sh), every
    key present and digest-pinned, none rejected by the installer's validator."""
    result = _run(sandbox, fake_bin, ['--push'])
    assert result.returncode == 0, result.stderr
    lock = sandbox / 'images.lock'
    check = subprocess.run(
        [
            'bash',
            '-c',
            f'OP_SOURCE_ONLY=1 source "{REPO_ROOT}/setup-openprocessor.sh"; '
            f'source "{REPO_ROOT}/scripts/lib/image_keys.sh"; '
            'validate_images_lock "$1" || exit 10; '
            'for k in $(image_keys); do '
            '  ref="$(lock_value "$1" "$k")"; _lock_line_valid "$k=$ref" || { echo "bad $k"; exit 11; }; '
            '  echo "$k $(image_key_field "$k" env) $ref"; '
            'done',
            '_',
            str(lock),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert check.returncode == 0, check.stdout + check.stderr
    rows = [line.split() for line in check.stdout.splitlines()]
    by_key = {key: (env, ref) for key, env, ref in rows}
    assert by_key['api'][0] == 'OP_API_IMAGE'
    assert by_key['api'][1].startswith('davidamacey/openprocessor:1.2.3@sha256:')
    assert by_key['triton'][1].startswith('davidamacey/openprocessor-triton:1.2.3@sha256:')
    assert by_key['opensearch'] == ('OPENSEARCH_IMAGE', by_key['opensearch'][1])
    assert by_key['opensearch'][1].startswith('opensearchproject/opensearch:3.6.0@sha256:')
    assert len(rows) == len(lock.read_text().splitlines())


def _docker_calls(fake_bin: Path) -> list[str]:
    return (fake_bin.parent / 'docker_calls.log').read_text().splitlines()


def test_stable_release_pushes_latest(sandbox: Path, fake_bin: Path) -> None:
    result = _run(sandbox, fake_bin, ['--push', '--only', 'api'])
    assert result.returncode == 0, result.stderr
    assert any(
        c.startswith('push') and c.endswith('openprocessor:latest') for c in _docker_calls(fake_bin)
    )


def test_prerelease_version_is_refused_so_latest_is_stable_only(
    sandbox: Path, fake_bin: Path
) -> None:
    (sandbox / 'VERSION').write_text('1.2.3-rc1\n')
    subprocess.run(['git', 'commit', '-q', '-am', 'rc'], cwd=sandbox, check=True)
    result = _run(sandbox, fake_bin, ['--push', '--only', 'api'])
    assert result.returncode != 0
    assert 'plain X.Y.Z semver' in result.stderr
    log = fake_bin.parent / 'docker_calls.log'
    assert not log.exists() or not any(c.startswith('push') for c in log.read_text().splitlines())
