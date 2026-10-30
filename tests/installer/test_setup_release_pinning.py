"""Release artifact verification, image pinning and version/branch modes."""

from __future__ import annotations

import json
import re
import shutil
from typing import TYPE_CHECKING

import pytest
from _setup_dry_run_helpers import configure, dry, env_file
from installer_harness import PROJECT, RELEASE, build_fake_release, fake_digest, image_key_refs


if TYPE_CHECKING:
    from pathlib import Path

    from installer_harness import Shimmed


def _release_copy(shimmed: Shimmed, tmp_path: Path) -> Path:
    release = tmp_path / 'rel'
    shutil.copytree(shimmed.release, release)
    shimmed.release = release
    return release / 'assets' / RELEASE


def test_truncated_tarball_fails_before_anything_is_installed(
    shimmed: Shimmed, tmp_path: Path
) -> None:
    assets = _release_copy(shimmed, tmp_path)
    tarball = assets / f'openprocessor-deploy-{RELEASE}.tar.gz'
    data = tarball.read_bytes()
    tarball.write_bytes(data[: len(data) // 2])
    result = dry(shimmed)
    assert result.returncode == 7
    inst = shimmed.root / 'inst'
    assert not (inst / '.env').exists()
    assert not (inst / 'docker-compose.yml').exists()
    assert shimmed.mutating_docker_calls() == []


def test_raw_fallback_verifies_every_file(shimmed: Shimmed, tmp_path: Path) -> None:
    assets = _release_copy(shimmed, tmp_path)
    (assets / f'openprocessor-deploy-{RELEASE}.tar.gz').unlink()
    raw_env = shimmed.release / 'raw' / RELEASE / 'env.template'
    ok = dry(shimmed)
    assert ok.returncode == 0, ok.stderr
    assert 'fetching verified raw files' in ok.stderr
    raw_env.write_text(raw_env.read_text() + 'OP_BIND_ADDRESS=0.0.0.0\n')
    assert not (shimmed.root / 'inst').exists()
    bad = dry(shimmed)
    assert bad.returncode == 7
    assert 'env.template failed verification' in bad.stderr


def test_images_lock_with_latest_is_rejected(shimmed: Shimmed, tmp_path: Path) -> None:
    shimmed.release = build_fake_release(
        tmp_path / 'r',
        lock_override='api=davidamacey/openprocessor:latest@sha256:' + 'a' * 64 + '\n',
    )
    result = dry(shimmed)
    assert result.returncode == 7
    assert 'images.lock' in result.stderr


def _lock_with(key: str, ref: str) -> str:
    lines = []
    for k, r in image_key_refs():
        lines.append(f'{k}={ref if k == key else r}@{fake_digest(k)}')
    return '\n'.join(lines) + '\n'


@pytest.mark.parametrize(
    ('key', 'ref', 'expected'),
    [
        ('opensearch', 'evil/opensearch:3.6.0', 'opensearchproject/opensearch'),
        ('api', 'davidamacey/openprocessor-triton', 'davidamacey/openprocessor'),
        ('grafana', 'grafana/loki:3.6.12', 'grafana/grafana'),
    ],
)
def test_lock_line_with_the_wrong_repo_for_its_key_is_refused(
    shimmed: Shimmed, tmp_path: Path, key: str, ref: str, expected: str
) -> None:
    # Review s4: a well-formed, checksum-covered lock line whose repo is not
    # the one scripts/lib/image_keys.sh names for that key.
    shimmed.release = build_fake_release(tmp_path / 'r', lock_override=_lock_with(key, ref))
    result = dry(shimmed)
    assert result.returncode == 7, result.stderr[-2000:]
    assert (
        f"images.lock '{key}' names {ref.split(':', maxsplit=1)[0]}, not {expected}"
        in result.stderr
    )
    assert 'integrity check' in result.stderr
    assert shimmed.mutating_docker_calls() == []
    assert [ln for ln in shimmed.log_lines('docker') if ' pull ' in f' {ln} '] == []


def test_lock_repo_check_accepts_a_tagged_build_image(shimmed: Shimmed, tmp_path: Path) -> None:
    shimmed.release = build_fake_release(
        tmp_path / 'r', lock_override=_lock_with('api', 'davidamacey/openprocessor:v9.9.9')
    )
    result = dry(shimmed)
    assert result.returncode == 0, result.stderr[-2000:]


def test_placeholder_lock_is_refused_not_installed(
    shimmed: Shimmed, tmp_path: Path, repo_root: Path
) -> None:
    # A development checkout's images.lock carries placeholder digests; the
    # committed lock is a release pin now, so drive the refusal with a copy.
    placeholder = re.sub(
        r'@sha256:[0-9a-f]{64}',
        '@sha256:' + '0' * 60 + 'dev0',
        (repo_root / 'images.lock').read_text(),
    )
    shimmed.release = build_fake_release(tmp_path / 'r', lock_override=placeholder)
    result = dry(shimmed)
    assert result.returncode == 7
    assert '--image-tag' in result.stderr
    assert 'development placeholder digest' in result.stderr
    assert 'this checkout is not a release' in result.stderr


def test_committed_locks_are_fully_pinned(repo_root: Path) -> None:
    """A placeholder or floating reference must never ship in a release."""
    images = [
        ln
        for ln in (repo_root / 'images.lock').read_text().splitlines()
        if ln.strip() and not ln.startswith('#')
    ]
    assert images
    for line in images:
        assert re.fullmatch(r'[a-z0-9_]+=\S+@sha256:[0-9a-f]{64}', line), line
        assert ':latest@' not in line, line


def _local_images(shimmed: Shimmed, repo: str, tag: str) -> None:
    images = shimmed.state / 'images'
    images.mkdir(exist_ok=True)
    for name in ('openprocessor', 'openprocessor-triton'):
        (images / f'{repo}_{name}_{tag}').write_text(f'{repo}/{name}@sha256:{"0" * 64}\n')


def test_image_tag_mode_uses_local_tags_without_lock_pins(shimmed: Shimmed) -> None:
    _local_images(shimmed, 'opinst', 'localtest')
    result = configure(shimmed, '--image-tag', 'localtest', OP_IMAGE_REPO='opinst')
    assert result.returncode == 0, result.stderr
    env = env_file(shimmed)
    assert env['OP_IMAGE_REPO'] == 'opinst'
    assert env['OP_IMAGE_TAG'] == 'localtest'
    assert env['OP_API_IMAGE'] == ''
    assert not [ln for ln in shimmed.mutating_docker_calls() if 'pull opinst/' in ln]
    assert 'UNPINNED install' in result.stderr


def test_image_tag_mode_never_pulls_an_unpinned_image_of_ours(shimmed: Shimmed) -> None:
    # r11: a missing local build must not be fetched by tag from a registry.
    result = configure(shimmed, '--image-tag', 'localtest', OP_IMAGE_REPO='opinst')
    assert result.returncode == 7
    assert 'local builds only' in result.stderr
    assert not [
        ln for ln in shimmed.mutating_docker_calls() if ln.startswith('docker pull opinst/')
    ]


def test_non_https_artifact_base_is_refused(shimmed: Shimmed) -> None:
    result = dry(shimmed, OP_ARTIFACT_BASE_URL='http://release.test/assets')
    assert result.returncode == 2
    assert shimmed.log_lines('curl') == []


def test_version_must_be_a_release_tag(shimmed: Shimmed) -> None:
    result = shimmed.run(
        [
            '--dry-run',
            '--unattended',
            '--dir',
            'inst',
            '--project',
            PROJECT,
            '--version',
            '../../other/repo/main',
            '--tiers',
            'core',
        ]
    )
    assert result.returncode == 2
    assert shimmed.log_lines('curl') == []


def test_branch_mode_uses_the_full_sha_and_a_sha_image_tag(
    shimmed: Shimmed, tmp_path: Path
) -> None:
    sha = 'ab' * 20
    _release_copy(shimmed, tmp_path)
    shutil.copytree(shimmed.release / 'raw' / RELEASE, shimmed.release / 'raw' / sha)
    shimmed.flag('allow_mutations')
    result = shimmed.run(
        [
            '--no-start',
            '--unattended',
            '--dir',
            'inst',
            '--project',
            PROJECT,
            '--branch',
            'main',
            '--tiers',
            'core',
        ],
        SHIM_BRANCH_SHA=sha,
    )
    assert result.returncode == 0, result.stderr
    assert any(
        f'release.test/raw/{sha}/release-manifest.txt' in ln for ln in shimmed.log_lines('curl')
    )
    assert env_file(shimmed)['OP_IMAGE_TAG'] == f'sha-{sha[:12]}'
    assert 'TESTING install' in result.stderr


def test_branch_mode_prints_the_short_sha_but_records_the_full_one(
    shimmed: Shimmed, tmp_path: Path
) -> None:
    sha = 'cd' * 20
    _release_copy(shimmed, tmp_path)
    shutil.copytree(shimmed.release / 'raw' / RELEASE, shimmed.release / 'raw' / sha)
    shimmed.flag('allow_mutations')
    result = shimmed.run(
        [
            *('--no-start', '--unattended', '--dir', 'inst', '--project', PROJECT),
            *('--branch', 'main', '--tiers', 'core'),
        ],
        SHIM_BRANCH_SHA=sha,
    )
    assert result.returncode == 0, result.stderr
    assert sha not in result.stdout + result.stderr
    assert sha[:12] in result.stdout
    state = json.loads((shimmed.root / 'inst' / '.install' / 'state.json').read_text())
    assert state['version'] == sha


def test_branch_mode_refuses_an_unparseable_api_answer(shimmed: Shimmed) -> None:
    result = shimmed.run(
        [
            '--dry-run',
            '--unattended',
            '--dir',
            'inst',
            '--project',
            PROJECT,
            '--branch',
            'main',
            '--tiers',
            'core',
        ],
        SHIM_BRANCH_SHA='',
    )
    assert result.returncode != 0
    assert 'head commit' in result.stderr
