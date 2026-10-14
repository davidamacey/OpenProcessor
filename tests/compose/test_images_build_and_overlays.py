"""Pin build identity, the deploy-safe base file, image pinning and the compose overlays."""

from __future__ import annotations

import json
import shutil
import subprocess

import pytest
import yaml

from compose._helpers import COMPOSE_PATH, REPO_ROOT, _load_yaml, _services


# =============================================================================
# Build identity -- OP_BUILD_SHA baked in at build time so
# code_versions.api_sha/trainer_sha reflect the actual built commit rather
# than a dev checkout's live `git rev-parse HEAD` fallback.
# =============================================================================

_BUILD_SHA_SERVICES = ('api', 'curation-trainer', 'curation-evaluator')

_BUILD_SHA_DOCKERFILES = (
    REPO_ROOT / 'Dockerfile',
    REPO_ROOT / 'docker' / 'trainer' / 'Dockerfile',
    REPO_ROOT / 'docker' / 'evaluator' / 'Dockerfile',
)


def test_build_sha_passed_as_a_build_arg_on_core_services() -> None:
    """Installer plan §1: every `build:` block (and its `args:`) moved from
    docker-compose.yml to docker-compose.dev.yml -- the deploy-safe base
    file carries no `build:` at all, so this reads the overlay instead."""
    services = _services()
    dev_services = _load_yaml(REPO_ROOT / 'docker-compose.dev.yml')['services']
    missing: list[str] = []
    for name in _BUILD_SHA_SERVICES:
        assert name in services, f'expected service {name!r} in docker-compose.yml'
        assert name in dev_services, f'expected service {name!r} in docker-compose.dev.yml'
        args = (dev_services[name].get('build') or {}).get('args') or {}
        if 'OP_BUILD_SHA' not in args:
            missing.append(name)
    assert not missing, f'services missing build.args.OP_BUILD_SHA: {missing}'


def test_build_sha_baked_into_the_final_stage_of_every_dockerfile() -> None:
    for path in _BUILD_SHA_DOCKERFILES:
        lines = path.read_text(encoding='utf-8').splitlines()
        from_indices = [i for i, line in enumerate(lines) if line.startswith('FROM ')]
        assert from_indices, f'{path}: no FROM instruction found'
        tail = '\n'.join(lines[from_indices[-1] :])
        assert 'ARG OP_BUILD_SHA' in tail, f'{path}: missing ARG OP_BUILD_SHA after final FROM'
        assert 'ENV OP_BUILD_SHA=${OP_BUILD_SHA}' in tail, f'{path}: missing ENV OP_BUILD_SHA'
        assert 'org.opencontainers.image.revision=${OP_BUILD_SHA}' in tail, (
            f'{path}: missing revision LABEL binding OP_BUILD_SHA'
        )


# =============================================================================
# Installer plan §1/§3.2/§7 -- deploy-safe base compose contract.
# =============================================================================

# Source-code mounts that must never appear in the deploy-safe base file
# (docker-compose.yml). Every one of these lives only in
# docker-compose.dev.yml, which restores today's checkout hot-reload
# workflow (see the file's own header comment).
_FORBIDDEN_BASE_MOUNT_PREFIXES = (
    './src',
    './scripts',
    './export',
    './tests',
    './benchmarks',
    './test_images',
    './VERSION',
    './examples',
)

# Every custom (davidamacey/openprocessor*) image, and the per-service
# override var docker-compose.yml pins it through.
_CUSTOM_IMAGE_OVERRIDE_VARS = {
    'triton-server': 'OP_TRITON_IMAGE',
    'api': 'OP_API_IMAGE',
    'curation-detection-worker': 'OP_API_IMAGE',
    'curation-vlm-worker': 'OP_API_IMAGE',
    'curation-auto-label-worker': 'OP_API_IMAGE',
    'curation-cluster-refresh': 'OP_API_IMAGE',
    'curation-evaluator': 'OP_EVALUATOR_IMAGE',
    'segmenter': 'OP_SEGMENTER_IMAGE',
    'curation-trainer': 'OP_TRAINER_IMAGE',
}


def _version() -> str:
    return (REPO_ROOT / 'VERSION').read_text().strip()


def test_base_compose_has_no_build_blocks() -> None:
    """docker-compose.yml must be pull-only -- every `build:` block lives in
    docker-compose.dev.yml instead (installer plan §1)."""
    services = _services()
    with_build = [name for name, spec in services.items() if 'build' in spec]
    assert not with_build, f'docker-compose.yml still has build: blocks: {with_build}'


def test_base_compose_has_no_source_mounts() -> None:
    """docker-compose.yml must not bind-mount source code -- an installer
    dropping only this file into an empty dir would otherwise get Docker
    silently creating empty host dirs that shadow the image's /app/src,
    /app/export, etc. (installer plan §1)."""
    services = _services()
    bad: list[str] = []
    for name, spec in services.items():
        for v in spec.get('volumes') or []:
            source = v.get('source', '') if isinstance(v, dict) else str(v).split(':', 1)[0]
            if str(source).startswith(_FORBIDDEN_BASE_MOUNT_PREFIXES):
                bad.append(f'{name}: {v!r}')
    assert not bad, 'docker-compose.yml has forbidden source-code mounts:\n' + '\n'.join(bad)


def test_dev_overlay_carries_every_build_block_and_source_mount() -> None:
    """docker-compose.dev.yml must restore exactly what the base file gave
    up: a `build:` for every service that used to have one, and every
    forbidden source mount, so the checkout dev workflow is unchanged."""
    dev_services = _load_yaml(REPO_ROOT / 'docker-compose.dev.yml')['services']
    for name in _CUSTOM_IMAGE_OVERRIDE_VARS:
        assert name in dev_services, f'{name} missing from docker-compose.dev.yml'
        assert 'build' in dev_services[name], f'{name} missing build: in docker-compose.dev.yml'
    api_mounts = [str(v) for v in dev_services['api'].get('volumes') or []]
    for prefix in _FORBIDDEN_BASE_MOUNT_PREFIXES:
        assert any(m.startswith(prefix + ':') for m in api_mounts), (
            f'docker-compose.dev.yml/api missing a mount for {prefix}'
        )


def test_every_custom_image_has_a_per_service_override_and_no_latest_fallback() -> None:
    """installer plan §3.2: every custom image is overridable through its
    own OP_*_IMAGE var, and its fallback tag equals VERSION -- never
    `latest` (the reference implementation's staleness bug class)."""
    services = _services()
    version = _version()
    for name, override_var in _CUSTOM_IMAGE_OVERRIDE_VARS.items():
        image = str(services[name]['image'])
        assert image.startswith(f'${{{override_var}:-'), (
            f'{name} image not driven by ${{{override_var}:-...}}: {image!r}'
        )
        assert 'latest' not in image, f'{name} image fallback still uses latest: {image!r}'
        assert f':-{version}}}' in image or f':-{version}}}}}' in image, (
            f'{name} image fallback tag does not match VERSION ({version}): {image!r}'
        )


def test_compose_default_tag_matches_version() -> None:
    """Regression guard: VERSION, the compose fallback tag and the release
    process must move together (installer plan §3.4 release checklist)."""
    version = _version()
    for name in _CUSTOM_IMAGE_OVERRIDE_VARS:
        image = str(_services()[name]['image'])
        assert f'OP_IMAGE_TAG:-{version}' in image, (
            f'{name} image fallback tag does not match VERSION ({version}): {image!r}'
        )


def test_no_latest_in_any_custom_image_fallback() -> None:
    services = _services()
    for name in _CUSTOM_IMAGE_OVERRIDE_VARS:
        assert 'latest' not in str(services[name]['image'])


def test_gpu_arbiter_overlay_exists_and_mounts_docker_socket() -> None:
    """G-15: the docker socket must be opt-in (a separate overlay file),
    never a default mount on api."""
    overlay_path = REPO_ROOT / 'docker-compose.gpu-arbiter.yml'
    assert overlay_path.is_file(), 'expected docker-compose.gpu-arbiter.yml overlay'
    with overlay_path.open() as fh:
        overlay = yaml.safe_load(fh)
    mounts = [str(v) for v in (overlay['services']['api'].get('volumes') or [])]
    assert any('/var/run/docker.sock' in m for m in mounts), mounts
    # And the base compose must NOT already mount it (defeats the point of
    # an opt-in overlay).
    base_mounts = [str(v) for v in (_services()['api'].get('volumes') or [])]
    assert not any('docker.sock' in m for m in base_mounts), base_mounts


GPU_CLUSTERING_OVERLAY = REPO_ROOT / 'docker-compose.gpu-clustering.yml'


def test_gpu_clustering_overlay_touches_only_the_auto_label_worker() -> None:
    overlay = _load_yaml(GPU_CLUSTERING_OVERLAY)['services']
    assert set(overlay) == {'curation-auto-label-worker'}
    assert 'curation-auto-label-worker' in _load_yaml(COMPOSE_PATH)['services']
    worker = overlay['curation-auto-label-worker']
    (device,) = worker['deploy']['resources']['reservations']['devices']
    assert device['device_ids'] == ['${OP_CLUSTER_GPU_DEVICE:-0}']
    assert 'NVIDIA_VISIBLE_DEVICES=${OP_CLUSTER_GPU_DEVICE:-0}' in worker['environment']


@pytest.mark.skipif(shutil.which('docker') is None, reason='no docker CLI')
def test_gpu_clustering_overlay_renders_one_pinned_device() -> None:
    result = subprocess.run(
        [
            'docker',
            'compose',
            '-f',
            str(COMPOSE_PATH),
            '-f',
            str(GPU_CLUSTERING_OVERLAY),
            '--profile',
            'curation',
            'config',
            '--format',
            'json',
        ],
        check=False,
        capture_output=True,
        text=True,
        env={
            'PATH': '/usr/bin:/bin:/usr/local/bin',
            'HOME': str(REPO_ROOT),
            'OP_CLUSTER_GPU_DEVICE': '2',
        },
    )
    if result.returncode != 0:
        pytest.skip(f'docker compose could not render the files here: {result.stderr[:200]}')
    worker = json.loads(result.stdout)['services']['curation-auto-label-worker']
    (device,) = worker['deploy']['resources']['reservations']['devices']
    assert device['device_ids'] == ['2']
    assert worker['environment']['NVIDIA_VISIBLE_DEVICES'] == '2'
    assert worker['runtime'] == 'nvidia'


@pytest.mark.skipif(shutil.which('make') is None, reason='no make')
def test_cluster_make_targets_recreate_only_the_worker() -> None:
    def dry_run(target: str) -> str:
        return subprocess.run(
            ['make', '-n', target],
            check=True,
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
        ).stdout

    gpu, cpu = dry_run('cluster-gpu'), dry_run('cluster-cpu')
    assert '-f docker-compose.gpu-clustering.yml' in gpu
    assert 'docker-compose.gpu-clustering.yml' not in cpu
    for out in (gpu, cpu):
        assert '--no-deps --force-recreate curation-auto-label-worker' in out
