"""Dockerfile model-repo seed contract (installer plan section 9.1)."""

from __future__ import annotations

import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _stage(name: str) -> str:
    text = (REPO_ROOT / 'Dockerfile').read_text()
    match = re.search(rf'^FROM \S+ AS {name}\n(.*?)(?=^FROM |\Z)', text, re.M | re.S)
    assert match, f'no stage {name}'
    return match.group(1)


def test_export_image_carries_export_scripts_and_the_model_seed() -> None:
    export = _stage('export')
    assert re.search(r'COPY\s+(--chown=\S+\s+)?export/\s+\./export/', export)
    assert re.search(
        r'COPY\s+(--chown=\S+\s+)?models/\s+/opt/openprocessor/model_repo_seed/', export
    )


def test_api_image_carries_examples_but_neither_export_scripts_nor_the_seed() -> None:
    api = _stage('api')
    assert re.search(r'COPY\s+(--chown=\S+\s+)?examples/\s+\./examples/', api)
    assert not re.search(r'COPY\s+(--chown=\S+\s+)?(export|models)/', api)


def test_every_tracked_config_pbtxt_would_land_in_the_seed() -> None:
    """Every models/*/config.pbtxt tracked in git is not excluded by
    .dockerignore (which would silently drop it from the model repo
    seed)."""
    dockerignore = (REPO_ROOT / '.dockerignore').read_text()
    config_pbtxt_files = sorted((REPO_ROOT / 'models').glob('*/config.pbtxt'))
    assert config_pbtxt_files, 'expected at least one models/*/config.pbtxt in the repo'

    # .dockerignore explicitly documents that config.pbtxt is included;
    # a regression would add a line like "*.pbtxt" or "models/**" without
    # a config.pbtxt re-include (!models/**/config.pbtxt).
    exclude_lines = [
        line.strip()
        for line in dockerignore.splitlines()
        if line.strip() and not line.strip().startswith('#') and not line.strip().startswith('!')
    ]
    for line in exclude_lines:
        assert 'config.pbtxt' not in line, f'.dockerignore excludes config.pbtxt: {line}'
        # A bare "models/**" or "models" exclude with no re-include would
        # also drop every config.pbtxt.
        if line in ('models/**', 'models', 'models/*'):
            reincludes = [
                ln for ln in dockerignore.splitlines() if ln.strip().startswith('!models')
            ]
            assert reincludes, f'.dockerignore excludes {line} with no !models re-include'


def test_dockerfile_copies_config_templates_for_preflight_profile() -> None:
    """preflight_profile runs generate_all_configs inside the API image with
    PROJECT_DIR=/app, which reads /app/config_templates/profiles/*.json."""
    text = _stage('export')
    assert re.search(r'COPY\s+(--chown=\S+\s+)?config_templates/\s+\./config_templates/', text)
    assert (REPO_ROOT / 'config_templates' / 'profiles' / 'minimal.json').is_file()
    for line in (REPO_ROOT / '.dockerignore').read_text().splitlines():
        assert not line.strip().rstrip('/').startswith('config_templates'), line


def test_dockerfile_triton_holds_tensorrt_before_upgrade() -> None:
    """A cold apt layer must not upgrade TensorRT past the pinned lockstep."""
    text = (REPO_ROOT / 'Dockerfile.triton').read_text()
    install = text.find('apt-get install -y --no-install-recommends --allow-downgrades $spec')
    hold = text.find('apt-mark hold')
    upgrade = text.find('apt-get upgrade')
    # the base image's stock TensorRT is older than the pin: install the pin, then hold it
    assert 0 <= install < hold < upgrade


def test_dockerfile_triton_drops_unused_dev_packages_before_the_upgrade_rewrites_them() -> None:
    """The base image's -dev packages (cuBLAS/NCCL/NVSHMEM headers and static libraries, the
    TensorRT Windows builder resource) are unused at inference. The upgrade layer rewrites every
    installed package, so removing them first, in the same RUN, keeps ~5 GB out of the image."""
    text = (REPO_ROOT / 'Dockerfile.triton').read_text()
    remove = text.find('apt-get remove')
    install = text.find('apt-get install -y --no-install-recommends --allow-downgrades $spec')
    upgrade = text.find('apt-get upgrade')
    assert 0 <= remove < install < upgrade
    removed = text[remove:install]
    for package in (
        'libnvinfer-win-builder-resource11',
        'tensorrt-dev',
        'libcublas-dev-13-3',
        'libnccl-dev',
        'libnvshmem3-dev-cuda-13',
    ):
        assert package in removed
    # trtexec (libnvinfer-bin) and the runtime libraries stay
    assert 'libnvinfer-bin' not in removed
    assert 'libcublas-13-3' not in removed


def test_dockerfile_triton_upgrades_the_python_packages_the_scan_flags() -> None:
    """The NGC image's python env lags its security fixes; the release gate fails on a fixable
    CRITICAL, so certifi (GHSA-6q48-6c7w-9gw2) is raised with starlette and anyio. filelock and
    virtualenv carry fixable HIGHs the CI image scan fails on; the exact pins are installed with
    --upgrade before the torch set, which keeps what it finds when the pin satisfies it."""
    dockerfile = (REPO_ROOT / 'Dockerfile.triton').read_text()
    assert '--upgrade -r /tmp/triton-requirements/requirements-security.txt' in dockerfile
    assert dockerfile.index('requirements-security.txt') < dockerfile.index(
        'requirements-torch.txt'
    )
    pins = (REPO_ROOT / 'docker' / 'triton' / 'requirements-security.txt').read_text()
    floors = {
        'starlette': (1, 3, 1),
        'anyio': (4, 14, 2),
        'certifi': (2026, 10, 18),
        'filelock': (4, 2, 1),
        'virtualenv': (21, 7, 13),
    }
    for package, floor in floors.items():
        match = re.search(rf'^{package}==([0-9.]+)\s*$', pins, re.M)
        assert match, package
        assert tuple(int(p) for p in match.group(1).split('.')) >= floor, package


def test_dockerfile_segmenter_replaces_the_base_python_packages_the_scan_flags() -> None:
    """The PyTorch base's system python ships pillow, certifi, filelock, dnspython and uv with
    fixable findings. The builder installs fixed copies into the user site; the runtime stage runs
    no pip and deletes the stale system copies so only the fixed ones remain."""
    text = (REPO_ROOT / 'docker' / 'segmenter' / 'Dockerfile').read_text()
    runtime_start = text.rindex('FROM pytorch/pytorch')
    builder, runtime = text[:runtime_start], text[runtime_start:]
    assert (
        'pip install --user --no-cache-dir --no-warn-script-location -r /tmp/requirements.txt'
        in builder
    )
    pins = (REPO_ROOT / 'docker' / 'segmenter' / 'requirements.txt').read_text()
    floors = {
        'fastapi': (0, 162, 0),
        'pillow': (13, 0, 0),
        'certifi': (2026, 10, 18),
        'filelock': (4, 2, 1),
        'dnspython': (2, 9, 0),
    }
    for package, floor in floors.items():
        match = re.search(rf'^{package}==([0-9.]+)\s*$', pins, re.M)
        assert match, package
        assert tuple(int(p) for p in match.group(1).split('.')) >= floor, package
    assert 'pip install' not in runtime
    for stale in (
        '"$site"/PIL ',
        '"$site"/certifi ',
        '"$site"/filelock ',
        '"$site"/dns ',
        '"$site"/uv ',
        '/usr/local/bin/uv ',
    ):
        assert stale in runtime
