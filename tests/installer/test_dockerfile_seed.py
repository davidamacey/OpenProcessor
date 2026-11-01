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
