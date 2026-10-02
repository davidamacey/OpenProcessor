"""One release version: VERSION, the package metadata, the docs and the
default image tag every compose service falls back to must all agree."""

from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
VERSION = (ROOT / 'VERSION').read_text().strip()
DOC_ROOTS = (
    ROOT / 'README.md',
    ROOT / 'INSTALLATION.md',
    ROOT / 'CLAUDE.md',
    *ROOT.glob('docs/*.md'),
    *(ROOT / 'docs-site' / 'docs').rglob('*.mdx'),
)
RELEASE_MENTION = re.compile(r'(?<![\w.])v(0\.\d+\.\d+)(?![\w.])')


def test_version_file_is_a_plain_release_number() -> None:
    assert re.fullmatch(r'\d+\.\d+\.\d+', VERSION)


def test_package_metadata_matches_version() -> None:
    pyproject = tomllib.loads((ROOT / 'pyproject.toml').read_text())
    assert pyproject['project']['version'] == VERSION
    assert json.loads((ROOT / 'docs-site/package.json').read_text())['version'] == VERSION
    lock = json.loads((ROOT / 'docs-site/package-lock.json').read_text())
    assert lock['version'] == lock['packages']['']['version'] == VERSION


def test_every_compose_image_tag_default_is_version() -> None:
    compose = yaml.safe_load((ROOT / 'docker-compose.yml').read_text())
    tags = {
        match
        for spec in compose['services'].values()
        for match in re.findall(r'OP_IMAGE_TAG:-([^}]+)\}', str(spec.get('image', '')))
    }
    assert tags == {VERSION}


def test_the_installer_shim_assumes_the_same_default_tag() -> None:
    shim = (ROOT / 'tests/installer/shims/docker').read_text()
    assert f'tag:-{VERSION}}}' in shim


def test_docs_name_only_the_current_release() -> None:
    wrong = {
        f'{path.relative_to(ROOT)}: v{found}'
        for path in DOC_ROOTS
        for found in RELEASE_MENTION.findall(path.read_text())
        if found != VERSION
    }
    assert wrong == set()


def test_the_roadmap_ships_the_current_release() -> None:
    roadmap = json.loads((ROOT / 'docs-site/src/data/roadmap.json').read_text())
    shipped = [r['version'] for r in roadmap['releases'] if r['stage'] == 'shipped']
    assert shipped == [f'v{VERSION}']
