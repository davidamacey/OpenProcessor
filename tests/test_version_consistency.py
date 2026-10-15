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
    ROOT / 'SECURITY.md',
    ROOT / 'CONTRIBUTING.md',
    *ROOT.glob('docs/*.md'),
    *(ROOT / 'docs-site' / 'docs').rglob('*.md'),
    *(ROOT / 'docs-site' / 'docs').rglob('*.mdx'),
    # The frontend's own docs. CHANGELOG.md and docs/releases/ are history: they name old releases.
    *(p for p in (ROOT / 'frontend').glob('*.md') if p.name != 'CHANGELOG.md'),
    *(
        p
        for p in (ROOT / 'frontend' / 'docs').rglob('*.md')
        if 'releases' not in p.relative_to(ROOT / 'frontend' / 'docs').parts
    ),
)
# A release mention ends at a word character or a version dot, not at a sentence period
# ("shipped in v0.4.0." is a mention; "v0.4.0.1" and "v0.4.0rc1" are not this pattern).
RELEASE_MENTION = re.compile(r'(?<![\w.])v(0\.\d+\.\d+)(?!\w|\.\d)')


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


def test_the_installer_script_reports_the_current_release() -> None:
    script = (ROOT / 'setup-openprocessor.sh').read_text()
    assert re.search(rf'^SCRIPT_VERSION="{re.escape(VERSION)}"$', script, re.M)


def test_release_mention_pattern_catches_a_mention_at_a_sentence_end() -> None:
    assert RELEASE_MENTION.findall('Shipped in v0.9.9.') == ['0.9.9']
    assert RELEASE_MENTION.findall('v0.9.9, v0.9.8; (v0.9.7)') == ['0.9.9', '0.9.8', '0.9.7']


def test_release_mention_pattern_ignores_longer_versions_and_words() -> None:
    assert RELEASE_MENTION.findall('v0.9.9.1 v0.9.9rc1 dev0.9.9') == []


def test_the_scan_covers_the_frontend_docs_but_not_their_release_history() -> None:
    names = {p.relative_to(ROOT).as_posix() for p in DOC_ROOTS}
    assert 'frontend/README.md' in names
    assert 'frontend/docs/FEATURES.md' in names
    assert 'frontend/CHANGELOG.md' not in names
    assert not [n for n in names if n.startswith('frontend/docs/releases/')]
