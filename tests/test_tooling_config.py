"""The lint, type and hook configuration must point at files that exist.

A glob or path that matches nothing is how a guard goes quiet: the directory it
named was renamed or deleted and nothing noticed. These checks fail on the first
stale entry.
"""

from __future__ import annotations

import importlib.util
import re
import subprocess
import tomllib
from pathlib import Path

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]

# Cache and tool directories that never exist in a clean checkout.
EPHEMERAL = ('.git/', '.venv/', '__pycache__/', '.pytest_cache/', '.mypy_cache/', '.ruff_cache/')


def _tracked() -> list[str]:
    out = subprocess.run(
        ['git', 'ls-files', '-z'], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout
    return [p for p in out.split('\0') if p]


TRACKED = _tracked()
PYPROJECT = tomllib.loads((REPO_ROOT / 'pyproject.toml').read_text(encoding='utf-8'))
PRE_COMMIT = yaml.safe_load((REPO_ROOT / '.pre-commit-config.yaml').read_text(encoding='utf-8'))


def _glob_matches(pattern: str) -> bool:
    """Gitignore-style glob: ``**/`` spans directories, ``*`` stays inside one path segment."""
    regex = re.escape(pattern).replace(r'\*\*/', '(?:.*/)?').replace(r'\*', '[^/]*')
    compiled = re.compile(regex + r'\Z')
    return any(compiled.match(p) for p in TRACKED)


@pytest.mark.parametrize('pattern', PYPROJECT['tool']['ruff']['include'])
def test_ruff_include_globs_match_a_tracked_file(pattern: str) -> None:
    assert _glob_matches(pattern), f'ruff include {pattern!r} matches no tracked file'


@pytest.mark.parametrize('pattern', PYPROJECT['tool']['ruff']['lint']['per-file-ignores'])
def test_ruff_per_file_ignore_keys_match_a_tracked_file(pattern: str) -> None:
    assert _glob_matches(pattern), f'per-file-ignores key {pattern!r} matches no tracked file'


def _first_party_overrides() -> list[str]:
    modules: list[str] = []
    for override in PYPROJECT['tool']['mypy']['overrides']:
        names = override['module']
        modules += [names] if isinstance(names, str) else names
    return [m for m in modules if m.split('.')[0] in {'src', 'scripts', 'export', 'docker'}]


@pytest.mark.parametrize('module', _first_party_overrides())
def test_mypy_override_modules_resolve(module: str) -> None:
    base = module.removesuffix('.*')
    if module.endswith('.*'):
        assert (REPO_ROOT / base.replace('.', '/')).is_dir(), f'{module}: package not found'
    else:
        path = REPO_ROOT / base.replace('.', '/')
        assert path.with_suffix('.py').is_file() or path.is_dir(), f'{module}: module not found'


@pytest.mark.parametrize('module', _first_party_overrides())
def test_mypy_override_modules_are_importable_names(module: str) -> None:
    top = module.split('.', maxsplit=1)[0]
    assert importlib.util.find_spec(top) is not None or (REPO_ROOT / top).is_dir()


def _literal_prefixes(regex: str) -> list[str]:
    """Leading literal path of every top-level alternative of a (possibly verbose) regex."""
    text = re.sub(r'\(\?x\)', '', regex)
    text = re.sub(r'\s+', '', text).replace('\\.', '.')
    found: list[str] = []
    for match in re.finditer(r'(?:^|[(|])\^?([A-Za-z0-9_-][A-Za-z0-9_./-]*)', text):
        token = match.group(1)
        after = text[match.end() : match.end() + 1]
        if after in {'*', '+', '?', '['}:
            token = token.rsplit('/', 1)[0] + '/' if '/' in token else ''
        if '/' in token:
            found.append(token)
    return found


def _hook_patterns() -> list[tuple[str, str]]:
    patterns: list[tuple[str, str]] = []
    if PRE_COMMIT.get('exclude'):
        patterns.append(('global exclude', PRE_COMMIT['exclude']))
    patterns.extend(
        (f'{hook["id"]}.{key}', hook[key])
        for repo in PRE_COMMIT['repos']
        for hook in repo['hooks']
        for key in ('files', 'exclude')
        if hook.get(key)
    )
    return patterns


@pytest.mark.parametrize(('where', 'regex'), _hook_patterns())
def test_pre_commit_path_prefixes_exist(where: str, regex: str) -> None:
    for prefix in _literal_prefixes(regex):
        if prefix.startswith(EPHEMERAL):
            continue
        assert any(p.startswith(prefix) or p == prefix.rstrip('/') for p in TRACKED), (
            f'{where}: path prefix {prefix!r} matches no tracked file'
        )


def test_literal_prefix_extraction_reads_alternations() -> None:
    assert _literal_prefixes(r'^(src/ultralytics_patches/|tests/|frontend/)') == [
        'src/ultralytics_patches/',
        'tests/',
        'frontend/',
    ]
    assert _literal_prefixes('(?x)^(\n  a/b/|\n  c\\.d/\n)') == ['a/b/', 'c.d/']
    assert _literal_prefixes(r'^(.*\.md|security-reports/)') == ['security-reports/']
