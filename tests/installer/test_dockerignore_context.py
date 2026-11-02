"""Local model artifacts must never reach an image build context.

The 0.6.0 API image shipped untracked ONNX files, a promoted project model and
bytecode from the developer's models/ directory, because the root .dockerignore
only matched at the context root. Two layers hold independently: the release
script builds from the git tree (tests/test_release_build_and_publish.py), and
every .dockerignore here excludes the artifact patterns at any depth.
"""

from __future__ import annotations

import posixpath
import re
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]

# (Dockerfile, context directory) for every published image.
BUILDS = [
    ('Dockerfile', '.'),
    ('Dockerfile.triton', '.'),
    ('docker/trainer/Dockerfile', '.'),
    ('docker/segmenter/Dockerfile', 'docker/segmenter'),
    ('frontend/Dockerfile', 'frontend'),
]

ARTIFACTS = [
    'x__y/1/model.onnx',
    'x__y/1/model.plan',
    'x__y/1/model.plan.old',
    'x__y/1/model.engine',
    'x__y/1/model.profile',
    'x__y/1/trt_cache/blob.bin',
    'x__y/1/__pycache__/m.cpython-312.pyc',
    'x__y/1/m.pyc',
    'x__y/config.pbtxt.generated',
    'x__y/promote.json',
    'x__y/.lock',
    'x__y/model.lock',
    'pe_text_encoder/1/model.onnx',
    'a/1/__pycache__/m.pyc',
]


def _to_regex(pattern: str) -> re.Pattern[str]:
    out, i = '', 0
    while i < len(pattern):
        if pattern.startswith('**/', i):
            out += '(?:.*/)?'
            i += 3
        elif pattern.startswith('**', i):
            out += '.*'
            i += 2
        elif pattern[i] == '*':
            out += '[^/]*'
            i += 1
        elif pattern[i] == '[' and (j := pattern.find(']', i)) > i:
            out += pattern[i : j + 1]
            i = j + 1
        elif pattern[i] == '?':
            out += '[^/]'
            i += 1
        else:
            out += re.escape(pattern[i])
            i += 1
    return re.compile(f'^{out}$')


def is_ignored(rules: list[str], path: str) -> bool:
    """Docker .dockerignore semantics: last matching rule wins, a match on a
    parent directory excludes everything below it, '!' re-includes."""
    parts = path.split('/')
    ignored = False
    for rule in rules:
        negate = rule.startswith('!')
        pat = posixpath.normpath(rule.lstrip('!').strip().lstrip('/'))
        rx = _to_regex(pat)
        if any(rx.match('/'.join(parts[: n + 1])) for n in range(len(parts))):
            ignored = not negate
    return ignored


def load_rules(directory: str) -> list[str]:
    text = (REPO_ROOT / directory / '.dockerignore').read_text()
    return [ln.strip() for ln in text.splitlines() if ln.strip() and not ln.startswith('#')]


def test_matcher_follows_docker_semantics() -> None:
    assert is_ignored(['*.onnx'], 'model.onnx')
    assert not is_ignored(['*.onnx'], 'a/model.onnx')  # no ** : root only
    assert is_ignored(['**/*.onnx'], 'a/b/model.onnx')
    assert is_ignored(['**/*.onnx'], 'model.onnx')
    assert is_ignored(['**/*.py[cod]'], 'a/b.pyc')
    assert is_ignored(['models/*__*/'], 'models/x__y/1/m.json')
    assert not is_ignored(['**/*.onnx', '!a/keep.onnx'], 'a/keep.onnx')


@pytest.mark.parametrize('artifact', ARTIFACTS)
def test_root_dockerignore_excludes_local_model_artifacts(artifact: str) -> None:
    assert is_ignored(load_rules('.'), f'models/{artifact}'), artifact


def test_root_dockerignore_excludes_nested_pycache_anywhere() -> None:
    rules = load_rules('.')
    assert is_ignored(rules, 'src/pkg/__pycache__/m.pyc')
    assert is_ignored(rules, 'export/sub/__pycache__/m.cpython-312.pyc')


def test_root_dockerignore_keeps_every_tracked_model_file() -> None:
    tracked = subprocess.run(
        ['git', 'ls-files', 'models'], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout.split()
    assert any(t.endswith('config.pbtxt') for t in tracked)
    rules = load_rules('.')
    dropped = [t for t in tracked if is_ignored(rules, t)]
    assert not dropped, f'.dockerignore would drop tracked model files: {dropped}'


def _directory_copies(dockerfile: str) -> list[str]:
    sources: list[str] = []
    for line in (REPO_ROOT / dockerfile).read_text().splitlines():
        m = re.match(r'\s*COPY\s+(.*)', line)
        if not m or '--from=' in m.group(1):
            continue
        args = [a for a in m.group(1).split() if not a.startswith('--')]
        sources += [a for a in args[:-1] if a.endswith('/') or a == '.']
    return sources


@pytest.mark.parametrize(('dockerfile', 'context'), BUILDS)
def test_every_directory_copy_context_excludes_artifacts(dockerfile: str, context: str) -> None:
    """Each context that copies a directory ignores model artifacts and bytecode at depth."""
    for src in _directory_copies(dockerfile):
        base = '' if src == '.' else src
        rules = load_rules(context)
        for artifact in ('m/1/model.onnx', 'm/1/model.plan', 'a/__pycache__/x.pyc', 'a/x.pyc'):
            assert is_ignored(rules, f'{base}{artifact}'), (dockerfile, src, artifact)
