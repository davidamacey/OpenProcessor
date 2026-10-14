"""Dockerfile and build-context rules: pinned bases, non-root runtime, slim stages, and a
`COPY scripts/...` set that covers what the container actually imports."""

from __future__ import annotations

import ast
import re
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]

# `COPY` instructions with a trailing backslash continuation are joined before parsing.
_INSTRUCTION_RE = re.compile(r'^(FROM|COPY|USER|RUN)\s+(.*)$', re.M)
_PINNED_RE = re.compile(r'@sha256:[0-9a-f]{64}\b')

# A final stage that sets no USER inherits one from its base: nginx-unprivileged runs as 101.
_USER_FROM_BASE = {'docs-site/Dockerfile': 'nginx-unprivileged sets USER 101'}

# Images built from the repository root; a path under an ignored tree cannot be COPY-ed there.
_ROOT_CONTEXT = (
    'Dockerfile',
    'Dockerfile.triton',
    'docker/evaluator/Dockerfile',
    'docker/trainer/Dockerfile',
)


def _dockerfiles() -> list[str]:
    out = subprocess.run(  # nosec B603 B607 - fixed git invocation
        ['git', 'ls-files', '*Dockerfile*'],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.split()
    return sorted(out)


def _instructions(rel: str) -> list[tuple[str, str]]:
    text = (REPO_ROOT / rel).read_text(encoding='utf-8').replace('\\\n', ' ')
    return [(m.group(1), m.group(2)) for m in _INSTRUCTION_RE.finditer(text)]


def _stages(rel: str) -> list[list[tuple[str, str]]]:
    stages: list[list[tuple[str, str]]] = []
    for kind, rest in _instructions(rel):
        if kind == 'FROM':
            stages.append([])
        stages[-1].append((kind, rest))
    return stages


@pytest.mark.parametrize('rel', _dockerfiles())
def test_every_base_image_is_pinned_by_digest(rel: str) -> None:
    stage_names = set()
    for kind, rest in _instructions(rel):
        if kind != 'FROM':
            continue
        words = [w for w in rest.split() if not w.startswith('--')]
        ref = words[0]
        if ref in stage_names or ref == 'scratch':
            pass
        else:
            # An ARG-built reference is pinned by a sha256 ARG default (hardened triton).
            expanded = ref
            for name, default in re.findall(
                r'^ARG (\w+)=(\S+)$', (REPO_ROOT / rel).read_text(), re.M
            ):
                expanded = expanded.replace('${' + name + '}', default)
            assert _PINNED_RE.search(expanded), f'{rel}: FROM {ref} is not pinned by digest'
        if len(words) == 3 and words[1].upper() == 'AS':
            stage_names.add(words[2])


@pytest.mark.parametrize('rel', [r for r in _dockerfiles() if r not in _USER_FROM_BASE])
def test_final_stage_runs_as_a_non_root_user(rel: str) -> None:
    users = [rest.split()[0] for kind, rest in _stages(rel)[-1] if kind == 'USER']
    assert users, f'{rel}: the final stage never sets USER'
    assert users[-1] not in {'root', '0'}, f'{rel}: the final stage ends as root'


@pytest.mark.parametrize('rel', ['docker/evaluator/Dockerfile', 'docker/segmenter/Dockerfile'])
def test_image_builds_dependencies_in_a_separate_stage(rel: str) -> None:
    stages = _stages(rel)
    assert len(stages) >= 3 if 'segmenter' in rel else len(stages) >= 2
    installs = [rest for kind, rest in stages[-1] if kind == 'RUN' and 'pip install' in rest]
    assert installs == [], f'{rel}: the runtime stage runs pip install'


def test_root_dockerignore_excludes_trees_no_build_needs() -> None:
    ignored = {
        ln.strip().rstrip('/')
        for ln in (REPO_ROOT / '.dockerignore').read_text().splitlines()
        if ln.strip() and not ln.startswith(('#', '!'))
    }
    for entry in ('artifacts_local', 'docs-site', 'frontend', '.claude', '.install', 'export_*'):
        assert entry in ignored, f'.dockerignore does not exclude {entry}/'
    for rel in _ROOT_CONTEXT:
        for kind, rest in _instructions(rel):
            if kind != 'COPY' or '--from=' in rest:
                continue
            for src in [w for w in rest.split() if not w.startswith('--')][:-1]:
                assert src.split('/')[0] not in ignored, f'{rel}: COPY {src} is dockerignored'


def _copied_scripts(rel: str) -> tuple[set[str], set[str]]:
    """(directories, files) under scripts/ that the Dockerfile copies into the image."""
    dirs: set[str] = set()
    files: set[str] = set()
    for kind, rest in _instructions(rel):
        if kind != 'COPY' or '--from=' in rest:
            continue
        words = [w for w in rest.split() if not w.startswith('--')]
        for src in words[:-1]:
            if src == 'scripts/':
                dirs.add('scripts')
            elif src.startswith('scripts/') and src.endswith('/'):
                dirs.add(src.rstrip('/'))
            elif src.startswith('scripts/'):
                files.add(src)
    return dirs, files


def _covered(path: str, dirs: set[str], files: set[str]) -> bool:
    return path in files or any(path == d or path.startswith(d + '/') for d in dirs)


def _scripts_modules_imported(roots: list[str]) -> set[str]:
    """Repo-relative files under scripts/ reachable by import from the given trees."""
    seen: set[Path] = set()
    todo = [p for r in roots for p in (REPO_ROOT / r).rglob('*.py')] if roots else []
    needed: set[str] = set()
    while todo:
        path = todo.pop()
        if path in seen:
            continue
        seen.add(path)
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
            names: list[str] = []
            if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                names = [node.module, *[f'{node.module}.{a.name}' for a in node.names]]
            elif isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            for name in names:
                if name.split('.')[0] != 'scripts':
                    continue
                base = REPO_ROOT / name.replace('.', '/')
                for cand in (base.with_suffix('.py'), base / '__init__.py'):
                    if cand.is_file():
                        needed.add(str(cand.relative_to(REPO_ROOT)))
                        todo.append(cand)
    return needed


def test_api_image_copies_every_scripts_file_it_imports_or_sources() -> None:
    dirs, files = _copied_scripts('Dockerfile')
    assert 'scripts' not in dirs, 'the api image must copy a runtime subset of scripts/'
    roots = ['src', 'export', 'models', 'config_templates', 'scripts/curation']
    needed = _scripts_modules_imported(roots)
    # `openprocessor models install` and `sample coco` run these inside the container
    # (scripts/lib/model_setup.sh).
    needed |= _scripts_modules_imported(['scripts/datasets'])
    needed |= {
        'scripts/datasets/fetch_coco_subset.py',
        'scripts/datasets/manifests/coco_va_200.json',
        'scripts/datasets/manifests/coco_va_800.json',
        'scripts/lib/config.sh',
        'scripts/lib/download.sh',
    }
    for lib in sorted(needed):
        if lib.endswith('.sh'):
            text = (REPO_ROOT / lib).read_text()
            needed |= {
                f'scripts/lib/{m}' for m in re.findall(r'source "\$\{\w+\}/(\w+\.sh)"', text)
            }
    missing = sorted(n for n in needed if not _covered(n, dirs, files))
    assert missing == [], f'Dockerfile does not COPY: {missing}'


def test_evaluator_image_copies_every_scripts_file_it_imports() -> None:
    dirs, files = _copied_scripts('docker/evaluator/Dockerfile')
    assert 'scripts' not in dirs
    needed = _scripts_modules_imported(['src', 'scripts/curation/bakeoff'])
    missing = sorted(n for n in needed if not _covered(n, dirs, files))
    assert missing == [], f'evaluator Dockerfile does not COPY: {missing}'
