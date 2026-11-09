"""Dockerfile and build-context rules: pinned bases, non-root runtime, slim stages, and a
`COPY scripts/...` set that covers what the container actually imports."""

from __future__ import annotations

import ast
import re
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]

# `COPY` instructions with a trailing backslash continuation are joined before parsing.
_INSTRUCTION_RE = re.compile(r'^(FROM|COPY|USER|RUN|ARG)\s+(.*)$', re.M)
_PINNED_RE = re.compile(r'@sha256:[0-9a-f]{64}\b')

# A final stage that sets no USER inherits one from its base: nginx-unprivileged runs as 101.
_USER_FROM_BASE = {'docs-site/Dockerfile': 'nginx-unprivileged sets USER 101'}

# Images built from the repository root; a path under an ignored tree cannot be COPY-ed there.
_ROOT_CONTEXT = (
    'Dockerfile',
    'Dockerfile.triton',
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
        if stages:  # a global ARG may precede the first FROM
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


@pytest.mark.parametrize('rel', ['docker/segmenter/Dockerfile'])
def test_image_builds_dependencies_in_a_separate_stage(rel: str) -> None:
    stages = _stages(rel)
    assert len(stages) >= 3
    installs = [rest for kind, rest in stages[-1] if kind == 'RUN' and 'pip install' in rest]
    assert installs == [], f'{rel}: the runtime stage runs pip install'


# --- the root Dockerfile: one build graph, three published targets --------------------------
_STAGE_ORDER = [
    'base',
    'builder',
    'builder-export',
    'builder-eval',
    'runtime-base',
    'export',
    'evaluator',
    'api',
]
_TARGETS = ['api', 'export', 'evaluator']
_RUNTIME_STAGES = ['base', 'runtime-base', 'export', 'evaluator', 'api']


def _named_stages(rel: str = 'Dockerfile') -> dict[str, list[tuple[str, str]]]:
    named: dict[str, list[tuple[str, str]]] = {}
    current: list[tuple[str, str]] = []
    for kind, rest in _instructions(rel):
        if kind == 'FROM':
            words = [w for w in rest.split() if not w.startswith('--')]
            assert len(words) == 3, f'{rel}: unnamed stage: {rest}'
            assert words[1].upper() == 'AS', f'{rel}: unnamed stage: {rest}'
            current = named.setdefault(words[2], [])
        current.append((kind, rest))
    return named


def test_root_dockerfile_stage_graph() -> None:
    assert list(_named_stages()) == _STAGE_ORDER, 'api must stay the last stage (untargeted builds)'


@pytest.mark.parametrize('target', _TARGETS)
def test_every_published_target_ends_as_a_non_root_user(target: str) -> None:
    users = [rest.split()[0] for kind, rest in _named_stages()[target] if kind == 'USER']
    assert users, f'target {target} never sets USER'
    assert users[-1] not in {'root', '0'}, f'target {target} does not end non-root'


@pytest.mark.parametrize('stage', _RUNTIME_STAGES)
def test_no_runtime_stage_installs_packages_with_pip(stage: str) -> None:
    installs = [r for kind, r in _named_stages()[stage] if kind == 'RUN' and 'pip install' in r]
    assert installs == [], f'stage {stage} runs pip install; installs belong in builder stages'


@pytest.mark.parametrize('target', _TARGETS)
def test_build_sha_is_set_after_the_big_layers(target: str) -> None:
    """OP_BUILD_SHA changes every commit; declared early it would invalidate the cache of every
    layer below it (the multi-GB dependency copies) on each build."""
    stage = _named_stages()[target]
    sha_at = next(
        i for i, (k, r) in enumerate(stage) if k == 'ARG' and r.startswith('OP_BUILD_SHA')
    )
    last_copy = max(i for i, (k, _) in enumerate(stage) if k == 'COPY')
    assert sha_at > last_copy, f'target {target}: ARG OP_BUILD_SHA precedes a COPY'
    runtime_base = _named_stages()['runtime-base'] + _named_stages()['base']
    assert not [1 for k, r in runtime_base if k == 'ARG' and r.startswith('OP_BUILD_SHA')]


def test_api_target_carries_no_export_toolchain_models_or_venv() -> None:
    for kind, rest in _named_stages()['api']:
        if kind != 'COPY':
            continue
        assert '--from=builder-export' not in rest, rest
        assert '--from=builder-eval' not in rest, rest
        sources = [w for w in rest.split() if not w.startswith('--')][:-1]
        assert not [s for s in sources if s.startswith(('export/', 'models/'))], rest
    assert not [1 for k, r in _named_stages()['api'] if '/opt/venv' in r]


def test_export_target_carries_the_toolchain_and_the_seed() -> None:
    stage = ' '.join(r for _, r in _named_stages()['export'])
    assert '--from=builder-export' in stage
    assert 'export/' in stage
    assert 'models/' in stage
    assert '/opt/venv-y11' in stage


def test_evaluator_target_takes_only_the_evaluator_delta() -> None:
    stage = ' '.join(r for _, r in _named_stages()['evaluator'])
    assert '--from=builder-eval' in stage
    assert '--from=builder-export' not in stage


def test_extra_dependency_stages_install_under_the_runtime_freeze() -> None:
    for name in ('builder-export', 'builder-eval'):
        installs = [r for k, r in _named_stages()[name] if k == 'RUN' and 'pip install --user' in r]
        assert installs, name
        assert all('-c /tmp/runtime.freeze' in r for r in installs), name


def test_evaluator_dockerfile_is_gone() -> None:
    assert not (REPO_ROOT / 'docker' / 'evaluator').exists()


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


def _copied_scripts(rel: str, target: str | None = None) -> tuple[set[str], set[str]]:
    """(directories, files) under scripts/ that the Dockerfile (or one stage of it) copies."""
    dirs: set[str] = set()
    files: set[str] = set()
    instructions = _named_stages(rel)[target] if target else _instructions(rel)
    for kind, rest in instructions:
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


_SHELL_LIBS = {'scripts/lib/config.sh', 'scripts/lib/download.sh'}
_COCO_SAMPLE = {
    'scripts/datasets/fetch_coco_subset.py',
    'scripts/datasets/manifests/coco_va_200.json',
    'scripts/datasets/manifests/coco_va_800.json',
}


def _with_sourced_libs(needed: set[str]) -> set[str]:
    for lib in sorted(needed):
        if lib.endswith('.sh'):
            text = (REPO_ROOT / lib).read_text()
            needed |= {
                f'scripts/lib/{m}' for m in re.findall(r'source "\$\{\w+\}/(\w+\.sh)"', text)
            }
    return needed


def test_api_image_copies_every_scripts_file_it_imports_or_sources() -> None:
    dirs, files = _copied_scripts('Dockerfile', 'api')
    assert 'scripts' not in dirs, 'the api image must copy a runtime subset of scripts/'
    needed = _scripts_modules_imported(['src', 'config_templates', 'scripts/curation'])
    needed |= _scripts_modules_imported(['scripts/datasets'])
    needed |= _COCO_SAMPLE | _SHELL_LIBS
    missing = sorted(n for n in _with_sourced_libs(needed) if not _covered(n, dirs, files))
    assert missing == [], f'api target does not COPY: {missing}'


def _flat_strings(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [s for item in value for s in _flat_strings(item)]


def _api_services_runtime_paths() -> dict[str, set[str]]:
    """``path under /app -> services`` for what each api-image service runs in compose.

    docker-compose.yml bind-mounts no source, so a path its command, entrypoint or healthcheck
    names under /app, or the file of a ``python -m`` module, must be baked into the image. A
    module that reads ``alembic.ini`` relative to the repository root needs that file too.
    """
    compose = yaml.safe_load((REPO_ROOT / 'docker-compose.yml').read_text())['services']
    found: dict[str, set[str]] = {}
    for name, svc in compose.items():
        if not str(svc.get('image', '')).startswith('${OP_API_IMAGE'):
            continue
        text = ' '.join(
            _flat_strings(svc.get('command'))
            + _flat_strings(svc.get('entrypoint'))
            + _flat_strings((svc.get('healthcheck') or {}).get('test'))
        )
        paths = {p.rstrip('/.') for p in re.findall(r'/app/([\w./-]+)', text)}
        for module in re.findall(r'python3?\s+-m\s+([\w.]+)', text):
            base = Path(*module.split('.'))
            for cand in (base.with_suffix('.py'), base / '__init__.py'):
                if (REPO_ROOT / cand).is_file():
                    paths.add(cand.as_posix())
                    if 'alembic.ini' in (REPO_ROOT / cand).read_text():
                        paths.add('alembic.ini')
        for path in paths - {'data'}:
            found.setdefault(path, set()).add(name)
    return found


def test_api_image_copies_every_path_its_compose_services_run() -> None:
    needed = _api_services_runtime_paths()
    assert 'scripts/flower/flowerconfig.py' in needed, 'the scan found nothing: not vacuous'
    copied = [
        w
        for kind, rest in _named_stages()['api']
        if kind == 'COPY' and '--from=' not in rest
        for w in [x for x in rest.split() if not x.startswith('--')][:-1]
    ]
    dirs = {c.rstrip('/') for c in copied if c.endswith('/')}
    files = {c for c in copied if not c.endswith('/')}
    missing = {
        p: sorted(users) for p, users in sorted(needed.items()) if not _covered(p, dirs, files)
    }
    assert missing == {}, f'api target does not COPY paths its services run: {missing}'


def test_export_image_copies_every_scripts_file_it_imports_or_sources() -> None:
    """`openprocessor models install` and `sample coco` run in this image
    (scripts/lib/model_setup.sh)."""
    dirs, files = _copied_scripts('Dockerfile', 'export')
    assert 'scripts' not in dirs
    needed = _scripts_modules_imported(
        ['export', 'models', 'config_templates', 'src/ultralytics_patches']
    )
    needed |= _scripts_modules_imported(['scripts/datasets'])
    needed |= _COCO_SAMPLE | _SHELL_LIBS
    missing = sorted(n for n in _with_sourced_libs(needed) if not _covered(n, dirs, files))
    assert missing == [], f'export target does not COPY: {missing}'


def test_evaluator_image_copies_every_scripts_file_it_imports() -> None:
    dirs, files = _copied_scripts('Dockerfile', 'evaluator')
    assert 'scripts' not in dirs
    needed = _scripts_modules_imported(['src', 'scripts/curation/bakeoff'])
    missing = sorted(n for n in needed if not _covered(n, dirs, files))
    assert missing == [], f'evaluator target does not COPY: {missing}'


def test_trainer_sets_its_build_identity_after_the_big_layers() -> None:
    stage = _stages('docker/trainer/Dockerfile')[-1]
    sha_at = next(
        i for i, (k, r) in enumerate(stage) if k == 'ARG' and r.startswith('OP_BUILD_SHA')
    )
    sha_pin = next(
        i for i, (k, r) in enumerate(stage) if k == 'ARG' and r.startswith('ULTRALYTICS_SHA')
    )
    last_copy = max(i for i, (k, _) in enumerate(stage) if k == 'COPY')
    assert sha_at > last_copy
    assert sha_pin > last_copy


@pytest.mark.parametrize('rel', ['docker/trainer/Dockerfile', 'docker/segmenter/Dockerfile'])
def test_runtime_stage_strips_the_base_images_pip(rel: str) -> None:
    runs = ' '.join(r for k, r in _stages(rel)[-1] if k == 'RUN')
    assert re.search(r'rm -rf /usr/local/lib/python3\.12/(site|dist)-packages/pip\b', runs)


def test_segmenter_has_no_wget_probe_or_package() -> None:
    """The probe is Python (urllib), in the Dockerfile and in the compose override alike."""
    assert 'wget' not in (REPO_ROOT / 'docker' / 'segmenter' / 'Dockerfile').read_text().replace(
        '`wget`', ''
    ).replace('# wget', '')
    compose = (REPO_ROOT / 'docker-compose.yml').read_text()
    segmenter = compose[compose.index('\n  segmenter:') :].split('\n  curation-trainer:')[0]
    assert 'wget' not in segmenter


def _stage_chain(target: str) -> list[str]:
    named = _named_stages()
    chain = [target]
    while True:
        first = named[chain[-1]][0][1].split()
        parent = next(w for w in first if not w.startswith('--'))
        if parent not in named:
            return chain
        chain.append(parent)


def _apt_packages(stage: str) -> set[str]:
    text = ' '.join(rest for kind, rest in _named_stages()[stage] if kind == 'RUN')
    return set(re.findall(r'[a-z][a-z0-9+.-]+', text))


@pytest.mark.parametrize('target', ['export', 'evaluator'])
def test_psycopg_c_has_a_compiler_and_libpq_in_the_image_chain(target: str) -> None:
    """psycopg[c] compiles in the builder and links libpq5 at run time (R-27)."""
    text = (REPO_ROOT / 'requirements.txt').read_text(encoding='utf-8')
    requirement = re.search(r'^psycopg\[([^\]]+)\]', text, re.M)
    assert requirement is not None
    assert requirement.group(1) == 'c', 'requirements.txt must install psycopg[c]'
    assert 'runtime-base' in _stage_chain(target)
    # The compiled site comes from `builder` through runtime-base; libpq5 is in `base`.
    assert any(k == 'COPY' and '--from=builder ' in r for k, r in _named_stages()['runtime-base'])
    assert {'build-essential', 'gcc', 'libpq-dev'} <= _apt_packages('builder')
    assert 'libpq5' in _apt_packages('base')


def test_psycopg_c_is_the_loaded_implementation_where_it_is_installed() -> None:
    import importlib.util
    import sys

    if importlib.util.find_spec('psycopg_c') is None:
        pytest.skip('psycopg_c is not installed in this interpreter (CI uses the binary wheel)')
    out = subprocess.run(
        [sys.executable, '-I', '-c', 'import psycopg; print(psycopg.pq.__impl__)'],
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.stdout.strip() == 'c'
