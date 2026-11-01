"""Offline tests for the pre-publish image audit (scripts/release/audit_image_files.py).

The audit compares the files inside a built image with the committed tree. These
tests feed it fake listings; nothing here runs docker.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / 'scripts' / 'release' / 'audit_image_files.py'


def _load():
    spec = importlib.util.spec_from_file_location('audit_image_files', MODULE_PATH)
    assert spec
    assert spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules['audit_image_files'] = mod
    spec.loader.exec_module(mod)
    return mod


DOCKERFILE = """\
FROM python:3.13 AS builder
COPY models/ /builder/models/
FROM python:3.13
WORKDIR /app
COPY --chown=u:u src/ ./src/
COPY --chown=u:u models/   /opt/openprocessor/model_repo_seed/
COPY --chown=u:u scripts/lib/a.sh scripts/lib/b.sh ./scripts/lib/
COPY --from=builder /root/.local /home/u/.local
"""

TRACKED = {
    'src/main.py',
    'models/yolo/config.pbtxt',
    'models/yolo/labels.txt',
    'models/ocr/1/model.py',
    'scripts/lib/a.sh',
    'scripts/lib/b.sh',
}

SEED = '/opt/openprocessor/model_repo_seed'


def test_clean_image_passes() -> None:
    audit = _load()
    listing = {
        '/app/src/main.py',
        f'{SEED}/yolo/config.pbtxt',
        f'{SEED}/ocr/1/model.py',
        '/app/scripts/lib/a.sh',
    }
    assert audit.untracked_files(DOCKERFILE, '.', TRACKED, listing) == []


def test_untracked_model_in_the_seed_fails() -> None:
    audit = _load()
    listing = {
        f'{SEED}/yolo/config.pbtxt',
        f'{SEED}/x__y/1/model.onnx',
        f'{SEED}/x__y/promote.json',
        f'{SEED}/ocr/1/__pycache__/model.cpython-312.pyc',
    }
    bad = audit.untracked_files(DOCKERFILE, '.', TRACKED, listing)
    assert f'{SEED}/x__y/1/model.onnx' in bad
    assert f'{SEED}/x__y/promote.json' in bad
    assert f'{SEED}/ocr/1/__pycache__/model.cpython-312.pyc' in bad
    assert f'{SEED}/yolo/config.pbtxt' not in bad


def test_pycache_allowed_only_when_the_dockerfile_compiles_it() -> None:
    audit = _load()
    listing = {'/app/src/__pycache__/main.cpython-312.pyc'}
    assert audit.untracked_files(DOCKERFILE, '.', TRACKED, listing)
    compiled = DOCKERFILE + 'RUN python -m compileall /app/src\n'
    assert audit.untracked_files(compiled, '.', TRACKED, listing) == []


def test_non_final_stage_copies_are_not_expected_in_the_image() -> None:
    audit = _load()
    assert audit.untracked_files(DOCKERFILE, '.', TRACKED, {'/builder/models/yolo/x'}) == []


def test_subcontext_sources_map_to_the_context_directory() -> None:
    audit = _load()
    text = 'FROM a\nWORKDIR /app\nCOPY pkg/ /app/pkg/\n'
    tracked = {'docker/seg/pkg/m.py'}
    assert audit.untracked_files(text, 'docker/seg', tracked, {'/app/pkg/m.py'}) == []
    assert audit.untracked_files(text, 'docker/seg', tracked, {'/app/pkg/m.onnx'}) == [
        '/app/pkg/m.onnx'
    ]


@pytest.mark.parametrize(
    'path',
    [
        '/opt/x/model.onnx',
        '/a/model.plan',
        '/a/model.engine',
        '/a/w.pt',
        '/a/w.pth',
        '/a/w.safetensors',
        f'{SEED}/proj__car',
        f'{SEED}/proj__model/1/model.onnx',
    ],
)
def test_forbidden_artifacts_fail_unless_allowlisted(path: str) -> None:
    audit = _load()
    assert audit.forbidden_files([path], [], model_roots=[SEED]) == [path]
    assert audit.forbidden_files([path], [path], model_roots=[SEED]) == []
    assert audit.forbidden_files([path], ['/opt/*', '/a/*'], model_roots=[SEED]) == []


SITE = '/usr/local/lib/python3.13/site-packages'


def test_dunder_names_inside_library_trees_are_not_project_models() -> None:
    audit = _load()
    paths = [
        f'{SITE}/scipy/optimize/tests/test__numdiff.py',
        f'{SITE}/torch/include/ATen/ops/linalg__powsum.h',
        f'{SITE}/scipy/spatial/tests/test__procrustes.py',
        '/app/foo__bar/x.txt',
        '/app/foo__bar',
        '/usr/lib/python3/__pycache__',
    ]
    assert audit.forbidden_files(paths, [], model_roots=[SEED, '/app/models']) == []


def test_project_model_dir_is_forbidden_only_as_a_direct_child_of_a_model_root() -> None:
    audit = _load()
    roots = [SEED]
    assert audit.forbidden_files([f'{SEED}/proj__model'], [], model_roots=roots)
    assert audit.forbidden_files([f'{SEED}/proj__model/1/model.onnx'], [], model_roots=roots)
    assert audit.forbidden_files([f'{SEED}/deep/proj__model'], [], model_roots=roots) == []
    assert audit.forbidden_files(['/app/foo__bar/x.txt'], [], model_roots=roots) == []


def test_small_extension_matches_in_library_trees_pass_but_large_ones_fail() -> None:
    audit = _load()
    pth = f'{SITE}/_pynvml_redirector.pth'
    fixture = f'{SITE}/torch/bin/test_interpreter_async.pt'
    weights = f'{SITE}/somepkg/weights.pt'
    assert audit.forbidden_files([pth, fixture, weights], [], large=set()) == []
    assert audit.forbidden_files([pth, fixture, weights], [], large={weights}) == [weights]


def test_library_tree_prefixes_cover_the_known_install_locations() -> None:
    audit = _load()
    for tree in (
        '/usr/lib/python3/dist-packages',
        '/usr/local/lib/python3.12/site-packages',
        '/opt/tritonserver/backends/x',
        '/opt/venv/lib/python3.13/site-packages',
        '/home/appuser/.local/lib/python3.13/site-packages',
    ):
        path = f'{tree}/small.pt'
        assert audit.forbidden_files([path], [], large=set()) == [], tree
        assert audit.forbidden_files([path], [], large={path}) == [path], tree


def test_model_roots_come_from_the_dockerfile_models_copy() -> None:
    audit = _load()
    roots = audit.model_roots(DOCKERFILE)
    assert SEED in roots
    assert '/app/src' not in roots
    other = 'FROM a\nWORKDIR /srv\nCOPY models/ ./model_repo/\n'
    assert '/srv/model_repo' in audit.model_roots(other)


def test_the_checked_in_allowlist_is_empty() -> None:
    audit = _load()
    allowlist = REPO_ROOT / 'scripts' / 'release' / 'image_audit_allowlist.txt'
    assert audit.load_allowlist(allowlist) == []
