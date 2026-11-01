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
        f'{SEED}/ocr/1/__pycache__/generated.cpython-312.pyc',
    }
    bad = audit.untracked_files(DOCKERFILE, '.', TRACKED, listing)
    assert f'{SEED}/x__y/1/model.onnx' in bad
    assert f'{SEED}/x__y/promote.json' in bad
    assert f'{SEED}/ocr/1/__pycache__/generated.cpython-312.pyc' in bad
    assert f'{SEED}/yolo/config.pbtxt' not in bad


def test_pyc_is_accepted_only_next_to_its_tracked_source() -> None:
    audit = _load()
    tracked = TRACKED | {'src/utils/retry.py', 'models/gpu_image_prep/1/model.py'}
    ok = '/app/src/utils/__pycache__/retry.cpython-312.pyc'
    assert audit.untracked_files(DOCKERFILE, '.', tracked, {ok}) == []
    no_source = f'{SEED}/gpu_image_prep/1/__pycache__/foo.cpython-312.pyc'
    orphan = '/app/src/__pycache__/ghost.cpython-312.pyc'
    assert audit.untracked_files(DOCKERFILE, '.', tracked, {no_source, orphan}) == [
        orphan,
        no_source,
    ]
    # the source must be in the same directory as the __pycache__, not just somewhere
    wrong_dir = '/app/src/__pycache__/retry.cpython-312.pyc'
    assert audit.untracked_files(DOCKERFILE, '.', tracked, {wrong_dir}) == [wrong_dir]
    # a non-.pyc file in __pycache__ is never accepted
    junk = '/app/src/utils/__pycache__/retry.txt'
    assert audit.untracked_files(DOCKERFILE, '.', tracked, {junk}) == [junk]


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
    kw = {'large': {path}, 'model_roots': [SEED]}
    assert audit.forbidden_files([path], [], **kw) == [path]
    assert audit.forbidden_files([path], [path], **kw) == []
    assert audit.forbidden_files([path], ['/opt/*', '/a/*'], **kw) == []


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
    onnx = f'{SEED}/proj__model/1/model.onnx'
    assert audit.forbidden_files([onnx], [], large={onnx}, model_roots=roots)
    assert audit.forbidden_files([f'{SEED}/deep/proj__model'], [], model_roots=roots) == []
    assert audit.forbidden_files(['/app/foo__bar/x.txt'], [], model_roots=roots) == []


def test_model_extension_is_size_gated_everywhere() -> None:
    audit = _load()
    tutor = '/usr/share/x/tutor.pt'
    weights = '/opt/foo/model.pt'
    pth = f'{SITE}/_pynvml_redirector.pth'
    assert audit.forbidden_files([tutor, weights, pth], [], large=set()) == []
    assert audit.forbidden_files([tutor, weights, pth], [], large={weights}) == [weights]
    assert audit.forbidden_files([pth], [], large={pth}) == [pth]


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


# The exact lines the 0.6.0 API and evaluator Dockerfiles use.
REAL_LINES = """\
FROM python:3.13-slim-trixie
WORKDIR /app
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser models/   /opt/openprocessor/model_repo_seed/
"""
REAL_TRACKED = {
    'src/utils/cpu_pool.py',
    'src/utils/retry.py',
    'src/utils/vectors.py',
    'models/ocr_pipeline/config.pbtxt',
    'models/pe_image_encoder/config.pbtxt',
    'models/paddleocr_rec_trt/ppocrv5_dict.txt',
    'models/yolo26_small_trt/labels.txt',
}


def _image_listing_for(tracked: set[str]) -> set[str]:
    out = set()
    for t in tracked:
        if t.startswith('src/'):
            out.add(f'/app/{t}')
        else:
            out.add(f'{SEED}/{t.removeprefix("models/")}')
    return out


def test_real_dockerfile_lines_report_zero_untracked_for_tracked_files() -> None:
    audit = _load()
    listing = _image_listing_for(REAL_TRACKED)
    assert audit.untracked_files(REAL_LINES, '.', REAL_TRACKED, listing) == []
    extra = f'{SEED}/x__y/1/model.onnx'
    assert audit.untracked_files(REAL_LINES, '.', REAL_TRACKED, listing | {extra}) == [extra]


def test_context_that_is_not_a_repo_relative_directory_is_a_clear_error() -> None:
    """An absolute or wrong --context maps every source to nothing; that must not read as
    "every file is untracked"."""
    audit = _load()
    with pytest.raises(audit.ContextMismatchError, match='no tracked files'):
        audit.untracked_files(REAL_LINES, '/tmp/export', REAL_TRACKED, set())
    with pytest.raises(audit.ContextMismatchError, match='no tracked files'):
        audit.directory_copy_roots(REAL_LINES, 'docker/segmenter', REAL_TRACKED)


MULTI_TARGET = """\
FROM python:3.13 AS base
FROM base AS export
COPY --chown=u:u export/ ./export/
COPY --chown=u:u models/ /opt/openprocessor/model_repo_seed/
FROM base AS api
COPY --chown=u:u src/ ./src/
"""


def test_target_selects_the_stage_whose_copies_are_audited() -> None:
    audit = _load()
    tracked = {'src/main.py', 'export/a.py', 'models/yolo/config.pbtxt'}
    roots = audit.directory_copy_roots(MULTI_TARGET, '.', tracked, target='export')
    assert roots == ['/export', '/opt/openprocessor/model_repo_seed']
    assert audit.directory_copy_roots(MULTI_TARGET, '.', tracked) == ['/src']
    listing = {'/export/a.py', '/export/stray.py'}
    assert audit.untracked_files(MULTI_TARGET, '.', tracked, listing, target='export') == [
        '/export/stray.py'
    ]
    assert audit.model_roots(MULTI_TARGET, target='export') == [
        '/opt/openprocessor/model_repo_seed'
    ]


def test_unknown_target_is_a_clear_error() -> None:
    audit = _load()
    with pytest.raises(audit.ContextMismatchError, match='no stage'):
        audit.directory_copy_roots(MULTI_TARGET, '.', {'src/main.py'}, target='nope')


INHERITED_WORKDIR = """\
FROM python:3.13 AS base
WORKDIR /app
FROM base AS runtime-base
FROM runtime-base AS export
COPY --chown=u:u export/ ./export/
FROM runtime-base AS api
COPY --chown=u:u src/ ./src/
"""


def test_a_stage_inherits_the_workdir_of_the_stage_it_builds_on() -> None:
    audit = _load()
    tracked = {'src/main.py', 'export/a.py'}
    assert audit.directory_copy_roots(INHERITED_WORKDIR, '.', tracked) == ['/app/src']
    assert audit.directory_copy_roots(INHERITED_WORKDIR, '.', tracked, target='export') == [
        '/app/export'
    ]
