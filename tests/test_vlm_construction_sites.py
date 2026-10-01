"""Where a VLM labeler may be built, and how an HTTP client to an endpoint
may be built (W9.0). One factory constructs every labeler, so an endpoint's
image cap, JSON mode, timeout, rate limit and identity apply the same way
everywhere; and no client to a model endpoint follows a redirect, so a
registered URL cannot bounce a request (or its key) somewhere else."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCAN = ('src', 'scripts')
FACTORY = 'src/services/labeling/vlm_factory.py'


def _sources() -> list[tuple[str, ast.AST]]:
    return [
        (path.relative_to(ROOT).as_posix(), ast.parse(path.read_text()))
        for top in SCAN
        for path in sorted((ROOT / top).rglob('*.py'))
    ]


def _calls(tree: ast.AST, name: str) -> list[ast.Call]:
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            called = func.id if isinstance(func, ast.Name) else getattr(func, 'attr', None)
            if called == name:
                out.append(node)
    return out


def test_only_the_factory_constructs_a_vlm_labeler() -> None:
    builders = {rel for rel, tree in _sources() if _calls(tree, 'VlmLabeler') and rel != FACTORY}
    assert builders == set(), f'construct labelers through vlm_factory: {sorted(builders)}'
    assert _calls(dict(_sources())[FACTORY], 'VlmLabeler')


def test_the_worker_builds_its_labeler_through_the_factory_alias() -> None:
    rel = 'scripts/curation/worker/__init__.py'
    tree = dict(_sources())[rel]
    imported = {
        alias.asname or alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == 'src.services.labeling.vlm_factory'
        for alias in node.names
    }
    assert 'build_vlm_labeler' in imported
    assert not any(
        isinstance(n, ast.ImportFrom)
        and n.module == 'src.services.labeling.vlm_labeler'
        and any(a.name == 'VlmLabeler' for a in n.names)
        for n in ast.walk(tree)
    )


def test_no_http_client_in_the_labeling_package_follows_redirects() -> None:
    offenders = []
    for rel, tree in _sources():
        if not rel.startswith('src/services/labeling/'):
            continue
        for call in _calls(tree, 'AsyncClient') + _calls(tree, 'Client'):
            flag = next((k.value for k in call.keywords if k.arg == 'follow_redirects'), None)
            if not (isinstance(flag, ast.Constant) and flag.value is False):
                offenders.append(f'{rel}:{call.lineno}')
    assert offenders == [], f'follow_redirects=False must be explicit: {offenders}'


def test_nothing_turns_redirects_on_anywhere_a_model_endpoint_is_called() -> None:
    offenders = [
        f'{rel}:{node.value.lineno}'
        for rel, tree in _sources()
        for node in ast.walk(tree)
        if isinstance(node, ast.keyword)
        and node.arg == 'follow_redirects'
        and isinstance(node.value, ast.Constant)
        and node.value.value is True
    ]
    assert offenders == [], offenders


def test_only_the_endpoint_registry_reads_the_vlm_environment_variables() -> None:
    """``OP_VLM_URL`` / ``OP_VLM_MODEL`` / ``OP_VLM_API_KEY`` are the ``env``
    built-in's definition. Reading them anywhere else would be a second,
    ungated way to pick an endpoint."""
    allowed = {
        'src/services/labeling/vlm_endpoints.py',  # the built-in itself
        'src/services/labeling/vlm_client.py',  # the labeler constructor's unit-test defaults
        'src/config/retired_env.py',  # names of retired variables, for the startup warning
    }
    readers = set()
    for rel, tree in _sources():
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and node.value in (
                'OP_VLM_URL',
                'OP_VLM_MODEL',
                'OP_VLM_API_KEY',
            ):
                readers.add(rel)
    assert readers <= allowed, sorted(readers - allowed)
