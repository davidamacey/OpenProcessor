"""No index is created, migrated or warmed in the API lifespan (R-7): every uvicorn worker of
every replica, and every worker recycled by ``--limit-max-requests``, would repeat it. The work
runs once in ``db-migrate`` (``scripts/opensearch/migrate.py``); the lifespan only checks stamps."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import src.core.dependencies as dependencies_module
from src.core.dependencies import bootstrap_opensearch_indexes as startup_check


SRC = Path(__file__).resolve().parents[2] / 'src'
FORBIDDEN = {
    'create_all_indexes',
    'create_curation_indexes',
    'ensure_indexes',
    'warm_knn_indexes',
    '_bootstrap_indexes',
    'ensure_indexes_cached',
}


def _called_names(path: Path, function: str | None = None) -> set[str]:
    tree = ast.parse(path.read_text(encoding='utf-8'))
    scopes: list[ast.AST] = [tree]
    if function is not None:
        scopes = [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef) and n.name == function
        ]
        assert scopes, f'{function} not found in {path}'
    names: set[str] = set()
    for scope in scopes:
        for node in ast.walk(scope):
            if isinstance(node, ast.Call):
                func = node.func
                names.add(func.attr if isinstance(func, ast.Attribute) else getattr(func, 'id', ''))
            elif isinstance(node, ast.alias):
                names.add(node.name)
    return names


def test_the_lifespan_module_makes_no_bootstrap_call() -> None:
    assert not FORBIDDEN & _called_names(SRC / 'app' / 'lifespan.py')


def test_the_function_the_lifespan_calls_only_checks_stamps() -> None:
    names = _called_names(SRC / 'core' / 'dependencies.py', 'bootstrap_opensearch_indexes')

    assert not FORBIDDEN & names
    assert 'check_mappings_current' in names


def test_no_curation_router_bootstraps_an_index() -> None:
    offenders = [
        str(path.relative_to(SRC))
        for path in (SRC / 'routers').rglob('*.py')
        if FORBIDDEN & _called_names(path)
    ]

    assert offenders == []


@pytest.mark.asyncio
async def test_it_returns_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    class Client:
        client = object()

    async def get_client() -> Client:
        return Client()

    monkeypatch.setattr(dependencies_module.OpenSearchClientFactory, 'get_client', get_client)
    monkeypatch.setattr(
        'src.services.projects.bootstrap.for_each_project', lambda *_a, **_k: iter(())
    )

    assert await startup_check() is None
