"""``ensure_indexes`` needs no in-process lock, and its fast path touches nothing.

The API runs about thirty uvicorn worker processes plus separate worker containers, all
bootstrapping the same indexes, so an ``asyncio.Lock`` gave no exclusion across them. The
bootstrap is safe without one because every step is idempotent (creation treats
``resource_already_exists_exception`` as success, the mapping puts are additive, the
schema marker is written last); ``test_index_schema_marker.py`` runs it concurrently.
"""

from __future__ import annotations

import ast
import asyncio
import inspect

import pytest

from src.services.curation import index_bootstrap


@pytest.fixture(autouse=True)
def _reset_bootstrapped_flag():
    index_bootstrap._INDEXES_BOOTSTRAPPED = set()
    yield
    index_bootstrap._INDEXES_BOOTSTRAPPED = set()


def test_the_bootstrap_holds_no_in_process_lock() -> None:
    """A lock here would only hide the races between processes (and bind to one event
    loop), so the module must not build one."""
    tree = ast.parse(inspect.getsource(index_bootstrap))
    locks = [
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and node.attr in {'Lock', 'Semaphore', 'Condition'}
    ]
    assert not locks


@pytest.mark.asyncio
async def test_an_already_bootstrapped_project_makes_no_opensearch_call() -> None:
    index_bootstrap._INDEXES_BOOTSTRAPPED = {'default'}

    class _Untouchable:
        def __getattr__(self, name: str) -> object:
            msg = f'unexpected OpenSearch use: {name}'
            raise AssertionError(msg)

    assert await asyncio.wait_for(index_bootstrap.ensure_indexes(_Untouchable()), timeout=1.0)
