"""A job handler's OpenSearch client is the worker's own, never the API's shared one.

``make_curation_opensearch`` unwraps the API's shared client, whose project registry is kept
fresh by the API's poll loop. A Celery worker runs no such loop, so that registry stays empty and
the guard refuses every index as 'belongs to no known project'.
:func:`src.services.curation.job_binding.opensearch_client` refreshes the registry on use.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest


TASKS = Path(__file__).resolve().parents[2] / 'src' / 'tasks'
FORBIDDEN = 'make_curation_opensearch'


@pytest.mark.parametrize(
    'path', sorted(TASKS.rglob('*.py')), ids=lambda p: str(p.relative_to(TASKS))
)
def test_the_task_module_does_not_build_the_api_client(path: Path) -> None:
    names = {
        node.id if isinstance(node, ast.Name) else node.attr
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8')))
        if isinstance(node, ast.Name | ast.Attribute)
    } | {
        alias.name
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8')))
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    assert FORBIDDEN not in names
