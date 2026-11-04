"""Three bulk writers count what OpenSearch refused instead of discarding the response.

A bulk answers 200 with ``errors: true`` and a per-item ``status``; a job that ignores the
body reports every item written.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from src.services.curation import embedding_viz, probe_predictions as pp
from src.services.curation.item_scores import job as scores_job


def _bulk_response(n: int, *, refused: int) -> dict[str, Any]:
    items = [{'update': {'_id': f'c{i}', 'status': 200}} for i in range(n - refused)]
    items += [
        {'update': {'_id': f'c{i}', 'status': 400, 'error': {'type': 'mapper_parsing_exception'}}}
        for i in range(n - refused, n)
    ]
    return {'errors': bool(refused), 'items': items}


class _Bulk:
    def __init__(self, *, refused: int) -> None:
        self.refused = refused

    async def bulk(self, *, body: list[Any], refresh: bool = False) -> dict[str, Any]:  # noqa: ARG002
        return _bulk_response(len(body) // 2, refused=min(self.refused, len(body) // 2))


@pytest.mark.asyncio
async def test_score_write_reports_the_refused_items() -> None:
    result = SimpleNamespace(fields={f'c{i}': {'score': 1.0} for i in range(4)})

    assert await scores_job.bulk_write_result(_Bulk(refused=3), result) == 3  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_score_write_control_all_ok_reports_none() -> None:
    result = SimpleNamespace(fields={f'c{i}': {'score': 1.0} for i in range(4)})

    assert await scores_job.bulk_write_result(_Bulk(refused=0), result) == 0  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_viz_write_counts_only_the_accepted_items() -> None:
    ids = [f'c{i}' for i in range(5)]
    xy = np.zeros((5, 2), dtype=np.float32)

    assert await embedding_viz._bulk_write_coordinates(_Bulk(refused=2), ids, xy) == 3  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_viz_write_fails_when_nothing_was_written() -> None:
    ids = [f'c{i}' for i in range(2)]
    xy = np.zeros((2, 2), dtype=np.float32)

    with pytest.raises(RuntimeError, match='no viz coordinate'):
        await embedding_viz._bulk_write_coordinates(_Bulk(refused=2), ids, xy)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_viz_write_control_all_ok_counts_all() -> None:
    ids = [f'c{i}' for i in range(3)]
    xy = np.zeros((3, 2), dtype=np.float32)

    assert await embedding_viz._bulk_write_coordinates(_Bulk(refused=0), ids, xy) == 3  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_probe_flush_returns_the_refused_count() -> None:
    body = [x for i in range(4) for x in ({'update': {'_id': f'c{i}'}}, {'doc': {}})]

    assert await pp._flush_bulk(_Bulk(refused=1), body) == 1
    assert await pp._flush_bulk(_Bulk(refused=0), body) == 0


def test_no_curation_service_discards_a_bulk_response() -> None:
    """An awaited ``.bulk(...)`` used as a bare statement drops the per-item errors."""
    import ast
    from pathlib import Path

    offenders: list[str] = []
    for path in sorted(Path('src/services/curation').rglob('*.py')):
        for node in ast.walk(ast.parse(path.read_text())):
            call = node.value if isinstance(node, ast.Expr) else None
            call = call.value if isinstance(call, ast.Await) else None
            func = getattr(call, 'func', None)
            if isinstance(func, ast.Attribute) and func.attr == 'bulk':
                offenders.append(f'{path}:{node.lineno}')
    assert not offenders, offenders
