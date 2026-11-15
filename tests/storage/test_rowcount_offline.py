"""An INSERT's row count is only available when the statement asks for it.

SQLAlchemy 2.1 ``CursorResult.rowcount``: "in the default case is *only* useful in conjunction with an
UPDATE or DELETE statement"; for INSERT the value is -1 unless the ``preserve_rowcount`` execution
option is set (https://docs.sqlalchemy.org/en/21/core/connections.html#sqlalchemy.engine.CursorResult.rowcount,
https://docs.sqlalchemy.org/en/21/core/connections.html#sqlalchemy.engine.Connection.execution_options.params.preserve_rowcount).
The server run showed it: every upsert reported -1 rows written. PostgreSQL reports the number of rows
inserted or updated in the INSERT command tag, and a row skipped by the ``WHERE ... IS DISTINCT FROM``
of ``ON CONFLICT DO UPDATE`` is not counted (https://www.postgresql.org/docs/18/sql-insert.html, "Outputs").
"""

from __future__ import annotations

import ast
import asyncio
from pathlib import Path

import pytest
from _fakes import FakeSession

from src.db.models import Detection, ItemProjection
from src.db.repos import upserts


REPOS = Path(upserts.__file__).parent
INSERT_MARKERS = ('insert(', 'on_conflict')


def _upsert_statements(model, rows, **kwargs) -> list:
    session = FakeSession()
    asyncio.run(upserts.upsert_distinct(session, model, rows, **kwargs))
    return [statement for statement, _ in session.calls]


@pytest.mark.parametrize('with_body', [True, False])
def test_every_upsert_statement_asks_for_its_row_count(with_body: bool) -> None:
    if with_body:
        statements = _upsert_statements(
            ItemProjection,
            [{'item_key': 1, 'project_id': 'p', 'analyser_id': 1, 'x': 0.1, 'y': 0.2}],
            key=('item_key',),
        )
    else:
        statements = _upsert_statements(Detection, [{'item_key': 1}], key=('item_key',))
    assert statements
    for statement in statements:
        assert statement.get_execution_options().get('preserve_rowcount') is True


def test_every_chunk_of_a_large_upsert_asks_for_its_row_count(monkeypatch) -> None:
    monkeypatch.setattr(upserts, 'PARAMETER_BUDGET', 10)
    rows = [
        {'item_key': i, 'project_id': 'p', 'analyser_id': 1, 'x': 0.1, 'y': 0.2} for i in range(9)
    ]
    statements = _upsert_statements(ItemProjection, rows, key=('item_key',))
    assert len(statements) > 1
    assert all(s.get_execution_options().get('preserve_rowcount') is True for s in statements)


def test_no_repository_function_reads_rowcount_from_an_insert_without_asking_for_it() -> None:
    offenders = []
    for path in sorted(REPOS.glob('*.py')):
        source = path.read_text()
        for node in ast.walk(ast.parse(source)):
            if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            text = ast.get_source_segment(source, node) or ''
            if '.rowcount' not in text:
                continue
            if any(m in text for m in INSERT_MARKERS) and 'preserve_rowcount' not in text:
                offenders.append(f'{path.name}:{node.name}')
    assert offenders == []
