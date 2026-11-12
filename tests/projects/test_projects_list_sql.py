"""The project list is paged by Postgres: the statement carries the keyset predicate, the order
and the LIMIT (compiled offline; the rows are checked in tests/db/repos/test_projects_repo.py)."""

from __future__ import annotations

import datetime as dt

import pytest
from sqlalchemy.dialects import postgresql

from src.db.repos.projects import list_page_stmt


def _sql(**kw: object) -> str:
    args: dict[str, object] = {
        'statuses': ('active',),
        'q': None,
        'sort': 'name:asc',
        'after': None,
        'limit': 26,
    }
    return str(list_page_stmt(**{**args, **kw}).compile(dialect=postgresql.dialect()))  # type: ignore[arg-type]


def test_name_page_orders_by_slug_and_limits() -> None:
    sql = _sql()
    assert 'ORDER BY projects.slug' in sql
    assert 'LIMIT' in sql


def test_name_keyset_predicate_follows_the_cursor() -> None:
    assert 'projects.slug >' in _sql(after=('b', 'b'))


def test_recency_keyset_is_one_row_comparison() -> None:
    sql = _sql(sort='updated_at:desc', after=(dt.datetime(2026, 1, 1, tzinfo=dt.UTC), 'b'))
    assert '(projects.updated_at, projects.slug) <' in sql
    assert 'ORDER BY projects.updated_at DESC, projects.slug DESC' in sql
    assert 'LIMIT' in sql


def test_text_search_is_escaped() -> None:
    assert 'ESCAPE' in _sql(q='a_b')


def test_unknown_sort_has_no_statement() -> None:
    with pytest.raises(ValueError, match='no keyset sort'):
        _sql(sort='items:desc')
