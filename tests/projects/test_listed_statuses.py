"""``listed_statuses``: which project statuses a ``GET /projects`` request matches (no database)."""

from __future__ import annotations

import pytest

from src.services.projects import listing


def _listed(**kw: object) -> tuple[str, ...]:
    args: dict[str, object] = {'include_archived': False, 'selectable': False, 'status': None}
    return listing.listed_statuses(**{**args, **kw})  # type: ignore[arg-type]


def test_an_unknown_status_is_refused_not_an_empty_list() -> None:
    with pytest.raises(listing.UnknownStatusError, match='bogus'):
        _listed(status=['active', 'bogus'])


def test_a_deleted_project_is_never_a_filter_value() -> None:
    with pytest.raises(listing.UnknownStatusError):
        _listed(status=['deleted'])


def test_naming_archived_lists_the_archived_projects_without_include_archived() -> None:
    assert _listed(status=['archived']) == ('archived',)
    assert _listed(status=['archived', 'failed']) == ('failed', 'archived')


def test_the_default_list_hides_archived() -> None:
    assert 'archived' not in _listed()
    assert 'archived' in _listed(include_archived=True)


def test_selectable_still_narrows_the_named_statuses() -> None:
    assert _listed(selectable=True, status=['active', 'failed']) == ('active',)
