"""The guard's ownership map comes from ``project_indexes``, through the registry snapshot."""

from __future__ import annotations

import asyncio

import pytest

from src.config.project_context import bind_project
from src.services.projects.guard_errors import CrossProjectAccess
from src.services.projects.guard_policy import check_request
from src.services.projects.registry import ProjectRegistry

from .conftest import aseed_project


pytestmark = [pytest.mark.db, pytest.mark.usefixtures('pg_registry')]


def _snapshot():
    async def _load():
        await aseed_project('alpha')
        await aseed_project('beta')
        registry = ProjectRegistry()
        await registry.refresh_strict()
        return registry.snapshot()

    return asyncio.run(_load())


def test_a_request_naming_another_projects_index_is_refused() -> None:
    snapshot = _snapshot()
    with bind_project(snapshot['alpha']), pytest.raises(CrossProjectAccess):
        check_request('POST', '/op_prj_beta__items/_search', {}, snapshot)


def test_a_request_naming_the_removed_registry_index_is_refused_while_bound() -> None:
    snapshot = _snapshot()
    with bind_project(snapshot['alpha']), pytest.raises(CrossProjectAccess):
        check_request('GET', '/op_projects/_doc/project:alpha', None, snapshot)


def test_the_bound_projects_own_indexes_pass() -> None:
    snapshot = _snapshot()
    with bind_project(snapshot['alpha']):
        check_request('POST', '/op_prj_alpha__items/_search', {}, snapshot)
        check_request('POST', '/op_prj_alpha__images/_search', {}, snapshot)


def test_an_unknown_index_under_the_project_prefix_is_refused() -> None:
    snapshot = _snapshot()
    with (
        bind_project(snapshot['alpha']),
        pytest.raises(CrossProjectAccess, match='belongs to no known project'),
    ):
        check_request('POST', '/op_prj_gamma__items/_search', {}, snapshot)
