"""P1 commit 3: the OpenSearch project guard (§2.4)."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pytest

from src.config.curation import base_curation_config
from src.config.project_context import ProjectNotBound, bind_project
from src.config.projects import ProjectRecord, resources_for_new
from src.services.projects.guard import (
    CrossProjectAccess,
    ProjectGuardedTransport,
    ProjectReadOnly,
    check_request,
    cross_project_access_count,
)


# These tests are about binding itself: no autouse `default` binding.
pytestmark = pytest.mark.unbound


def _record(slug: str) -> ProjectRecord:
    now = datetime.now(UTC).isoformat()
    return ProjectRecord(
        slug=slug,
        display_name=slug,
        description='',
        status='active',
        revision=1,
        created_at=now,
        updated_at=now,
        origin=None,
        resources=resources_for_new(slug, base_curation_config()),
    )


@pytest.fixture
def snapshot() -> dict[str, ProjectRecord]:
    return {'alpha': _record('alpha'), 'beta': _record('beta')}


def test_own_index_read_passes(snapshot) -> None:
    with bind_project(snapshot['alpha']):
        check_request('GET', '/op_prj_alpha__items/_doc/x', None, snapshot)


def test_other_project_index_raises_cross_project_access(snapshot) -> None:
    before = cross_project_access_count()
    with bind_project(snapshot['alpha']), pytest.raises(CrossProjectAccess):
        check_request('GET', '/op_prj_beta__items/_doc/x', None, snapshot)
    assert cross_project_access_count() == before + 1


def test_wildcard_rejected(snapshot) -> None:
    with bind_project(snapshot['alpha']), pytest.raises(CrossProjectAccess):
        check_request('GET', '/*/_search', None, snapshot)


def test_all_rejected(snapshot) -> None:
    with bind_project(snapshot['alpha']), pytest.raises(CrossProjectAccess):
        check_request('GET', '/_all/_search', None, snapshot)


def test_unowned_index_passes_unbound(snapshot) -> None:
    check_request('GET', '/visual_search_global/_search', None, snapshot)
    check_request('GET', '/op_projects/_doc/project:default', None, snapshot)


def test_global_configs_index_is_a_legitimate_unowned_index(snapshot) -> None:
    """M3: ``op_global_configs`` (the one config-store index scoped to no
    project -- sibling to ``op_projects``, ``visual_search_*``) is
    readable and writable unbound, same as any other unowned index -- the
    shape its future global-router routes (W9) will use. A request
    already bound to a project has no business touching it, so it is
    refused there exactly like any other unowned index (fail-closed)."""
    check_request('GET', '/op_global_configs/_doc/pack:local_vlm', None, snapshot)
    check_request('PUT', '/op_global_configs/_doc/pack:local_vlm', {'a': 1}, snapshot)
    with bind_project(snapshot['alpha']), pytest.raises(CrossProjectAccess):
        check_request('GET', '/op_global_configs/_doc/pack:local_vlm', None, snapshot)


def test_unbound_plus_project_index_raises_not_bound(snapshot) -> None:
    with pytest.raises(ProjectNotBound):
        check_request('GET', '/op_prj_alpha__items/_doc/x', None, snapshot)


def test_read_only_binding_rejects_write(snapshot) -> None:
    with (
        bind_project(snapshot['alpha'], read_only=True),
        pytest.raises(ProjectReadOnly),
    ):
        check_request('PUT', '/op_prj_alpha__items/_doc/x', b'{}', snapshot)


def test_read_only_binding_allows_read(snapshot) -> None:
    with bind_project(snapshot['alpha'], read_only=True):
        check_request('GET', '/op_prj_alpha__items/_doc/x', None, snapshot)


def test_bulk_body_cross_project_index_rejected(snapshot) -> None:
    body = b'{"index": {"_index": "op_prj_beta__items", "_id": "x"}}\n{"class_id": 0}\n'
    with bind_project(snapshot['alpha']), pytest.raises(CrossProjectAccess):
        check_request('POST', '/_bulk', body, snapshot)


def test_bulk_body_own_project_passes(snapshot) -> None:
    body = b'{"index": {"_index": "op_prj_alpha__items", "_id": "x"}}\n{"class_id": 0}\n'
    with bind_project(snapshot['alpha']):
        check_request('POST', '/_bulk', body, snapshot)


def test_msearch_body_cross_project_rejected(snapshot) -> None:
    body = b'{"index": "op_prj_beta__items"}\n{"query": {"match_all": {}}}\n'
    with bind_project(snapshot['alpha']), pytest.raises(CrossProjectAccess):
        check_request('POST', '/_msearch', body, snapshot)


def test_mget_body_cross_project_rejected(snapshot) -> None:
    body = b'{"docs": [{"_index": "op_prj_beta__items", "_id": "x"}]}'
    with bind_project(snapshot['alpha']), pytest.raises(CrossProjectAccess):
        check_request('POST', '/_mget', body, snapshot)


def test_update_by_query_write_verb_on_own_project_passes(snapshot) -> None:
    with bind_project(snapshot['alpha']):
        check_request('POST', '/op_prj_alpha__items/_update_by_query', b'{}', snapshot)


def test_guarded_transport_delegates_and_checks() -> None:
    calls: list[tuple[str, str]] = []

    class _InnerTransport:
        async def perform_request(self, method, url, params=None, body=None, **kwargs):  # noqa: ARG002
            calls.append((method, url))
            return {'ok': True}

    class _Registry:
        def __init__(self, snap):
            self._snap = snap

        def snapshot(self):
            return self._snap

    snap = {'alpha': _record('alpha')}
    guarded = ProjectGuardedTransport(_InnerTransport(), _Registry(snap))

    async def _run() -> None:
        with bind_project(snap['alpha']):
            result = await guarded.perform_request('GET', '/op_prj_alpha__items/_doc/x')
            assert result == {'ok': True}
        with pytest.raises(ProjectNotBound):
            await guarded.perform_request('GET', '/op_prj_alpha__items/_doc/x')

    asyncio.run(_run())
    assert calls == [('GET', '/op_prj_alpha__items/_doc/x')]
