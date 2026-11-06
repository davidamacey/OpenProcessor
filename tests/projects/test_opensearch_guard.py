"""P1 commit 3: the OpenSearch project guard (§2.4)."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pytest

from src.config.curation_accessors import base_curation_config
from src.config.project_context import ProjectNotBound, bind_project
from src.config.projects import ProjectRecord, resources_for_new
from src.services.projects.guard import ProjectGuardedTransport
from src.services.projects.guard_errors import (
    CrossProjectAccess,
    ProjectReadOnly,
    cross_project_access_count,
)
from src.services.projects.guard_policy import check_request


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


def _bulk(n_items: int, index: str) -> bytes:
    """An ingest-shaped bulk body: an action line, then a document with a vector."""
    doc = '{"crop_id": "c", "pe_embedding": [' + ', '.join(['0.25'] * 1024) + ']}'
    action = f'{{"create": {{"_index": "{index}", "_id": "x"}}}}'
    return ('\n'.join([action, doc] * n_items) + '\n').encode()


def test_bulk_document_lines_are_not_parsed(snapshot, monkeypatch) -> None:
    from src.services.projects import guard_parse

    parsed: list[int] = []
    real_loads = guard_parse.json.loads

    def counting_loads(text, *args, **kwargs):
        parsed.append(len(text))
        return real_loads(text, *args, **kwargs)

    monkeypatch.setattr(guard_parse.json, 'loads', counting_loads)
    with bind_project(snapshot['alpha']):
        check_request('POST', '/_bulk', _bulk(6, 'op_prj_alpha__items'), snapshot)
    assert len(parsed) == 6  # one action line per item
    assert max(parsed) < 200  # none of the 8 KB document lines


def test_bulk_second_action_naming_a_foreign_index_is_still_rejected(snapshot) -> None:
    own = _bulk(1, 'op_prj_alpha__items')
    foreign = _bulk(1, 'op_prj_beta__items')
    with bind_project(snapshot['alpha']), pytest.raises(CrossProjectAccess):
        check_request('POST', '/_bulk', own + foreign, snapshot)


def test_bulk_delete_has_no_document_line(snapshot) -> None:
    body = (
        b'{"delete": {"_index": "op_prj_alpha__items", "_id": "x"}}\n'
        b'{"delete": {"_index": "op_prj_beta__items", "_id": "y"}}\n'
    )
    with bind_project(snapshot['alpha']), pytest.raises(CrossProjectAccess):
        check_request('POST', '/_bulk', body, snapshot)


def test_bulk_document_line_that_is_not_an_object_is_refused(snapshot) -> None:
    body = b'{"index": {"_index": "op_prj_alpha__items", "_id": "x"}}\nnot json\n'
    with bind_project(snapshot['alpha']), pytest.raises(CrossProjectAccess):
        check_request('POST', '/_bulk', body, snapshot)


def test_bulk_action_line_that_is_not_json_is_refused(snapshot) -> None:
    with bind_project(snapshot['alpha']), pytest.raises(CrossProjectAccess):
        check_request('POST', '/_bulk', b'oops\n{}\n', snapshot)
