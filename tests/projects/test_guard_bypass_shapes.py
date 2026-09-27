"""The guard fails closed by construction (projects_plan.md §2.4): only the
request shapes the codebase really sends pass, each naming the bound
project's concrete indexes. One test per bypass shape found by the P1
review (``projects_p1_review_2026-09-26.md`` B1), plus the shapes the
allowlist has to keep refusing.

Every test binds ``beta`` and aims at ``alpha``'s data (or at every
index), the way a buggy or hostile call site would.
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from typing import Any

import pytest
from opensearchpy import AsyncOpenSearch

from src.config.curation import base_curation_config
from src.config.project_context import bind_project
from src.config.projects import ProjectRecord, resources_for_new
from src.services.projects.guard import CrossProjectAccess, check_request, install_project_guard


pytestmark = pytest.mark.unbound

ALPHA_ITEMS = 'op_prj_alpha__items'
BETA_ITEMS = 'op_prj_beta__items'


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


def _refused(snapshot: dict[str, ProjectRecord], method: str, url: str, body: Any = None) -> None:
    with bind_project(snapshot['beta']), pytest.raises(CrossProjectAccess):
        check_request(method, url, body, snapshot)


# --- The review's 11 shapes (B1 probe table) --------------------------------


def test_shape01_other_project_url_index_refused(snapshot) -> None:
    _refused(snapshot, 'POST', f'/{ALPHA_ITEMS}/_search', {'query': {'match_all': {}}})


def test_shape02_mget_dict_body_through_the_real_client_refused(snapshot) -> None:
    """opensearch-py passes ``mget``'s body to the transport as a dict (only
    bulk/msearch are pre-serialized), so this goes through a real
    ``AsyncOpenSearch.mget`` -- the exact path ``mget_crops`` takes."""
    sent: list[tuple[str, str, Any]] = []

    class _Bottom:
        async def perform_request(self, method, url, params=None, body=None, **_kw):  # noqa: ARG002
            sent.append((method, url, body))
            return {'docs': []}

        async def close(self) -> None:
            return None

    class _Registry:
        def snapshot(self):
            return snapshot

    client = AsyncOpenSearch(hosts=['http://127.0.0.1:9'])
    client.transport = _Bottom()  # type: ignore[assignment]
    install_project_guard(client, _Registry())

    async def _run() -> None:
        with bind_project(snapshot['beta']):
            with pytest.raises(CrossProjectAccess):
                await client.mget(body={'docs': [{'_index': ALPHA_ITEMS, '_id': 'a-1'}]})
            await client.mget(body={'docs': [{'_index': BETA_ITEMS, '_id': 'b-1'}]})

    asyncio.run(_run())
    assert [url for _m, url, _b in sent] == ['/_mget'], (
        'only the own-project mget reached OpenSearch'
    )


def test_shape03_indexless_search_refused(snapshot) -> None:
    _refused(snapshot, 'POST', '/_search', {'query': {'match_all': {}}})


def test_shape04_indexless_count_refused(snapshot) -> None:
    _refused(snapshot, 'POST', '/_count', {'query': {'match_all': {}}})


def test_shape05_msearch_header_without_index_refused(snapshot) -> None:
    _refused(snapshot, 'POST', '/_msearch', '{}\n{"query": {"match_all": {}}}\n')


def test_shape06_prefix_wildcard_refused(snapshot) -> None:
    _refused(snapshot, 'POST', '/op_prj_*/_search', {'query': {'match_all': {}}})


def test_shape07_other_project_wildcard_refused(snapshot) -> None:
    _refused(snapshot, 'POST', '/op_prj_alpha__*/_search', {'query': {'match_all': {}}})


def test_shape08_reindex_refused(snapshot) -> None:
    _refused(
        snapshot,
        'POST',
        '/_reindex',
        {'source': {'index': ALPHA_ITEMS}, 'dest': {'index': BETA_ITEMS}},
    )


def test_shape09_aliases_refused(snapshot) -> None:
    _refused(
        snapshot,
        'POST',
        '/_aliases',
        {'actions': [{'add': {'index': ALPHA_ITEMS, 'alias': 'beta_view'}}]},
    )


def test_shape10_sql_refused(snapshot) -> None:
    _refused(snapshot, 'POST', '/_plugins/_sql', {'query': f'SELECT * FROM {ALPHA_ITEMS}'})


def test_shape11_url_encoded_index_name_refused(snapshot) -> None:
    _refused(snapshot, 'POST', '/op%5Fprj%5Falpha__items/_search', {'query': {'match_all': {}}})


@pytest.mark.parametrize('indices', [[ALPHA_ITEMS], ALPHA_ITEMS])
def test_shape12_msearch_header_indices_key_refused(snapshot, indices: Any) -> None:
    """A header's ``indices`` overrides the URL index just like ``index``
    (re-review R1): an msearch on beta's URL must not read alpha."""
    body = json.dumps({'indices': indices}) + '\n{"query": {"match_all": {}}}\n'
    _refused(snapshot, 'POST', f'/{BETA_ITEMS}/_msearch', body)


def test_shape13_msearch_unknown_header_key_refused(snapshot) -> None:
    body = json.dumps({'index': BETA_ITEMS, 'expand_wildcards': 'all'}) + '\n{}\n'
    _refused(snapshot, 'POST', f'/{BETA_ITEMS}/_msearch', body)


# --- Other shapes the allowlist must keep refusing ---------------------------


@pytest.mark.parametrize(
    ('method', 'url', 'body'),
    [
        ('POST', '/_all/_search', None),
        ('POST', f'/{BETA_ITEMS},-{BETA_ITEMS}/_search', None),
        ('POST', f'/{BETA_ITEMS}?/_search', None),
        ('POST', f'/remote:{ALPHA_ITEMS}/_search', None),
        ('GET', '/_mget', {'docs': [{'_id': 'x'}]}),
        ('POST', '/_bulk', '{"index": {"_id": "x"}}\n{"a": 1}\n'),
        ('POST', f'/{BETA_ITEMS}/_clone/copy', None),
        ('POST', f'/{ALPHA_ITEMS}/_split/copy', None),
        ('PUT', '/_snapshot/repo/snap', {'indices': ALPHA_ITEMS}),
        ('POST', '/_plugins/_ppl', {'query': f'source={ALPHA_ITEMS}'}),
        ('PUT', f'/{BETA_ITEMS}/_alias/view', None),
        ('PUT', '/op_prj_beta__copy', {'aliases': {ALPHA_ITEMS: {}}}),
        ('GET', '/_cat/indices', None),
        ('POST', '/op_prj_gamma__items/_search', None),
        ('DELETE', '/_search/scroll/_all', None),
        ('GET', '/_tasks/node:1', None),
        ('POST', '/_search/scroll', {'scroll_id': 'unknown-scroll'}),
    ],
)
def test_unlisted_or_foreign_shape_refused(snapshot, method: str, url: str, body: Any) -> None:
    _refused(snapshot, method, url, body)


@pytest.mark.parametrize(
    'body',
    [
        # A terms lookup reads another index from inside a search body.
        {'query': {'terms': {'crop_id': {'index': ALPHA_ITEMS, 'id': 'a-1', 'path': 'ids'}}}},
        {'query': {'more_like_this': {'like': [{'_index': ALPHA_ITEMS, '_id': 'a-1'}]}}},
    ],
)
def test_cross_index_search_body_refused(snapshot, body: dict[str, Any]) -> None:
    _refused(snapshot, 'POST', f'/{BETA_ITEMS}/_search', body)
    _refused(snapshot, 'POST', f'/{BETA_ITEMS}/_search', json.dumps(body).encode())


def test_msearch_body_foreign_header_refused_even_after_own_header(snapshot) -> None:
    body = (
        f'{{"index": "{BETA_ITEMS}"}}\n{{"query": {{"match_all": {{}}}}}}\n'
        f'{{"index": "{ALPHA_ITEMS}"}}\n{{"query": {{"match_all": {{}}}}}}\n'
    )
    _refused(snapshot, 'POST', '/_msearch', body)


# --- The shapes the codebase sends stay allowed ------------------------------


@pytest.mark.parametrize(
    ('method', 'url', 'body'),
    [
        ('GET', f'/{BETA_ITEMS}/_doc/b-1', None),
        ('HEAD', f'/{BETA_ITEMS}', None),
        ('PUT', f'/{BETA_ITEMS}', {'mappings': {'properties': {'x': {'type': 'keyword'}}}}),
        (
            'PUT',
            f'/{BETA_ITEMS}/_mapping',
            {'properties': {'x': {'type': 'keyword', 'index': False}}},
        ),
        ('POST', f'/{BETA_ITEMS}/_search', {'query': {'term': {'crop_id': 'b-1'}}}),
        ('POST', f'/{BETA_ITEMS}/_count', {'query': {'match_all': {}}}),
        ('POST', f'/{BETA_ITEMS}/_update/b-1', {'doc': {'x': 1}}),
        ('POST', f'/{BETA_ITEMS}/_update_by_query', {'query': {'match_all': {}}}),
        ('POST', f'/{BETA_ITEMS}/_delete_by_query', {'query': {'match_all': {}}}),
        ('POST', f'/{BETA_ITEMS}/_refresh', None),
        ('POST', '/_mget', {'docs': [{'_index': BETA_ITEMS, '_id': 'b-1'}]}),
        ('POST', f'/{BETA_ITEMS}/_mget', {'ids': ['b-1']}),
        ('POST', '/_bulk', f'{{"index": {{"_index": "{BETA_ITEMS}", "_id": "x"}}}}\n{{}}\n'),
        ('POST', '/_msearch', f'{{"index": "{BETA_ITEMS}"}}\n{{"query": {{"match_all": {{}}}}}}\n'),
        ('GET', f'/_plugins/_knn/warmup/{BETA_ITEMS}', None),
    ],
)
def test_own_project_shapes_allowed(snapshot, method: str, url: str, body: Any) -> None:
    with bind_project(snapshot['beta']):
        check_request(method, url, body, snapshot)


def test_handles_are_owned_by_the_project_that_opened_them(snapshot) -> None:
    """Scroll, point-in-time and task ids handed out to alpha are useless
    to beta, and a point-in-time search reaches only its PIT."""
    handles = {'scroll-a': 'alpha', 'pit-a': 'alpha', 'node:1': 'alpha', 'pit-b': 'beta'}
    with bind_project(snapshot['beta']):
        for method, url, body in [
            ('POST', '/_search/scroll', {'scroll_id': 'scroll-a'}),
            ('DELETE', '/_search/scroll', {'scroll_id': ['scroll-a']}),
            ('POST', '/_search', {'pit': {'id': 'pit-a'}}),
            ('DELETE', '/_search/point_in_time', {'pit_id': ['pit-a']}),
            ('GET', '/_tasks/node:1', None),
            (
                'POST',
                '/_search',
                {'pit': {'id': 'pit-b'}, 'query': {'terms': {'x': {'index': ALPHA_ITEMS}}}},
            ),
        ]:
            with pytest.raises(CrossProjectAccess):
                check_request(method, url, body, snapshot, handles=handles)
        check_request('POST', '/_search', {'pit': {'id': 'pit-b'}}, snapshot, handles=handles)
        check_request('POST', f'/{BETA_ITEMS}/_search/point_in_time', None, snapshot)


def test_registry_index_is_writable_only_by_lifecycle_code(snapshot) -> None:
    from src.services.projects.guard import bind_registry_admin

    with bind_project(snapshot['beta']):
        check_request('GET', '/op_projects/_doc/project:beta', None, snapshot)
        with pytest.raises(CrossProjectAccess):
            check_request('PUT', '/op_projects/_doc/project:beta', {'status': 'active'}, snapshot)
        with bind_registry_admin():
            check_request('PUT', '/op_projects/_doc/project:beta', {'status': 'active'}, snapshot)
    with pytest.raises(CrossProjectAccess):
        check_request('GET', f'/_cat/indices/{ALPHA_ITEMS},{BETA_ITEMS}', None, snapshot)
    with bind_registry_admin():
        check_request('GET', f'/_cat/indices/{ALPHA_ITEMS},{BETA_ITEMS}', None, snapshot)
