"""``check_request`` decision branches, one case per allow/refuse outcome.

Pins the exact verdict and refusal reason of every branch of the guard's
request classifier, so splitting or decomposing it cannot silently widen or
narrow what a bound project may reach.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from src.config.curation import base_curation_config
from src.config.project_context import bind_project
from src.config.projects import ProjectRecord, resources_for_new
from src.services.projects.guard import CrossProjectAccess, bind_registry_admin, check_request


pytestmark = pytest.mark.unbound

OWN = '/op_prj_alpha__items'
OTHER = '/op_prj_beta__items'


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


def _ok(snapshot, method, url, body=None, **kw) -> None:
    with bind_project(snapshot['alpha']):
        check_request(method, url, body, snapshot, **kw)


def _refused(snapshot, match, method, url, body=None, **kw) -> None:
    with bind_project(snapshot['alpha']), pytest.raises(CrossProjectAccess, match=match):
        check_request(method, url, body, snapshot, **kw)


def test_method_is_case_insensitive(snapshot) -> None:
    _ok(snapshot, 'get', f'{OWN}/_doc/x')


# --- /{index}/... -----------------------------------------------------------


def test_point_in_time_open_on_own_index_passes(snapshot) -> None:
    _ok(snapshot, 'POST', f'{OWN}/_search/point_in_time?keep_alive=1m')


def test_point_in_time_open_on_other_index_refused(snapshot) -> None:
    _refused(snapshot, 'belongs to project', 'POST', f'{OTHER}/_search/point_in_time')


def test_point_in_time_open_needs_post(snapshot) -> None:
    _refused(snapshot, 'not an allowed request shape', 'GET', f'{OWN}/_search/point_in_time')


def test_unknown_index_action_refused_with_shape_reason(snapshot) -> None:
    _refused(
        snapshot, r'GET /op_prj_alpha__items/_reindex is not an allowed', 'GET', f'{OWN}/_reindex'
    )


def test_wrong_method_for_known_action_refused(snapshot) -> None:
    _refused(
        snapshot,
        'DELETE /op_prj_alpha__items/_search is not an allowed',
        'DELETE',
        f'{OWN}/_search',
    )


def test_too_many_segments_refused(snapshot) -> None:
    _refused(snapshot, 'is not an allowed request shape', 'GET', f'{OWN}/_doc/x/extra')


def test_doc_with_id_on_own_index_passes_and_other_refused(snapshot) -> None:
    _ok(snapshot, 'PUT', f'{OWN}/_doc/x', {'a': 1})
    _refused(snapshot, 'belongs to project', 'PUT', f'{OTHER}/_doc/x', {'a': 1})


def test_comma_separated_targets_each_checked(snapshot) -> None:
    _ok(snapshot, 'GET', f'{OWN},op_prj_alpha__images/_search')
    _refused(snapshot, 'belongs to project', 'GET', f'{OWN},op_prj_beta__items/_search')


def test_search_body_naming_another_index_refused(snapshot) -> None:
    body = {'query': {'terms': {'id': {'index': 'op_prj_beta__items', 'id': '1', 'path': 'x'}}}}
    _refused(snapshot, 'belongs to project', 'POST', f'{OWN}/_search', body)


def test_search_body_naming_own_index_passes(snapshot) -> None:
    body = {'query': {'terms': {'id': {'index': 'op_prj_alpha__items', 'id': '1', 'path': 'x'}}}}
    _ok(snapshot, 'POST', f'{OWN}/_search', body)


def test_search_body_not_json_refused(snapshot) -> None:
    _refused(snapshot, 'request body is not JSON', 'POST', f'{OWN}/_search', '{nope')


def test_document_write_body_is_not_scanned_for_index_keys(snapshot) -> None:
    _ok(snapshot, 'PUT', f'{OWN}/_doc/x', {'index': 'op_prj_beta__items'})


def test_bulk_on_own_index_passes(snapshot) -> None:
    body = json.dumps({'index': {'_id': '1'}}) + '\n' + json.dumps({'a': 1}) + '\n'
    _ok(snapshot, 'POST', f'{OWN}/_bulk', body)


def test_bulk_action_overriding_index_to_other_project_refused(snapshot) -> None:
    body = json.dumps({'index': {'_index': 'op_prj_beta__items'}}) + '\n{"a": 1}\n'
    _refused(snapshot, 'belongs to project', 'POST', f'{OWN}/_bulk', body)


def test_msearch_header_unknown_key_refused(snapshot) -> None:
    body = json.dumps({'bogus': 1}) + '\n' + json.dumps({'query': {'match_all': {}}}) + '\n'
    _refused(snapshot, 'msearch header key', 'POST', f'{OWN}/_msearch', body)


def test_create_index_with_aliases_refused(snapshot) -> None:
    _refused(
        snapshot,
        'creating an index with aliases is never allowed',
        'PUT',
        OWN,
        {'aliases': {'a': {}}},
    )


def test_create_index_without_aliases_passes(snapshot) -> None:
    _ok(snapshot, 'PUT', OWN, {'settings': {}})
    _ok(snapshot, 'PUT', OWN, None)


def test_index_put_with_aliases_on_other_method_not_alias_checked(snapshot) -> None:
    _ok(snapshot, 'GET', OWN)


# --- cluster-level shapes ---------------------------------------------------


@pytest.mark.parametrize(
    ('method', 'url'),
    [
        ('GET', '/'),
        ('HEAD', '/'),
        ('GET', '/_cluster/health'),
        ('GET', '/_cluster/health?level=indices'),
        ('GET', '/_cluster/settings'),
        ('GET', '/_nodes/stats/jvm'),
    ],
)
def test_global_shapes_pass(snapshot, method, url) -> None:
    _ok(snapshot, method, url)


@pytest.mark.parametrize(
    ('method', 'url'),
    [('POST', '/'), ('PUT', '/_cluster/settings'), ('GET', '/_cluster/state'), ('GET', '/_all')],
)
def test_other_cluster_requests_refused(snapshot, method, url) -> None:
    _refused(snapshot, 'is not an allowed request shape', method, url)


# --- index-less multi-doc calls ---------------------------------------------


def test_multi_doc_bulk_with_per_line_index_passes(snapshot) -> None:
    body = json.dumps({'index': {'_index': 'op_prj_alpha__items'}}) + '\n{"a": 1}\n'
    _ok(snapshot, 'POST', '/_bulk', body)


def test_multi_doc_bulk_line_without_index_refused(snapshot) -> None:
    body = json.dumps({'index': {'_id': '1'}}) + '\n{"a": 1}\n'
    _refused(snapshot, 'a bulk action names no index', 'POST', '/_bulk', body)


def test_multi_doc_bulk_other_project_refused(snapshot) -> None:
    body = json.dumps({'delete': {'_index': 'op_prj_beta__items', '_id': '1'}}) + '\n'
    _refused(snapshot, 'belongs to project', 'POST', '/_bulk', body)


def test_multi_doc_bulk_wrong_method_refused(snapshot) -> None:
    _refused(snapshot, r'GET /_bulk is not an allowed request shape', 'GET', '/_bulk')


def test_multi_doc_mget_doc_without_index_refused(snapshot) -> None:
    _refused(snapshot, 'an mget doc names no index', 'POST', '/_mget', {'docs': [{'_id': '1'}]})


def test_multi_doc_mget_with_indexes_passes(snapshot) -> None:
    _ok(snapshot, 'POST', '/_mget', {'docs': [{'_index': 'op_prj_alpha__items', '_id': '1'}]})


def test_multi_doc_msearch_header_without_index_refused(snapshot) -> None:
    body = json.dumps({}) + '\n' + json.dumps({'query': {'match_all': {}}}) + '\n'
    _refused(snapshot, 'an msearch header names no index', 'POST', '/_msearch', body)


def test_multi_doc_with_extra_segment_falls_through_to_refusal(snapshot) -> None:
    _refused(snapshot, 'is not an allowed request shape', 'POST', '/_bulk/extra')


# --- index-less search (point in time) --------------------------------------


def test_indexless_search_without_pit_refused(snapshot) -> None:
    _refused(snapshot, 'a search must name its index', 'POST', '/_search', {'query': {}})


def test_indexless_search_pit_id_must_be_string(snapshot) -> None:
    _refused(snapshot, 'a search must name its index', 'POST', '/_search', {'pit': {'id': 5}})


def test_indexless_search_pit_may_not_name_indexes(snapshot) -> None:
    body = {'pit': {'id': 'p1'}, 'query': {'terms': {'x': {'index': 'op_prj_alpha__items'}}}}
    _refused(
        snapshot, 'a point-in-time search may not name other indexes', 'POST', '/_search', body
    )


def test_indexless_search_known_pit_handle_passes(snapshot) -> None:
    _ok(snapshot, 'POST', '/_search', {'pit': {'id': 'p1'}}, handles={'p1': 'alpha'})


def test_indexless_search_unknown_pit_handle_refused(snapshot) -> None:
    _refused(
        snapshot,
        'unknown scroll/task handle',
        'POST',
        '/_search',
        {'pit': {'id': 'p1'}},
        handles={},
    )
    _refused(snapshot, 'unknown scroll/task handle', 'POST', '/_search', {'pit': {'id': 'p1'}})


def test_indexless_search_pit_of_another_project_refused(snapshot) -> None:
    _refused(
        snapshot,
        'belongs to another project',
        'POST',
        '/_search',
        {'pit': {'id': 'p1'}},
        handles={'p1': 'beta'},
    )


def test_indexless_search_pit_unbound_global_handle(snapshot) -> None:
    check_request('GET', '/_search', {'pit': {'id': 'p1'}}, snapshot, handles={'p1': None})


def test_indexless_search_needs_get_or_post(snapshot) -> None:
    _refused(snapshot, 'is not an allowed request shape', 'DELETE', '/_search')


def test_delete_point_in_time_checks_ids(snapshot) -> None:
    _ok(snapshot, 'DELETE', '/_search/point_in_time', {'pit_id': 'p1'}, handles={'p1': 'alpha'})
    _ok(snapshot, 'DELETE', '/_search/point_in_time', {'pit_id': ['p1']}, handles={'p1': 'alpha'})
    _refused(
        snapshot,
        'belongs to another project',
        'DELETE',
        '/_search/point_in_time',
        {'pit_id': 'p1'},
        handles={'p1': 'beta'},
    )


def test_delete_point_in_time_without_id_refused(snapshot) -> None:
    _refused(
        snapshot,
        'must name the handle',
        'DELETE',
        '/_search/point_in_time',
        {},
        handles={'p1': 'alpha'},
    )


# --- scroll and task handles ------------------------------------------------


def test_scroll_id_from_url_body_and_params(snapshot) -> None:
    handles = {'s1': 'alpha', 's2': 'alpha'}
    _ok(snapshot, 'GET', '/_search/scroll/s1', handles=handles)
    _ok(snapshot, 'POST', '/_search/scroll', {'scroll_id': 's1'}, handles=handles)
    _ok(snapshot, 'POST', '/_search/scroll', {'scroll_id': ['s1', 's2']}, handles=handles)
    _ok(snapshot, 'POST', '/_search/scroll', {'scroll_id': b's1'}, handles=handles)
    _ok(snapshot, 'GET', '/_search/scroll', params={'scroll_id': 's2'}, handles=handles)
    _ok(snapshot, 'DELETE', '/_search/scroll/s1,s2', handles=handles)


def test_scroll_without_handle_refused(snapshot) -> None:
    _refused(
        snapshot, 'must name the handle', 'POST', '/_search/scroll', {}, handles={'s1': 'alpha'}
    )


def test_scroll_clear_all_refused(snapshot) -> None:
    _refused(
        snapshot,
        'clearing every scroll is never allowed',
        'DELETE',
        '/_search/scroll/_all',
        handles={'_all': 'alpha'},
    )


def test_scroll_unknown_and_foreign_handles_refused(snapshot) -> None:
    _refused(
        snapshot,
        'unknown scroll/task handle',
        'POST',
        '/_search/scroll',
        {'scroll_id': 'zz'},
        handles={},
    )
    _refused(
        snapshot,
        'belongs to another project',
        'POST',
        '/_search/scroll',
        {'scroll_id': 's1'},
        handles={'s1': 'beta'},
    )


def test_scroll_wrong_method_or_depth_refused(snapshot) -> None:
    handles = {'s1': 'alpha'}
    _refused(
        snapshot, 'is not an allowed request shape', 'PUT', '/_search/scroll/s1', handles=handles
    )
    _refused(
        snapshot,
        'is not an allowed request shape',
        'GET',
        '/_search/scroll/s1/extra',
        handles=handles,
    )


def test_task_lookup_checks_handle(snapshot) -> None:
    _ok(snapshot, 'GET', '/_tasks/t1', handles={'t1': 'alpha'})
    _refused(snapshot, 'belongs to another project', 'GET', '/_tasks/t1', handles={'t1': 'beta'})
    _refused(snapshot, 'unknown scroll/task handle', 'GET', '/_tasks/t1')


def test_task_lookup_other_shapes_refused(snapshot) -> None:
    _refused(
        snapshot, 'is not an allowed request shape', 'POST', '/_tasks/t1', handles={'t1': 'alpha'}
    )
    _refused(snapshot, 'is not an allowed request shape', 'GET', '/_tasks')
    _refused(
        snapshot, 'is not an allowed request shape', 'GET', '/_tasks/a/b', handles={'a': 'alpha'}
    )


# --- knn warmup and _cat/indices --------------------------------------------


def test_knn_warmup_own_passes_other_refused(snapshot) -> None:
    _ok(snapshot, 'GET', '/_plugins/_knn/warmup/op_prj_alpha__items')
    _ok(snapshot, 'GET', '/_plugins/_knn/warmup/op_prj_alpha__items,op_prj_alpha__images')
    _refused(snapshot, 'belongs to project', 'GET', '/_plugins/_knn/warmup/op_prj_beta__items')


def test_knn_warmup_other_shapes_refused(snapshot) -> None:
    _refused(
        snapshot,
        'is not an allowed request shape',
        'POST',
        '/_plugins/_knn/warmup/op_prj_alpha__items',
    )
    _refused(snapshot, 'is not an allowed request shape', 'GET', '/_plugins/_knn/warmup')


def test_cat_indices_needs_registry_admin(snapshot) -> None:
    _refused(
        snapshot,
        '_cat/indices is only for the global project list',
        'GET',
        '/_cat/indices/op_projects',
    )
    with bind_registry_admin():
        check_request('GET', '/_cat/indices/op_prj_alpha__items,op_projects', None, snapshot)


def test_cat_indices_names_must_be_concrete_even_for_admin(snapshot) -> None:
    with bind_registry_admin():
        with pytest.raises(CrossProjectAccess, match='not a concrete index name'):
            check_request('GET', '/_cat/indices/op_prj_*', None, snapshot)
        with pytest.raises(CrossProjectAccess, match='is not an allowed request shape'):
            check_request('GET', '/_cat/indices', None, snapshot)
        with pytest.raises(CrossProjectAccess, match='is not an allowed request shape'):
            check_request('POST', '/_cat/indices/x', None, snapshot)


def test_final_refusal_names_method_and_path_without_query(snapshot) -> None:
    _refused(
        snapshot,
        r'^project guard refused a request \(bound .alpha.\): GET /_sql is not',
        'GET',
        '/_sql?format=json',
    )
    _refused(snapshot, r'POST /_reindex is not an allowed request shape', 'POST', '/_reindex')


def test_empty_url_is_the_root_shape(snapshot) -> None:
    _ok(snapshot, 'GET', '')
    _refused(snapshot, 'is not an allowed request shape', 'POST', '')
