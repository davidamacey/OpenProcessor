"""Bulk optimistic concurrency of ``review_resolve`` and the region ``edit_undo`` batch, on REAL OpenSearch.

Both routes write through ``occ_bulk_rounds``: one ``_mget`` and one conditional ``_bulk`` per page
(``if_seq_no`` / ``if_primary_term``), a 409 item retried, and an item that keeps conflicting
reported per item while the rest commit. OpenSearch semantics:
https://opensearch.org/docs/latest/api-reference/document-apis/index-document/ (optimistic
concurrency control with ``if_seq_no`` and ``if_primary_term``).

A route cannot be interrupted between its read and its write from outside the API process, so the
interference runs in the ``api`` container: the real service function (the one the route calls)
runs over a guarded client whose ``bulk`` first lets a second writer touch chosen documents, i.e.
bumps their ``_seq_no`` after the page was read. The route-level cases (100 crops, 501 crops ->
202 + job) go through HTTP.
"""

from __future__ import annotations

import json
import subprocess  # nosec B404 - the harness's own compose project
import uuid
from typing import Any

import pytest

from tests.live.conftest import (
    COMPOSE_FILE,
    COMPOSE_PROJECT,
    INDEXES,
    REPO_ROOT,
    get_doc,
    refresh,
    search,
    wait_job,
)


pytestmark = pytest.mark.live

# Runs in the api container. Reads one JSON request on stdin, binds the default project, wraps the
# client's bulk so the "other writer" acts between the page's read and its write, then calls the
# service function and prints its outcome as JSON.
_SCRIPT = r"""
import asyncio, json, sys
from src.config.settings import get_settings
from src.config.curation_accessors import items_index
from src.services.projects.script_binding import abind_script_project
from src.services.projects.guard import make_script_opensearch

req = json.loads(sys.stdin.read())

async def main():
    await abind_script_project('default')
    client = make_script_opensearch([get_settings().opensearch_url])
    real_bulk = client.bulk
    bulks = {'n': 0}

    async def bulk(*args, **kwargs):
        bulks['n'] += 1
        for doc_id, how in req['interfere'].items():
            if how == 'always' or (how in ('once', 'human') and bulks['n'] == 1):
                doc = {'updated_at': '2030-01-01T00:00:%02dZ' % bulks['n']}
                if how == 'human':
                    doc.update(class_validated=True, class_source='human', class_id=req['human_class_id'],
                               needs_new_class=False)
                await client.update(index=items_index(), id=doc_id, body={'doc': doc})
        return await real_bulk(*args, **kwargs)

    client.bulk = bulk
    try:
        if req['op'] == 'resolve':
            from src.services.curation.new_class_resolve import resolve_pending_proposals
            out = await resolve_pending_proposals(client, req['ids'], label=req['label'],
                class_id=req['class_id'], class_name=req['class_name'], label_source='new_class_proposal')
            result = {'updated': sorted(out.updated_ids), 'skipped': sorted(out.skipped),
                      'conflicts': out.conflicts}
        else:
            from src.services.curation.edit_history import EditKind, REGION_UNDO_WRITER
            from src.services.curation.edit_undo import undo_edit_writes
            by = await undo_edit_writes(client, req['ids'], EditKind.REGION, REGION_UNDO_WRITER)
            result = {k: sorted(v) for k, v in by.items()}
        result['bulk_calls'] = bulks['n']
        print(json.dumps(result))
    finally:
        await client.close()

asyncio.run(main())
"""


def _run_in_api(request: dict[str, Any]) -> dict[str, Any]:
    done = subprocess.run(  # nosec B603 - fixed argv, no shell
        ['docker', 'compose', '-p', COMPOSE_PROJECT, '-f', str(COMPOSE_FILE), 'exec', '-T',
         'api', 'python', '-c', _SCRIPT],
        input=json.dumps(request), capture_output=True, text=True, timeout=240, cwd=str(REPO_ROOT),
        check=False,
    )  # fmt: skip
    assert done.returncode == 0, done.stderr[-2500:]
    return json.loads(done.stdout.strip().splitlines()[-1])


def _clone_items(opensearch: Any, count: int) -> list[str]:
    """``count`` extra items (copies of one seeded doc) so a batch can pass the sync limit."""
    source = _source(opensearch, 'cls0_0000')
    ids = [f'occ_extra_{i:04d}' for i in range(count)]
    lines: list[str] = []
    for doc_id in ids:
        lines += [
            json.dumps({'index': {'_index': INDEXES['items'], '_id': doc_id}}),
            json.dumps({**source, 'crop_id': doc_id}),
        ]
    resp = opensearch.post(
        '/_bulk', content='\n'.join(lines) + '\n', params={'refresh': 'true'},
        headers={'Content-Type': 'application/x-ndjson'},
    )  # fmt: skip
    resp.raise_for_status()
    assert resp.json()['errors'] is False, resp.text[:500]
    return ids


def _crop_ids(opensearch: Any, count: int, *, skip: int = 0) -> list[str]:
    body = {'size': count + skip, '_source': False, 'query': {'bool': {'must_not': [{'prefix': {'crop_id': 'occ_extra_'}}]}},
            'sort': [{'crop_id': {'order': 'asc'}}]}  # fmt: skip
    hits = search(opensearch, INDEXES['items'], body)['hits']['hits']
    ids = [h['_id'] for h in hits][skip:]
    assert len(ids) == count, f'the seeded harness has only {len(ids)} items'
    return ids


def _make_pending(opensearch: Any, ids: list[str], label: str) -> None:
    """Turn ``ids`` into open new-class proposals for ``label`` (one real update-by-query)."""
    body = {
        'query': {'ids': {'values': ids}},
        'script': {
            'lang': 'painless',
            'source': (
                'ctx._source.class_id = null; ctx._source.class_name = null;'
                "ctx._source.class_source = 'vlm_new_class_pending';"
                'ctx._source.vlm_proposed_class = params.label; ctx._source.needs_new_class = true;'
                'ctx._source.class_validated = false; ctx._source.class_excluded = false;'
                "ctx._source.test_holdout = false; ctx._source.remove('review_dismissed_at')"
            ),
            'params': {'label': label},
        },
    }
    resp = opensearch.post(
        f'/{INDEXES["items"]}/_update_by_query', params={'refresh': 'true', 'conflicts': 'proceed'},
        json=body,
    )  # fmt: skip
    resp.raise_for_status()
    assert resp.json()['updated'] == len(ids), resp.text


def _source(opensearch: Any, crop_id: str) -> dict[str, Any]:
    return get_doc(opensearch, INDEXES['items'], crop_id)['_source']


@pytest.fixture
def label() -> str:
    return f'occ probe {uuid.uuid4().hex[:8]}'


def test_resolve_100_crops_over_http_updates_all(
    api_client: Any, opensearch: Any, class_ids: dict[str, int], label: str
) -> None:
    ids = _crop_ids(opensearch, 100)
    _make_pending(opensearch, ids, label)
    resp = api_client.post(
        '/review/new_class_proposals/resolve', json={'label': label, 'class_id': class_ids['box']}
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert (body['matched'], body['updated'], body['conflicts'], body['skipped']) == (
        100,
        100,
        [],
        [],
    )
    refresh(opensearch, INDEXES['items'])
    assert all(_source(opensearch, i)['class_source'] == 'human' for i in ids)


def test_resolve_501_crops_is_queued_and_the_job_completes(
    api_client: Any, opensearch: Any, class_ids: dict[str, int], label: str
) -> None:
    clones = _clone_items(opensearch, 120)
    try:
        ids = [*_crop_ids(opensearch, 381), *clones]
        assert len(set(ids)) == 501
        _make_pending(opensearch, ids, label)
        resp = api_client.post(
            '/review/new_class_proposals/resolve',
            json={'label': label, 'class_id': class_ids['box']},
        )
        assert resp.status_code == 202, resp.text
        job = wait_job(api_client, resp.json()['job_id'], timeout=240)
        assert job['status'] == 'completed', job
        assert job['result']['updated'] == 501, job['result']
    finally:
        opensearch.post(
            f'/{INDEXES["items"]}/_delete_by_query',
            params={'refresh': 'true'},
            json={'query': {'prefix': {'crop_id': 'occ_extra_'}}},
        ).raise_for_status()


def test_resolve_a_document_that_keeps_conflicting_is_a_per_item_conflict(
    opensearch: Any, class_ids: dict[str, int], label: str
) -> None:
    ids = _crop_ids(opensearch, 100)
    _make_pending(opensearch, ids, label)
    victim = ids[40]
    out = _run_in_api({
        'op': 'resolve', 'ids': ids, 'label': label, 'class_id': class_ids['box'],
        'class_name': 'box', 'interfere': {victim: 'always'},
    })  # fmt: skip
    assert out['conflicts'] == [{'crop_id': victim, 'current_source': 'vlm_new_class_pending'}], out
    assert out['updated'] == sorted(set(ids) - {victim}), out
    assert out['skipped'] == []
    assert _source(opensearch, victim)['class_source'] == 'vlm_new_class_pending'
    assert all(_source(opensearch, i)['class_source'] == 'human' for i in ids if i != victim)


def test_resolve_retries_a_single_conflict_and_commits_it(
    opensearch: Any, class_ids: dict[str, int], label: str
) -> None:
    ids = _crop_ids(opensearch, 20)
    _make_pending(opensearch, ids, label)
    out = _run_in_api({
        'op': 'resolve', 'ids': ids, 'label': label, 'class_id': class_ids['box'],
        'class_name': 'box', 'interfere': {ids[3]: 'once'},
    })  # fmt: skip
    assert out['conflicts'] == [], out
    assert out['skipped'] == [], out
    assert out['updated'] == sorted(ids), out
    assert out['bulk_calls'] == 2, 'one write, then the retry of the conflicted item'


def test_the_lock_rule_a_human_write_mid_batch_is_never_overwritten(
    opensearch: Any, class_ids: dict[str, int], label: str
) -> None:
    ids = _crop_ids(opensearch, 20)
    _make_pending(opensearch, ids, label)
    victim = ids[7]
    human_class = class_ids['envelope']
    out = _run_in_api({
        'op': 'resolve', 'ids': ids, 'label': label, 'class_id': class_ids['box'],
        'class_name': 'box', 'interfere': {victim: 'human'}, 'human_class_id': human_class,
    })  # fmt: skip
    assert out['skipped'] == [victim], out
    assert out['conflicts'] == []
    assert out['updated'] == sorted(set(ids) - {victim})
    survivor = _source(opensearch, victim)
    assert (survivor['class_id'], survivor['class_source']) == (human_class, 'human')
    assert survivor['class_validated'] is True


def _region_edited(api_client: Any, ids: list[str]) -> None:
    """A real bulk region edit, so each crop carries an undoable ``edit_history`` entry."""
    resp = api_client.put('/crops/batch_regions', json={'crop_ids': ids, 'boxes': []})
    assert resp.status_code == 200, resp.text
    assert resp.json()['updated'] == len(ids), resp.text


def test_edit_undo_batch_over_http_restores_all(api_client: Any, opensearch: Any) -> None:
    ids = _crop_ids(opensearch, 100, skip=200)
    _region_edited(api_client, ids)
    resp = api_client.post('/crops/region/undo_batch', json={'crop_ids': ids})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert (body['undone'], body['conflicts'], body['not_found']) == (100, [], [])


def test_edit_undo_a_document_that_keeps_conflicting_is_a_per_item_conflict(
    api_client: Any, opensearch: Any
) -> None:
    ids = _crop_ids(opensearch, 50, skip=320)
    _region_edited(api_client, ids)
    victim = ids[10]
    out = _run_in_api({'op': 'undo', 'ids': ids, 'interfere': {victim: 'always'}})
    assert out['conflicts'] == [victim], out
    assert out['undone'] == sorted(set(ids) - {victim}), out
    assert out['nothing_to_undo'] == []
    assert out['not_found'] == []
    refresh(opensearch, INDEXES['items'])
    # the conflicted crop still has its un-undone edit: a later undo restores it
    again = _run_in_api({'op': 'undo', 'ids': [victim], 'interfere': {}})
    assert again['undone'] == [victim], again
