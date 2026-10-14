"""Two queue rows for one key (a stale retried row and the replaced object's new row)
are fetched as the same current object, so the route answers once; both rows settle."""

from __future__ import annotations

import httpx

from src.services.curation.bulk_ingest.sender import ObjectBatchSender


def _sender() -> ObjectBatchSender:
    return ObjectBatchSender(httpx.AsyncClient(), api_prefix='/curation', project='p')


def test_two_rows_for_one_key_both_get_the_answer() -> None:
    old = 's3://b/k.jpg#etag=old'
    new = 's3://b/k.jpg#etag=new'
    body = {
        'results': [{'status': 'success', 'image_path': new, 'source_identifier': new}],
    }
    out = _sender()._answer(body, [old, new])
    got = {r['image_path']: r['status'] for r in out['results']}
    assert got == {old: 'success', new: 'success'}


def test_distinct_versions_keep_their_own_answers() -> None:
    v1 = 's3://b/k.jpg#etag=e1&version=v1'
    v2 = 's3://b/k.jpg#etag=e2&version=v2'
    body = {
        'results': [
            {'status': 'success', 'image_path': v1, 'source_identifier': v1},
            {'status': 'failed', 'image_path': v2, 'source_identifier': v2},
        ],
    }
    out = _sender()._answer(body, [v1, v2])
    got = {r['image_path']: r['status'] for r in out['results']}
    assert got == {v1: 'success', v2: 'failed'}


def test_single_row_with_changed_etag_maps_back() -> None:
    sent = 's3://b/k.jpg#etag=old'
    actual = 's3://b/k.jpg#etag=new'
    body = {'results': [{'status': 'success', 'image_path': actual, 'source_identifier': actual}]}
    out = _sender()._answer(body, [sent])
    assert [r['image_path'] for r in out['results']] == [sent]
