"""Live: the ingest routes against a real OpenSearch, with the harness's fake Triton.

``docker/test/fake_triton.py`` serves the fused detector (one box per picture) and the PE image
encoder, so every route runs its whole path: decode, detect, embed, dedup by content hash, write
the images and items documents and report counts. A fresh project has no region profile, which is
what a user's first upload meets. The routes covered:

* ``POST /ingest/upload`` (multipart bytes, content dedup, an unsupported type),
* ``POST /ingest/batch`` and ``/ingest/image`` (server paths under the source root and under a
  named alias; a path outside every root is refused per item),
* ``POST /ingest/bulk`` over a folder holding a dangling symlink (reported, not a clean run),
  with the job's progress on the project's event stream.

Assertions read the stores (the images and items indexes), not only the response envelopes.
"""

from __future__ import annotations

import json
import uuid
from typing import TYPE_CHECKING, Any

import pytest

from ._ingest_support import IngestProject, count, ingest_project, picture
from ._sse import SseCollector
from .conftest import (
    API_PREFIX,
    API_URL,
    VERIFY_DATA_DIR,
    create_project,
    wait_for_state,
    wait_until,
)


if TYPE_CHECKING:
    from collections.abc import Iterator


pytestmark = pytest.mark.live

RED, GREEN, BLUE, YELLOW = (200, 30, 30), (30, 200, 30), (30, 30, 200), (220, 220, 30)
BULK_TERMINAL = ('completed', 'completed_with_errors', 'failed', 'cancelled', 'interrupted')


@pytest.fixture(scope='module')
def project(api: Any, opensearch: Any) -> Iterator[IngestProject]:
    with ingest_project(api, opensearch, 'ingest') as handle:
        yield handle


def _images(opensearch: Any, project: IngestProject) -> dict[str, dict[str, Any]]:
    count(opensearch, project.images_index)
    hits = opensearch.post(
        f'/{project.images_index}/_search', json={'size': 100, 'query': {'match_all': {}}}
    ).json()['hits']['hits']
    return {h['_id']: h['_source'] for h in hits}


def test_upload_stores_images_and_items_and_dedups_by_content(
    api: Any, opensearch: Any, project: IngestProject
) -> None:
    files = [
        ('images', ('red.jpg', picture(RED), 'image/jpeg')),
        ('images', ('green.jpg', picture(GREEN), 'image/jpeg')),
    ]
    resp = api.post(
        f'{project.prefix}/ingest/upload',
        files=files,
        data={'image_paths': json.dumps(['phone/red.jpg', 'phone/green.jpg']), 'run_id': 'up-1'},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body['summary']['successful'] == 2, body
    assert [r['status'] for r in body['results']] == ['success', 'success']
    assert all(r['n_embedded'] >= 1 for r in body['results']), body

    stored = _images(opensearch, project)
    assert len(stored) == 2
    assert sorted(doc['source_identifier'] for doc in stored.values()) == [
        'phone/green.jpg',
        'phone/red.jpg',
    ]
    assert all(doc['ingest_run_id'] == 'up-1' for doc in stored.values())
    assert count(opensearch, project.items_index) == body['summary']['crops_indexed'] >= 2

    status = api.get(f'{project.prefix}/ingest/status', params={'run_id': 'up-1'}).json()
    assert status['total'] == 2, status

    # The same bytes again are a duplicate: nothing new is stored.
    again = api.post(
        f'{project.prefix}/ingest/upload',
        files=[('images', ('red-copy.jpg', picture(RED), 'image/jpeg'))],
    )
    assert again.status_code == 200, again.text
    assert again.json()['results'][0]['status'] == 'duplicate', again.text
    assert len(_images(opensearch, project)) == 2


def test_upload_reports_an_unsupported_file_per_item(
    api: Any, opensearch: Any, project: IngestProject
) -> None:
    before = count(opensearch, project.images_index)
    resp = api.post(
        f'{project.prefix}/ingest/upload',
        files=[
            ('images', ('blue.jpg', picture(BLUE), 'image/jpeg')),
            ('images', ('notes.txt', b'not a picture', 'text/plain')),
        ],
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body['summary']['successful'] == 1, body
    assert body['summary']['failed'] == 1, body
    failed = next(r for r in body['results'] if r['status'] == 'failed')
    assert failed['error_kind'], failed
    assert count(opensearch, project.images_index) == before + 1


def _write(path: Any, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def test_server_paths_under_the_source_root_and_an_alias_are_ingested(
    api: Any, opensearch: Any, project: IngestProject
) -> None:
    primary = project.host_dir / 'server' / 'yellow.jpg'
    alias_dir = VERIFY_DATA_DIR / 'archive' / project.slug
    alias = alias_dir / 'cyan.jpg'
    _write(primary, picture(YELLOW))
    _write(alias, picture((30, 200, 200)))
    try:
        before = count(opensearch, project.images_index)
        resp = api.post(
            f'{project.prefix}/ingest/batch',
            json={
                'items': [
                    {'path': f'{project.container_dir}/server/yellow.jpg', 'source': 'server'},
                    {'path': f'/verify-data/archive/{project.slug}/cyan.jpg', 'source': 'server'},
                    {'path': '/etc/hostname', 'source': 'server'},
                    {'path': f'{project.container_dir}/server/missing.jpg', 'source': 'server'},
                ]
            },
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        statuses = {r['image_path']: r for r in body['results']}
        assert statuses[f'{project.container_dir}/server/yellow.jpg']['status'] == 'success', body
        assert statuses[f'/verify-data/archive/{project.slug}/cyan.jpg']['status'] == 'success'
        # Outside every configured root, and absent: refused per item, the rest still stored.
        assert statuses['/etc/hostname']['status'] == 'failed', body
        assert statuses[f'{project.container_dir}/server/missing.jpg']['status'] == 'failed', body
        assert body['status'] == 'partial', body
        assert count(opensearch, project.images_index) == before + 2

        one = api.post(
            f'{project.prefix}/ingest/image',
            json={'path': f'{project.container_dir}/server/yellow.jpg'},
        )
        assert one.status_code == 200, one.text
        assert one.json()['status'] == 'duplicate', one.text
    finally:
        alias.unlink(missing_ok=True)
        alias_dir.rmdir()


def test_bulk_folder_reports_an_unreadable_entry_and_streams_progress(
    api: Any, opensearch: Any, project: IngestProject
) -> None:
    folder = project.host_dir / 'bulk'
    _write(folder / 'one.jpg', picture((90, 40, 160)))
    _write(folder / 'two.jpg', picture((160, 90, 40)))
    (folder / 'dangling.jpg').symlink_to('does-not-exist.jpg')
    before = count(opensearch, project.images_index)

    events = SseCollector(f'{API_URL}{project.prefix}/events').start()
    try:
        started = api.post(
            f'{project.prefix}/ingest/bulk',
            json={'folder': f'{project.container_dir}/bulk', 'source': 'bulk_live'},
        )
        assert started.status_code == 202, started.text
        run_id = started.json()['run_id']

        def fetch() -> dict[str, Any]:
            return api.get(f'{project.prefix}/ingest/bulk/{run_id}').json()

        run = wait_for_state(fetch, lambda r: r['status'] in BULK_TERMINAL, timeout=180)
        progress = run['progress']
        # Two readable pictures are stored; the dangling link is counted, named, and the run
        # does not end as a clean completion.
        assert run['status'] == 'completed_with_errors', run
        assert progress['ingested'] == 2, progress
        assert progress['discovery_complete'], progress
        assert progress['unreadable'] == 1, progress
        assert progress['unreadable_samples'] == ['dangling.jpg'], progress
        assert count(opensearch, project.images_index) == before + 2
        sources = {d['source'] for d in _images(opensearch, project).values()}
        assert 'bulk_live' in sources

        assert wait_until(
            lambda: any(
                e.get('event') == 'job.state' and e['data'].get('status') in BULK_TERMINAL
                for e in events.for_job(run_id)
            ),
            timeout=30,
        ), events.for_job(run_id)
        assert any(e.get('event') == 'job.progress' for e in events.for_job(run_id)) or any(
            e.get('event') == 'job.state' and e['data'].get('status') == 'running'
            for e in events.for_job(run_id)
        ), 'the stream carried no progress for the run'
    finally:
        events.close()


def test_bulk_refuses_a_folder_outside_the_source_roots(api: Any, project: IngestProject) -> None:
    resp = api.post(f'{project.prefix}/ingest/bulk', json={'folder': '/etc'})
    assert resp.status_code == 422, resp.text
    assert resp.json()['detail']['error'] == 'bulk_ingest_invalid_source', resp.text


def test_a_project_with_no_detector_refuses_ingest_and_stores_nothing(
    api: Any, opensearch: Any
) -> None:
    """No active ingest profile and no deployment detector: ingest fails closed (503)."""
    slug = f'nodet{uuid.uuid4().hex[:6]}'
    create_project(api, slug, 'No detector live')
    prefix = f'{API_PREFIX}/projects/{slug}'
    try:
        resp = api.post(
            f'{prefix}/ingest/upload',
            files=[('images', ('red.jpg', picture(RED), 'image/jpeg'))],
        )
        assert resp.status_code == 503, resp.text
        assert 'No detector configured' in resp.json()['detail'], resp.text
        assert count(opensearch, f'verify_prj_{slug}__images') == 0
    finally:
        deleted = api.delete(f'{API_PREFIX}/projects/{slug}', params={'confirm': slug})
        assert deleted.status_code == 202, deleted.text
