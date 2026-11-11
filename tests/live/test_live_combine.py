"""Live: combining two projects into a new one (``/projects/combine*``).

Two throwaway projects each ingest two pictures (one of them the same bytes in both), so the
content-hash dedup has something to merge. The test previews, starts with the preview's sha,
follows the job, and asserts what the target's stores hold. The target and the sources are
deleted afterwards.
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest

from ._ingest_support import count, ingest_project, picture
from .conftest import API_PREFIX, JOB_TERMINAL, wait_for_state, wait_until


pytestmark = pytest.mark.live

SHARED, ONLY_A, ONLY_B = (200, 30, 30), (30, 200, 30), (30, 30, 200)


def _upload(api: Any, prefix: str, pictures: dict[str, tuple[int, int, int]]) -> None:
    resp = api.post(
        f'{prefix}/ingest/upload',
        files=[('images', (name, picture(rgb), 'image/jpeg')) for name, rgb in pictures.items()],
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()['summary']['successful'] == len(pictures), resp.text


def test_combine_merges_two_projects_and_dedups_by_content(api: Any, opensearch: Any) -> None:
    target = f'combined{uuid.uuid4().hex[:6]}'
    with (
        ingest_project(api, opensearch, 'comba') as a,
        ingest_project(api, opensearch, 'combb') as b,
    ):
        _upload(api, a.prefix, {'shared.jpg': SHARED, 'only_a.jpg': ONLY_A})
        _upload(api, b.prefix, {'shared-b.jpg': SHARED, 'only_b.jpg': ONLY_B})
        request = {
            'target': {'slug': target, 'display_name': 'Combined live'},
            'sources': [{'project': a.slug}, {'project': b.slug}],
            'dedup': 'content_hash',
        }

        preview = api.post(f'{API_PREFIX}/projects/combine/preview', json=request)
        assert preview.status_code == 200, preview.text
        report = preview.json()
        assert report['ok'], report['errors']
        assert [s['project'] for s in report['sources']] == [a.slug, b.slug], report['sources']

        refused = api.post(
            f'{API_PREFIX}/projects/combine', json={**request, 'expected_preview_sha': 'stale'}
        )
        assert refused.status_code in (409, 422), refused.text

        started = api.post(
            f'{API_PREFIX}/projects/combine',
            json={**request, 'expected_preview_sha': report['preview_sha']},
        )
        assert started.status_code == 202, started.text
        assert started.json()['target'] == target
        job_id = started.json()['job_id']

        def fetch() -> dict[str, Any]:
            resp = api.get(f'{API_PREFIX}/projects/combine/{job_id}')
            assert resp.status_code == 200, resp.text
            return resp.json()

        job = wait_for_state(fetch, lambda j: j['status'] in JOB_TERMINAL, timeout=240)
        try:
            assert job['status'] == 'completed', job
            assert job['sources'] == [a.slug, b.slug]
            listed = api.get(f'{API_PREFIX}/projects').json()['projects']
            assert next(p for p in listed if p['slug'] == target)['status'] == 'active'

            images = f'verify_prj_{target}__images'
            # Four pictures went in, one is the same bytes twice: three images in the target,
            # and the source projects are untouched.
            assert count(opensearch, images) == 3
            assert count(opensearch, a.images_index) == 2
            assert count(opensearch, b.images_index) == 2
            assert count(opensearch, f'verify_prj_{target}__items') >= 3
        finally:
            deleted = api.delete(f'{API_PREFIX}/projects/{target}', params={'confirm': target})
            assert deleted.status_code == 202, deleted.text
            assert wait_until(
                lambda: api.get(f'{API_PREFIX}/projects/{target}').status_code == 404, timeout=120
            )


def test_combine_preview_refuses_an_unknown_source(api: Any) -> None:
    resp = api.post(
        f'{API_PREFIX}/projects/combine/preview',
        json={
            'target': {'slug': f'never{uuid.uuid4().hex[:6]}', 'display_name': 'Never'},
            'sources': [{'project': 'no-such-project'}],
        },
    )
    if resp.status_code == 200:
        assert resp.json()['ok'] is False, resp.text
        assert resp.json()['errors'], resp.text
    else:
        assert resp.status_code in (404, 422), resp.text
