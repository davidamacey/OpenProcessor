"""The live tier's write path, against a throwaway project the test creates and deletes.

Opt-in: the `writes` marker is deselected by `e2e/pytest.ini`, so this only runs as
`e2e/.venv/bin/pytest -c e2e/pytest.ini e2e/live -m writes` with `OP_CROPWRIGHT_LIVE_URL` set.
It never touches an existing project: every write goes to `e2e-writes-<random>`, which is
deleted in teardown (a real delete is guarded by `?confirm=<slug>`).

The calls go through Playwright's `APIRequestContext` (https://playwright.dev/python/docs/api-testing),
which brings the multipart upload and the basic-auth headers; the image is a public COCO
photo (the on-demand smoke image `make download-test-images` fetches; COCO photos are never committed) the deployment detector finds objects in.
A fresh project has no region profile, so the fixture creates and activates one first, as a user
would (region writes answer 409 `no_active_profile` without it).
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from conftest import API_PREFIX, _auth_headers


pytestmark = pytest.mark.writes

SAMPLE_IMAGE = Path(__file__).resolve().parents[3] / 'test_images' / 'coco' / '000000001584.jpg'
POLL_TIMEOUT_S = 90.0
POLL_STEP_S = 0.5


def _smoke_image_bytes() -> bytes:
    if not SAMPLE_IMAGE.is_file():
        pytest.fail(f'{SAMPLE_IMAGE} is missing; fetch it with `make download-test-images`')
    return SAMPLE_IMAGE.read_bytes()


def _poll(what: str, probe: Callable[[], Any]) -> Any:
    """Bounded poll: the first truthy `probe()` result, else a failure naming `what`."""
    deadline = time.monotonic() + POLL_TIMEOUT_S
    while time.monotonic() < deadline:
        found = probe()
        if found:
            return found
        time.sleep(POLL_STEP_S)
    pytest.fail(f'timed out after {POLL_TIMEOUT_S:.0f}s waiting for {what}')


@pytest.fixture(scope='module')
def api(playwright: Any, live_url: str) -> Iterator[Any]:
    ctx = playwright.request.new_context(base_url=live_url, extra_http_headers=_auth_headers())
    yield ctx
    ctx.dispose()


@pytest.fixture(scope='module')
def throwaway_project(api: Any) -> Iterator[dict[str, str]]:
    slug = f'e2e-writes-{uuid.uuid4().hex[:8]}'
    created = api.post(
        f'{API_PREFIX}/projects',
        data={'slug': slug, 'display_name': 'Cropwright live writes test'},
    )
    assert created.status in (200, 202), created.text()
    project = created.json()['project']
    try:
        _poll(
            f'project {slug} to become active',
            lambda: api.get(f'{API_PREFIX}/projects/{slug}').json().get('status') == 'active',
        )
        yield {'slug': slug, 'prefix': project['prefix']}
    finally:
        deleted = api.delete(f'{API_PREFIX}/projects/{slug}?confirm={slug}')
        assert deleted.status in (200, 202), deleted.text()
        _poll(
            f'project {slug} to be gone',
            lambda: api.get(f'{API_PREFIX}/projects/{slug}').status == 404,
        )


@pytest.fixture(scope='module')
def active_region_profile(api: Any, throwaway_project: dict[str, str]) -> str:
    prefix = throwaway_project['prefix']
    name = 'e2e_writes_objects'
    created = api.post(
        f'{prefix}/region_profiles',
        data={
            'name': name,
            'body': {
                'detector_model': 'yolov11_small_trt_end2end',
                'region_class_name': 'object',
                'display_name': 'Objects',
                'display_name_singular': 'Object',
                'text_reader': 'none',
            },
        },
    )
    assert created.status == 201, created.text()
    activated = api.post(f'{prefix}/region_profiles/{name}/activate', data={})
    assert activated.status == 200, activated.text()
    return name


@pytest.fixture(scope='module')
def ingested_crop_id(
    api: Any, throwaway_project: dict[str, str], active_region_profile: str
) -> str:
    prefix = throwaway_project['prefix']
    upload = api.post(
        f'{prefix}/ingest/upload',
        multipart={
            'images': {
                'name': SAMPLE_IMAGE.name,
                'mimeType': 'image/jpeg',
                'buffer': _smoke_image_bytes(),
            },
            'source': 'e2e-writes',
        },
    )
    assert upload.status == 200, upload.text()
    assert upload.json()['summary']['failed'] == 0, upload.json()
    assert upload.json()['summary']['crops_indexed'] > 0, upload.json()

    def first_crop() -> str | None:
        page = api.get(f'{prefix}/crops?limit=5').json()
        return page['crops'][0]['crop_id'] if page['crops'] else None

    return _poll('the ingested crop to be listed', first_crop)


def test_ingest_lands_in_the_project_status(
    api: Any, throwaway_project: dict[str, str], ingested_crop_id: str
) -> None:
    status = api.get(f'{throwaway_project["prefix"]}/ingest/status').json()
    assert status['total'] >= 1, status
    assert any(row.get('key') == 'e2e-writes' for row in status['by_source']), status


def test_region_confirm_then_undo(
    api: Any, throwaway_project: dict[str, str], ingested_crop_id: str
) -> None:
    prefix = throwaway_project['prefix']
    bbox = [0.1, 0.1, 0.6, 0.6]

    written = api.put(
        f'{prefix}/crops/{ingested_crop_id}/regions',
        data={'boxes': [{'box_id': None, 'bbox_norm': bbox}], 'region_label_source': 'human'},
    )
    assert written.status == 200, written.text()
    (box,) = written.json()['item']['region_boxes']
    assert box['state'] == 'accepted'
    assert box['bbox_norm'] == pytest.approx(bbox)

    undone = api.post(f'{prefix}/crops/{ingested_crop_id}/region/undo')
    assert undone.status == 200, undone.text()
    assert undone.json()['region_boxes'] == []

    again = api.post(f'{prefix}/crops/{ingested_crop_id}/region/undo')
    assert again.status == 409, 'nothing is left to undo, the server says so'
