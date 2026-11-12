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


def _item(api: Any, prefix: str, crop_id: str) -> dict[str, Any]:
    resp = api.get(f'{prefix}/crops/{crop_id}')
    assert resp.status == 200, resp.text()
    return resp.json()


def _image_items(api: Any, prefix: str, image_id: str) -> list[dict[str, Any]]:
    page = api.get(f'{prefix}/crops?image_id={image_id}&page_size=200&include_excluded=true').json()
    return list(page['crops'])


def _iou(a: list[float], b: list[float]) -> float:
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = w * h
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0


def test_draw_move_delete_survive_a_redetect(
    api: Any, throwaway_project: dict[str, str], ingested_crop_id: str
) -> None:
    """A drawn box, a moved detector box and a deleted detector box are all a person's: a
    re-detect of the image changes none of them and does not re-create the deleted one."""
    prefix = throwaway_project['prefix']
    detected = _item(api, prefix, ingested_crop_id)
    image_id = detected['image_id']
    detector_box = list(detected['bbox_norm'])
    api.post(f'{prefix}/classes', data={'name': 'e2e-thing', 'group': 'e2e'})

    drawn = api.post(
        f'{prefix}/images/{image_id}/items',
        data={
            'bbox_norm': [0.02, 0.02, 0.12, 0.12],
            'class_name': 'e2e-thing',
            'request_id': str(uuid.uuid4()),
        },
    )
    assert drawn.status == 201, drawn.text()
    drawn_id = drawn.json()['item']['crop_id']

    moved_to = [0.03, 0.03, 0.2, 0.2]
    moved = api.patch(
        f'{prefix}/crops/{drawn_id}/geometry',
        data={'bbox_norm': moved_to, 'expected_geometry_revision': 1},
    )
    assert moved.status == 200, moved.text()

    gone = api.post(
        f'{prefix}/crops/batch_exclude',
        data={'crop_ids': [ingested_crop_id], 'reason': 'not_an_object'},
    )
    assert gone.status == 200, gone.text()

    redetect = api.post(f'{prefix}/images/{image_id}/reprocess', data={'scopes': ['detect']})
    assert redetect.status in (200, 202), redetect.text()
    job = redetect.json().get('job')
    if job:
        _poll(
            're-detect job to finish',
            lambda: (
                api.get(f'{prefix}/reprocess/jobs/{job["job_id"]}').json().get('status')
                in ('done', 'completed', 'succeeded')
            ),
        )

    assert _item(api, prefix, drawn_id)['bbox_norm'] == pytest.approx(moved_to)
    deleted = _item(api, prefix, ingested_crop_id)
    assert deleted['class_excluded'] is True
    on_the_deleted_box = [
        row
        for row in _image_items(api, prefix, image_id)
        if _iou(list(row['bbox_norm']), detector_box) > 0.5
    ]
    assert [row['crop_id'] for row in on_the_deleted_box] == [ingested_crop_id]


def test_editor_draws_moves_deletes_and_undoes_in_the_browser(
    browser: Any,
    live_url: str,
    api: Any,
    throwaway_project: dict[str, str],
    ingested_crop_id: str,
) -> None:
    """The image editor on the throwaway project: draw a box and pick a class, move it, delete the
    detector's box and undo that, and every change is there after a reload."""
    prefix = throwaway_project['prefix']
    detected = _item(api, prefix, ingested_crop_id)
    image_id = detected['image_id']
    api.post(f'{prefix}/classes', data={'name': 'e2e_editor', 'group': 'e2e'})

    context = browser.new_context(
        base_url=live_url,
        extra_http_headers=_auth_headers(),
        viewport={'width': 1600, 'height': 1000},
    )
    page = context.new_page()
    try:
        url = f'/p/{throwaway_project["slug"]}/images/{image_id}'
        page.goto(url)
        canvas = page.get_by_test_id('editor-canvas')
        canvas.wait_for(timeout=POLL_TIMEOUT_S * 1000)
        page.locator('[data-testid="overlay-box"]').first.wait_for(timeout=POLL_TIMEOUT_S * 1000)

        def at(fx: float, fy: float) -> tuple[float, float]:
            box = page.get_by_test_id('editor-frame').bounding_box()
            assert box is not None
            return box['x'] + fx * box['width'], box['y'] + fy * box['height']

        # Draw a small box in the top-left corner and give it the class.
        page.keyboard.press('b')
        page.mouse.move(*at(0.02, 0.02))
        page.mouse.down()
        page.mouse.move(*at(0.12, 0.12), steps=8)
        page.mouse.up()
        page.get_by_role('combobox').fill('e2e_editor')
        page.get_by_role('option', name='e2e_editor').click()
        page.get_by_text('Saved', exact=True).wait_for(timeout=POLL_TIMEOUT_S * 1000)
        drawn = _poll(
            'the drawn item to be stored',
            lambda: next(
                (
                    row
                    for row in _image_items(api, prefix, image_id)
                    if row['crop_id'] != ingested_crop_id
                    and row['bbox_norm'] == pytest.approx([0.02, 0.02, 0.12, 0.12], abs=0.01)
                ),
                None,
            ),
        )
        assert drawn['class_name'] == 'e2e_editor'

        # Move it: the box is selected after the draw, so a drag from its middle moves it.
        page.mouse.move(*at(0.07, 0.07))
        page.mouse.down()
        page.mouse.move(*at(0.27, 0.27), steps=8)
        page.mouse.up()
        moved = _poll(
            'the move to be stored',
            lambda: (
                _item(api, prefix, drawn['crop_id'])['bbox_norm'][0] > 0.15
                and _item(api, prefix, drawn['crop_id'])
            ),
        )
        assert moved['geometry_revision'] >= 2

        # Delete the drawn box, then bring it back with undo.
        page.keyboard.press('Delete')
        _poll(
            'the delete to be stored',
            lambda: _item(api, prefix, drawn['crop_id'])['class_excluded'] is True,
        )
        page.keyboard.press('Control+z')
        _poll(
            'the undo to be stored',
            lambda: _item(api, prefix, drawn['crop_id'])['class_excluded'] is False,
        )

        page.reload()
        page.locator('[data-testid="overlay-box"]').first.wait_for(timeout=POLL_TIMEOUT_S * 1000)
        final = _item(api, prefix, drawn['crop_id'])
        assert final['class_excluded'] is False
        assert final['bbox_norm'][0] > 0.15
        assert final['geometry_source'] == 'human'
    finally:
        context.close()
