"""Multi-box editor modal: `SlotBboxEditor.svelte` (the CropCard pencil ✎, `/clusters/
[id]`) now reuses `MultiBoxCanvas`/`multiBoxRegionController` for the
region slot instead of the legacy single-box canvas — the served region
slot has no `bboxField` at all, so the old single-box modal was
unreachable for region. This proves the pencil opens the multi-box
canvas, adding a box and saving sends a plain `PUT /crops/{id}/regions`
with the accumulated diff and no `region_status` key (this modal has no
"confirm" concept, unlike /review's Enter).

No fixed sleeps (parallel e2e, CLAUDE.md): every wait is a real selector
or `page.expect_request`.
"""

from __future__ import annotations

from fixtures.contract import box_view, cluster_card, cluster_list, image_detail, item_cluster
from fixtures.wire import REGION_CLASS, make_item
from harness.constants import ACTION_TIMEOUT_MS, PLACEHOLDER_PHOTO
from harness.screens import save_screens
from harness.waits import expect_handled


CLASSES = [
    {
        'class_id': 1,
        'class_name': REGION_CLASS,
        'kind': 'region',
        'group': 'widgets',
        'hotkey_letter': 'l',
        'sample_count': 10,
        'validated_count': 5,
        'deprecated': False,
    },
]

CLUSTER_UID = 'cl-fixture-0001'
CLUSTERS = cluster_list(cluster_card(CLUSTER_UID, size=1, dominant_class_name=REGION_CLASS))


def _box(box_id: str, bbox: list[float]) -> dict:
    return {
        'box_id': box_id,
        'display_name': 'Tag 1',
        'state': 'accepted',
        'bbox_norm': bbox,
        'bbox_in_parent': bbox,
        'score': 0.8,
        'detector': 'tag_detector_v1',
        'detector_version': '1',
        'source': 'detector',
        'bbox_correct': None,
        'confidence': None,
        'rejection_reason': None,
        'text': None,
        'text_raw': None,
        'text_confidence': None,
        'text_source': None,
        'text_frame': None,
        'text_conf_char_min': None,
        'text_reader': None,
        'text_reader_version': None,
        'text_vlm': None,
        'text_ocr': None,
        'text_disagreement': None,
        'text_choice': None,
        'text_vlm_invalid': None,
        'cluster_partition_id': None,
        'cluster_uid': None,
        'cluster_state': None,
        'cluster_score': None,
        'cluster_margin': None,
        'cluster_assigned_by': None,
        'detected_at': '2026-05-06T07:08:09Z',
        'thumbnail_url': f'/curation/crops/tag-1/region_thumbnail?box_id={box_id}',
    }


def tag_item() -> dict:
    return make_item(
        crop_id='tag-1',
        image_id='img-1',
        class_id=1,
        class_name=REGION_CLASS,
        cluster=item_cluster(CLUSTER_UID),
        label_validated=False,
        label_source='model_suggestion',
        thumbnail_url='/curation/crops/tag-1/thumbnail',
        region_status='pending_verification',
        region_boxes=[_box('b1', [0.1, 0.1, 0.3, 0.3])],
    )


def test_cropcard_pencil_opens_multibox_editor_and_saves_with_no_region_status(stub, page, app_url):
    page.set_viewport_size({'width': 1280, 'height': 1400})
    puts: list[dict] = []

    stub.on('GET', r'(?<!/stats)/classes(\?|$)', {'classes': CLASSES})
    stub.on('GET', r'/clusters(\?|$)', CLUSTERS)
    stub.on(
        'GET',
        r'/crops(\?|$)',
        {'total': 1, 'next_cursor': None, 'limit': 60, 'crops': [tag_item()]},
    )
    stub.on('GET', r'/images/[^/]+/file$', (200, b'', 'image/jpeg'))

    def region_put(request, _match):
        body = request.post_data_json or {}
        puts.append(body)
        return (200, {'item': tag_item()})

    stub.on('PUT', r'/crops/([^/]+)/regions$', region_put)

    page.goto(f'{app_url}/p/default/clusters/{CLUSTER_UID}')
    page.wait_for_selector('img', timeout=ACTION_TIMEOUT_MS)

    thumbnail = page.locator('img').first
    thumbnail.hover()
    # An exact aria-label selector: a fuzzy role query must not pick another
    # control (toggling selection) instead of the pencil button.
    pencil = page.locator('button[aria-label="Edit widget tag"]')
    pencil.wait_for(timeout=ACTION_TIMEOUT_MS, state='visible')
    # The pencil must never share pixels with the card's other hover controls
    # (it once sat under "Expand", which swallowed the click).
    pb = pencil.bounding_box()
    assert pb is not None
    siblings = page.locator('button[aria-label="Expand"], button[aria-label="Show crop details"]')
    assert siblings.count() >= 1
    for sibling in siblings.all():
        sb = sibling.bounding_box()
        assert sb is not None
        overlap_x = min(pb['x'] + pb['width'], sb['x'] + sb['width']) - max(pb['x'], sb['x'])
        overlap_y = min(pb['y'] + pb['height'], sb['y'] + sb['height']) - max(pb['y'], sb['y'])
        assert overlap_x <= 0 or overlap_y <= 0, (pb, sb)
    pencil.click()

    canvas = page.get_by_test_id('multibox-canvas')
    canvas.wait_for(timeout=ACTION_TIMEOUT_MS)

    # Drag-draw a second box on empty canvas.
    box = canvas.bounding_box()
    assert box is not None
    start_x, start_y = box['x'] + box['width'] * 0.6, box['y'] + box['height'] * 0.6
    end_x, end_y = box['x'] + box['width'] * 0.8, box['y'] + box['height'] * 0.8
    page.mouse.move(start_x, start_y)
    page.mouse.down()
    page.mouse.move(end_x, end_y)
    page.mouse.up()

    save_button = page.get_by_role('button', name='Save', exact=True)
    save_button.wait_for(timeout=ACTION_TIMEOUT_MS)
    with expect_handled(
        page,
        lambda r: r.method == 'PUT' and '/regions' in r.url,
        timeout=ACTION_TIMEOUT_MS,
    ):
        save_button.click()

    assert len(puts) == 1, puts
    body = puts[0]
    assert 'region_status' not in body, (
        f'the standalone bbox editor has no confirm concept — must never send region_status: {body}'
    )
    assert body['boxes'][0] == {'box_id': 'b1'}, 'untouched sibling stays {box_id} only'
    assert body['boxes'][1]['box_id'] is None, 'the newly drawn box has no box_id'
    assert 'bbox_norm' in body['boxes'][1]

    errors = [c for c in stub.console_errors if c.startswith('pageerror')]
    assert not errors, f'no pageerror expected in the multi-box editor flow: {errors[:3]}'

    stub.assert_fail_closed()


def test_region_edit_and_crop_dialog_draw_the_served_names(stub, page, app_url):
    stub.on('GET', r'(?<!/stats)/classes(\?|$)', {'classes': CLASSES})
    stub.on('GET', r'/clusters(\?|$)', CLUSTERS)
    stub.on('GET', r'/crops(\?|$)', {'total': 1, 'page': 1, 'page_size': 60, 'crops': [tag_item()]})
    stub.on(
        'GET',
        r'/images/img-1$',
        image_detail(
            'img-1',
            [tag_item()],
            boxes=[
                box_view('tag-1', REGION_CLASS, bbox_norm=[0.1, 0.1, 0.5, 0.5]),
                box_view(
                    'tag-1',
                    'Tag 1',
                    box_id='b1',
                    kind='region',
                    label_kind='region',
                    bbox_norm=[0.15, 0.15, 0.3, 0.3],
                    state='accepted',
                ),
            ],
            width=640,
            height=480,
        ),
    )
    stub.on('GET', r'/images/[^/]+/file(\?|$)', (200, PLACEHOLDER_PHOTO, 'image/svg+xml'))
    stub.on('GET', r'/crops/[^/]+/history$', {'crop_id': 'tag-1', 'entries': []})

    page.goto(f'{app_url}/p/default/clusters/{CLUSTER_UID}')
    page.wait_for_selector('img', timeout=ACTION_TIMEOUT_MS)
    page.locator('img').first.hover()
    page.locator('button[aria-label="Edit widget tag"]').click()
    canvas = page.get_by_test_id('multibox-canvas')
    canvas.wait_for(timeout=ACTION_TIMEOUT_MS)
    assert canvas.locator('[aria-label="Tag 1"]').count() == 1, 'the served region name'
    save_screens(page, 'workflow_2c', 'region-edit')
    page.keyboard.press('Escape')

    page.locator('img').first.hover()
    page.locator('button[aria-label="Show crop details"]').click()
    source = page.get_by_test_id('source-box')
    source.get_by_test_id('overlay-box').first.wait_for(timeout=ACTION_TIMEOUT_MS)
    titles = [source.get_by_test_id('overlay-box').nth(i).get_attribute('title') for i in range(2)]
    assert titles == [REGION_CLASS, 'Tag 1'], titles
    save_screens(page, 'workflow_2c', 'crop-dialog')
    stub.assert_fail_closed()
