"""The image editor at `/p/default/images/<image_id>`.

- a dragged box, the class picker (the last class first) and Enter write `POST /images/{id}/items`
- a class typed as "Traffic light" is created under the slug the server answers (`traffic_light`)
- a move the server refuses with a 409 shows the served message and the server's box
- delete then Ctrl+Z writes `batch_exclude` then `batch_unexclude`
- a class change then Ctrl+Z undoes the label operation the write recorded (`POST /label_ops/{op_id}/undo`)
- Tab cycles the objects, `.` walks to the next image through `GET /images/locate`
- at 430 px the object list is a drawer and a touch drag draws
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any

from fixtures.image_editor import Editor
from harness.constants import ACTION_TIMEOUT_MS
from harness.label_ops import stub_label_op_undo
from harness.waits import wait_for_paint
from playwright.sync_api import expect


SCREENS = Path(__file__).resolve().parents[3] / 'artifacts_local' / 'screens' / 'workflow_3c'


def frame_point(page: Any, fx: float, fy: float) -> tuple[float, float]:
    box = page.get_by_test_id('editor-frame').bounding_box()
    return box['x'] + fx * box['width'], box['y'] + fy * box['height']


def drag(page: Any, a: tuple[float, float], b: tuple[float, float]) -> None:
    page.mouse.move(*frame_point(page, *a))
    page.mouse.down()
    page.mouse.move(*frame_point(page, *b), steps=6)
    page.mouse.up()


def shots(page: Any, name: str) -> None:
    SCREENS.mkdir(parents=True, exist_ok=True)
    for label, size in (('1600', (1600, 1000)), ('430', (430, 1000))):
        page.set_viewport_size({'width': size[0], 'height': size[1]})
        wait_for_paint(page)
        page.screenshot(path=str(SCREENS / f'{name}-{label}.png'), full_page=True)
    page.set_viewport_size({'width': 1600, 'height': 1000})


def test_draw_pick_and_save_with_the_last_class_first(stub, page, app_url):
    ed = Editor(stub)
    page.set_viewport_size({'width': 1600, 'height': 1000})
    ed.open(page, app_url)
    shots(page, 'select-state')

    page.keyboard.press('b')
    drag(page, (0.6, 0.1), (0.9, 0.4))
    picker = page.get_by_role('combobox')
    expect(picker).to_be_visible(timeout=ACTION_TIMEOUT_MS)
    shots(page, 'picker-open')
    page.get_by_role('option', name='bus').click()
    expect(page.get_by_test_id('save-state')).to_have_text('Saved', timeout=ACTION_TIMEOUT_MS)
    (body,) = ed.writes('POST', '/images/img-1/items')
    assert body['class_name'] == 'bus'
    assert [round(v, 2) for v in body['bbox_norm']] == [0.6, 0.1, 0.9, 0.4]
    assert len(body['request_id']) == 36
    expect(page.get_by_test_id('object-list').locator('li[data-crop-id]')).to_have_count(3)

    # The next box: the picker opens with the class just used on its first row, and Enter saves it.
    page.keyboard.press('b')
    drag(page, (0.1, 0.6), (0.3, 0.9))
    expect(page.get_by_role('combobox')).to_be_visible(timeout=ACTION_TIMEOUT_MS)
    expect(page.get_by_role('option').first).to_contain_text('bus')
    page.keyboard.press('Enter')
    expect(page.get_by_test_id('object-list').locator('li[data-crop-id]')).to_have_count(4)
    assert ed.writes('POST', '/images/img-1/items')[1]['class_name'] == 'bus'
    assert [e for e in stub.console_errors if e.startswith('pageerror')] == []


def test_a_typed_class_name_is_created_under_the_served_slug(stub, page, app_url):
    ed = Editor(stub)
    ed.open(page, app_url)
    page.keyboard.press('b')
    drag(page, (0.6, 0.1), (0.9, 0.4))
    box = page.get_by_role('combobox')
    box.fill('Traffic light')
    page.get_by_role('option', name='Create new class "Traffic light"').click()
    expect(page.get_by_test_id('object-list').locator('li[data-crop-id]')).to_have_count(
        3, timeout=ACTION_TIMEOUT_MS
    )
    assert ed.writes('POST', '/images/img-1/items')[0]['class_name'] == 'traffic_light'
    assert ed.writes('POST', '/classes')[-1] == {'name': 'Traffic light'}


def test_a_refused_move_shows_the_served_message_and_the_servers_box(stub, page, app_url):
    ed = Editor(stub)
    ed.fail_geometry = 'This box was changed by someone else.'
    page.set_viewport_size({'width': 1600, 'height': 1000})
    ed.open(page, app_url)
    page.keyboard.press('Tab')
    selected = page.get_by_test_id('selected-box')
    expect(selected).to_be_visible(timeout=ACTION_TIMEOUT_MS)
    drag(page, (0.2, 0.2), (0.5, 0.5))
    alert = page.get_by_test_id('save-message')
    expect(alert).to_have_text('This box was changed by someone else.', timeout=ACTION_TIMEOUT_MS)
    expect(page.get_by_test_id('save-state')).to_have_text('Changed by someone else, reloaded')
    # The editor shows the server's box now.
    server_box = page.locator('[data-testid="overlay-box"][data-crop-id="c1"]')
    expect(server_box).to_have_attribute('style', re.compile(r'left: 40%'))
    shots(page, 'conflict-banner')
    assert len(ed.writes('PATCH', '/geometry')) == 1


def test_delete_then_undo_restores_the_box(stub, page, app_url):
    ed = Editor(stub)
    ed.open(page, app_url)
    page.keyboard.press('Tab')
    page.keyboard.press('Delete')
    expect(page.get_by_test_id('object-deleted')).to_have_count(1, timeout=ACTION_TIMEOUT_MS)
    assert ed.writes('POST', '/crops/batch_exclude')[0] == {
        'crop_ids': ['c1'],
        'reason': 'not_an_object',
    }
    page.keyboard.press('Control+z')
    expect(page.get_by_test_id('object-deleted')).to_have_count(0, timeout=ACTION_TIMEOUT_MS)
    assert ed.writes('POST', '/crops/batch_unexclude')[0] == {'crop_ids': ['c1']}


def test_class_change_then_undo_reverts_the_recorded_label_operation(stub, page, app_url):
    ed = Editor(stub)
    undo_calls: list[dict] = []
    legacy: list[str] = []
    stub_label_op_undo(stub, 'op-c1', calls=undo_calls)
    stub.on(
        'POST',
        r'/crops/[^/]+/label/undo$',
        lambda request, match: legacy.append(match.string) or (200, {}),
    )
    ed.open(page, app_url)
    page.keyboard.press('Tab')
    page.keyboard.press('j')
    expect(page.get_by_test_id('save-state')).to_have_text('Saved', timeout=ACTION_TIMEOUT_MS)
    assert len(ed.writes('PUT', '/label')) == 1
    with page.expect_response(
        lambda r: r.request.method == 'POST' and r.url.endswith('/label_ops/op-c1/undo'),
        timeout=ACTION_TIMEOUT_MS,
    ):
        page.keyboard.press('Control+z')
    assert undo_calls == [{'dry_run': False}], undo_calls
    assert legacy == [], f'the class undo goes through the served operation: {legacy}'


def test_tab_cycles_and_dot_walks_to_the_next_image(stub, page, app_url):
    ed = Editor(stub)
    ed.open(page, app_url)
    rows = page.get_by_test_id('object-list').locator('li[data-crop-id]')
    page.keyboard.press('Tab')
    expect(rows.nth(0)).to_have_attribute('data-selected', 'true')
    page.keyboard.press('Tab')
    expect(rows.nth(1)).to_have_attribute('data-selected', 'true')
    expect(page.get_by_test_id('editor-position')).to_have_text('6 of 20')
    page.keyboard.press('.')
    page.wait_for_url('**/images/img-2?view=gallery', timeout=ACTION_TIMEOUT_MS)
    expect(page.get_by_test_id('editor-position')).to_have_text('7 of 20')
    page.keyboard.press(',')
    page.wait_for_url('**/images/img-1?view=gallery', timeout=ACTION_TIMEOUT_MS)


def test_narrow_screen_object_list_is_a_drawer_and_a_touch_drag_draws(stub, page, app_url):
    ed = Editor(stub)
    page.set_viewport_size({'width': 430, 'height': 900})
    ed.open(page, app_url)
    toggle = page.get_by_test_id('objects-toggle')
    expect(toggle).to_be_visible()
    expect(page.get_by_test_id('object-list')).to_be_hidden()
    toggle.click()
    expect(page.get_by_test_id('object-list')).to_be_visible()
    toggle.click()
    assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth')

    # A finger drag, sent as real touch input.
    cdp = page.context.new_cdp_session(page)
    cdp.send('Emulation.setTouchEmulationEnabled', {'enabled': True})
    page.keyboard.press('b')
    (x1, y1), (x2, y2) = frame_point(page, 0.6, 0.1), frame_point(page, 0.9, 0.4)
    cdp.send(
        'Input.dispatchTouchEvent', {'type': 'touchStart', 'touchPoints': [{'x': x1, 'y': y1}]}
    )
    for step in range(1, 6):
        cdp.send(
            'Input.dispatchTouchEvent',
            {
                'type': 'touchMove',
                'touchPoints': [{'x': x1 + (x2 - x1) * step / 5, 'y': y1 + (y2 - y1) * step / 5}],
            },
        )
    cdp.send('Input.dispatchTouchEvent', {'type': 'touchEnd', 'touchPoints': []})
    expect(page.get_by_role('combobox')).to_be_visible(timeout=ACTION_TIMEOUT_MS)
    shots(page, 'touch-draw-picker')


def test_three_boxes_and_two_class_changes_by_keyboard_in_under_a_minute(stub, page, app_url):
    """The lane's target: draw three boxes (a key, a drag, Enter) and change two classes (Tab, a
    class hotkey) in under a minute. The time is printed for the hand-back."""
    ed = Editor(stub)
    page.set_viewport_size({'width': 1600, 'height': 1000})
    ed.open(page, app_url)
    rows = page.get_by_test_id('object-list').locator('li[data-crop-id]')
    started = time.monotonic()
    for i, (a, b) in enumerate(
        [((0.55, 0.05), (0.75, 0.25)), ((0.8, 0.05), (0.95, 0.25)), ((0.55, 0.3), (0.75, 0.45))]
    ):
        page.keyboard.press('b')
        drag(page, a, b)
        page.keyboard.press('Enter')
        expect(rows).to_have_count(3 + i, timeout=ACTION_TIMEOUT_MS)
    # The drawn box is selected: a class hotkey changes it; Tab moves on to the next.
    page.keyboard.press('k')
    expect(rows.nth(4)).to_contain_text('car', timeout=ACTION_TIMEOUT_MS)
    page.keyboard.press('Tab')
    page.keyboard.press('j')
    elapsed = time.monotonic() - started
    expect(page.locator('li[data-selected="true"]')).to_contain_text(
        'bus', timeout=ACTION_TIMEOUT_MS
    )
    assert len(ed.writes('POST', '/images/img-1/items')) == 3
    assert len(ed.writes('PUT', '/label')) == 2
    SCREENS.mkdir(parents=True, exist_ok=True)
    (SCREENS / 'keyboard-target.txt').write_text(f'{elapsed:.1f} seconds\n')
    assert elapsed < 60, elapsed
