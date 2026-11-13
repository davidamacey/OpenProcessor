"""Detections: a plain click on a card selects it alone and the details follow it; a Ctrl click
adds to the selection and, from two cards, the details count the selection. The "view" overlay of
a card opens the source image without touching the selection. A thumbnail that fails twice is a
"No preview" tile."""

from __future__ import annotations

import re

from harness.constants import ACTION_TIMEOUT_MS
from harness.waits import wait_for_url_state
from playwright.sync_api import expect
from test_review_view import serve


def _open(page, app_url):
    page.set_viewport_size({'width': 1600, 'height': 1000})
    page.goto(f'{app_url}/p/default/workspace?view=detections')
    expect(page.locator('[data-crop-id]').nth(3)).to_be_visible(timeout=ACTION_TIMEOUT_MS)


def test_a_plain_click_focuses_the_card_and_a_ctrl_click_makes_the_details_count_them(
    stub, page, app_url
):
    serve(stub)
    _open(page, app_url)
    cards = page.locator('[data-crop-id]')
    cards.nth(1).click()
    wait_for_url_state(page, has=('focus=item:crop-1',))
    expect(page.get_by_test_id('bulk-count')).to_have_text('1 selected')
    cards.nth(2).click()
    wait_for_url_state(page, has=('focus=item:crop-2',))
    expect(page.get_by_test_id('bulk-count')).to_have_text('1 selected')
    expect(page.get_by_test_id('open-editor')).to_be_visible(timeout=ACTION_TIMEOUT_MS)

    cards.nth(3).click(modifiers=['Control'])
    expect(page.get_by_test_id('bulk-count')).to_have_text('2 selected')
    expect(page.get_by_test_id('selection-details')).to_be_visible(timeout=ACTION_TIMEOUT_MS)
    expect(page.get_by_test_id('open-editor')).to_have_count(0)


def test_the_view_overlay_keeps_the_selection(stub, page, app_url):
    serve(stub)
    stub.on('GET', r'/images/[^/]+/file$', (200, b'', 'image/jpeg'))
    stub.on(
        'GET',
        r'/images/[^/]+$',
        {'image': {'image_id': 'img-0', 'file_name': 'img-0.jpg'}, 'items': [], 'boxes': []},
    )
    _open(page, app_url)
    page.locator('[data-crop-id]').nth(1).click()
    expect(page.get_by_test_id('bulk-count')).to_have_text('1 selected')
    card = (
        page.locator('[data-crop-id]')
        .nth(2)
        .locator('xpath=ancestor::div[contains(@class,"group")][1]')
    )
    card.hover()
    card.get_by_role('button', name='Expand').click()
    expect(page.get_by_role('dialog', name='Source image')).to_be_visible(timeout=ACTION_TIMEOUT_MS)
    page.keyboard.press('Escape')
    expect(page.get_by_test_id('bulk-count')).to_have_text('1 selected')


def test_a_thumbnail_that_fails_twice_is_a_no_preview_tile(stub, page, app_url):
    serve(stub)
    stub.on('GET', r'/crops/crop-1/thumbnail', (404, {'detail': 'gone'}))
    _open(page, app_url)
    tile = page.get_by_test_id('no-preview')
    expect(tile).to_have_count(1, timeout=ACTION_TIMEOUT_MS)
    assert re.search('No preview', tile.inner_text())
