"""The image details panel: a preview with its boxes, one editor link ("Edit boxes"), "Show its N
detections" that switches to Detections scoped to the image, and, from a map cell, a Back to the
cell that restores the cell's thumbnails."""

from __future__ import annotations

import re

from harness.constants import ACTION_TIMEOUT_MS
from playwright.sync_api import expect
from test_gallery_view import serve as serve_gallery
from test_map_view import CANVAS, _centre, _meta, _open, _serve_map


def _open_details(page, app_url):
    page.set_viewport_size({'width': 1600, 'height': 900})
    page.goto(f'{app_url}/p/default/workspace?view=gallery')
    expect(page.get_by_test_id('image-card').first).to_be_visible(timeout=ACTION_TIMEOUT_MS)
    page.get_by_test_id('image-card-open').nth(1).click()
    expect(page.get_by_test_id('image-details')).to_be_visible(timeout=ACTION_TIMEOUT_MS)


def test_image_details_render_a_preview_and_one_editor_link(stub, page, app_url):
    serve_gallery(stub, [])
    _open_details(page, app_url)
    details = page.get_by_test_id('image-details')
    expect(details.get_by_test_id('image-preview')).to_be_visible()
    assert details.locator('img[alt="source"]').count() == 1
    editors = details.get_by_role('link', name=re.compile('Edit boxes'))
    expect(editors).to_have_count(1)
    expect(details.get_by_role('link', name='Open image')).to_have_count(0)
    expect(details.get_by_role('button', name='View full size')).to_be_visible()


def test_clicking_the_preview_opens_the_lightbox(stub, page, app_url):
    serve_gallery(stub, [])
    stub.on('GET', r'/images/img-\d+/file', (200, b'', 'image/jpeg'))
    _open_details(page, app_url)
    page.get_by_test_id('image-preview-open').click()
    expect(page.get_by_test_id('lightbox')).to_be_visible(timeout=ACTION_TIMEOUT_MS)
    page.keyboard.press('Escape')
    expect(page.get_by_test_id('lightbox')).to_have_count(0)


def test_show_its_detections_switches_to_detections_with_the_image_scope(stub, page, app_url):
    serve_gallery(stub, [])
    stub.on('GET', r'/crops(\?|$)', {'total': 0, 'next_cursor': None, 'limit': 60, 'crops': []})
    _open_details(page, app_url)
    page.get_by_test_id('scope-image').click()
    page.wait_for_url(
        re.compile(r'view=detections.*scope=image(:|%3A)img-1'), timeout=ACTION_TIMEOUT_MS
    )
    expect(page.get_by_test_id('scope-chip')).to_contain_text('Image:')


def test_back_to_the_cell_restores_the_cell_thumbnails(stub, page, app_url):
    _serve_map(stub, page, app_url, _meta(app_url))
    _open(page, app_url)
    page.set_viewport_size({'width': 1600, 'height': 1000})
    page.locator(CANVAS).wait_for(timeout=ACTION_TIMEOUT_MS)
    x, y = _centre(page)
    page.mouse.click(x, y)
    expect(page.get_by_test_id('cell-card').first).to_be_visible(timeout=ACTION_TIMEOUT_MS)
    page.get_by_test_id('cell-card').first.click()
    expect(page.get_by_test_id('image-details')).to_be_visible(timeout=ACTION_TIMEOUT_MS)
    expect(page.get_by_test_id('cell-cards')).to_have_count(0)
    expect(page.get_by_test_id('back-to-cell')).to_contain_text(re.compile(r'\(\d[\d,]* photos?\)'))
    page.get_by_test_id('back-to-cell').click()
    expect(page.get_by_test_id('cell-card')).to_have_count(12, timeout=ACTION_TIMEOUT_MS)
