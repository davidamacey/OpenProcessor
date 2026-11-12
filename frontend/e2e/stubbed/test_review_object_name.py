"""Review shows the served object name: an item the detector named but nobody has classed is
labelled by that name (marked as the detector's), never "no class yet" or a bare id. Set
REVIEW_NAME_SHOTS=<dir> to save screenshots at 1280 and 430 px."""

from __future__ import annotations

import os

from fixtures.wire import make_item
from harness.constants import ACTION_TIMEOUT_MS, review_url


def _unclassed() -> dict:
    item = make_item(
        crop_id='crop-1',
        image_id='img-1',
        class_id=None,
        class_name='',
        class_source='item_proposal',
        class_validated=False,
        label_validated=False,
        label_source='item_proposal',
        proposed_class_id=None,
        proposed_class_name='',
        proposal_name='stop sign',
        detector_class_name='stop sign',
        display_name='stop sign',
        display_name_kind='detector',
    )
    item['reason'] = 'needs human review'
    return item


def test_review_names_an_unclassed_item_by_its_served_name(stub, page, app_url):
    stub.on('GET', r'(?<!/stats)/classes(\?|$)', {'classes': []})
    stub.on(
        'GET',
        r'/review/(?!tabs)',
        (200, {'items': [_unclassed()], 'total': 1, 'next_cursor': None, 'limit': 30}),
    )
    page.goto(review_url(app_url, 'to_label'))
    row = page.get_by_test_id('current-label')
    row.wait_for(timeout=ACTION_TIMEOUT_MS)
    text = row.inner_text()
    assert 'stop sign' in text and "detector's label" in text
    assert 'no class yet' not in text
    shots = os.environ.get('REVIEW_NAME_SHOTS')
    if shots:
        page.screenshot(
            path=f'{shots}/review_name_{page.viewport_size["width"]}.png', full_page=True
        )
        page.set_viewport_size({'width': 430, 'height': 900})
        row.wait_for()
        page.screenshot(path=f'{shots}/review_name_430.png', full_page=True)
