"""The Gallery view: whole images with served boxes and counts, near-dup badge, unlabelled chip.

- Gallery is the default view when `GET /next_steps` serves it
- the filter in the URL is sent to `GET /images`, and "Has unlabelled objects" sends
  `label_state=has_unlabeled`
- two selected images are acted on through `selection.filter.image_ids`, with the server's dry-run count
- Details opens the image details panel with an Open image link
- at 430 px the page does not scroll sideways
"""

from __future__ import annotations

import re
from urllib.parse import parse_qs, urlparse

from fixtures.contract import fill, image_card, image_page, label_op_accepted
from harness.constants import ACTION_TIMEOUT_MS
from harness.label_ops import stub_label_op
from harness.screens import shot
from playwright.sync_api import expect


CLASSES = [
    {
        'class_id': 1,
        'class_name': 'widget',
        'kind': 'item',
        'group': 'g',
        'hotkey_letter': 'k',
        'sample_count': 10,
        'validated_count': 5,
        'deprecated': False,
    }
]
NEAR_DUP = fill('ImageNearDup', group_id='g1', purpose='dedup', size=3, is_representative=False)


def cards():
    return [
        image_card('img-0'),
        image_card('img-1', near_dup=NEAR_DUP),
        image_card('img-2', boxes_truncated=True),
    ]


def serve(stub, queries: list[dict]) -> None:
    stub.on('GET', r'(?<!/stats)/classes(\?|$)', {'classes': CLASSES})
    stub.on(
        'GET',
        r'/next_steps(\?|$)',
        fill('WorkflowSteps', steps=[], next=None, workspace_default_view='gallery'),
    )

    def images(request, _match):
        queries.append(parse_qs(urlparse(request.url).query))
        return (200, image_page(*cards()))

    stub.on('GET', r'/images(\?|$)', images)
    stub.on(
        'GET',
        r'/images/img-1$',
        fill(
            'ImageDetail',
            image=fill('ImageInfo', image_id='img-1', file_name='img-1.jpg', in_holdout=False),
            counts=fill('ImageCounts', items=2, labeled=1, unlabeled=1),
            items=[],
            boxes=[],
        ),
    )


def test_gallery_is_the_default_and_draws_served_counts_and_boxes(stub, page, app_url):
    queries: list[dict] = []
    serve(stub, queries)
    page.goto(f'{app_url}/p/default/workspace')
    expect(page.get_by_test_id('gallery-view')).to_be_visible(timeout=ACTION_TIMEOUT_MS)
    expect(page.get_by_test_id('image-card')).to_have_count(3)
    expect(page.get_by_test_id('gallery-total')).to_have_text('3')
    expect(page.get_by_test_id('near-dup-badge')).to_have_count(1)
    expect(page.get_by_test_id('boxes-truncated')).to_have_count(1)
    first = page.get_by_test_id('image-card').first
    expect(first.get_by_test_id('chip-items')).to_contain_text('2')
    expect(first.get_by_test_id('overlay-box')).to_have_count(2)
    shot(page, 'gallery', widths=(1600, 430))


def test_class_filter_carries_and_the_unlabelled_chip_sends_label_state(stub, page, app_url):
    queries: list[dict] = []
    serve(stub, queries)
    page.goto(f'{app_url}/p/default/workspace?view=gallery&class_name=widget')
    expect(page.get_by_test_id('image-card').first).to_be_visible(timeout=ACTION_TIMEOUT_MS)
    assert queries[-1]['class_name'] == ['widget']
    assert 'label_state' not in queries[-1]
    with page.expect_request(
        lambda r: '/images?' in r.url and 'label_state=has_unlabeled' in r.url,
        timeout=ACTION_TIMEOUT_MS,
    ):
        page.get_by_test_id('chip-has-unlabeled').click()
    expect(page.get_by_test_id('chip-has-unlabeled')).to_have_attribute('aria-pressed', 'true')
    page.get_by_test_id('image-card').first.wait_for(timeout=ACTION_TIMEOUT_MS)
    assert queries[-1]['class_name'] == ['widget']
    shot(page, 'gallery-unlabelled-chip', widths=(1600, 430))


def test_selected_images_are_acted_on_through_image_ids(stub, page, app_url):
    queries: list[dict] = []
    serve(stub, queries)
    label_calls: list[dict] = []
    stub_label_op(stub, r'/crops/batch_label$', 'op-g', updated=4, method='PUT')

    def label(request, _match):
        body = request.post_data_json or {}
        label_calls.append(body)
        if body.get('dry_run'):
            return (200, {'dry_run': True, 'selected': 4})
        return (202, label_op_accepted('op-g', 'job-op-g'))

    stub.on('PUT', r'/crops/batch_label$', label)
    page.goto(f'{app_url}/p/default/workspace?view=gallery')
    expect(page.get_by_test_id('image-card').first).to_be_visible(timeout=ACTION_TIMEOUT_MS)
    page.get_by_test_id('image-card-select').nth(0).click()
    page.get_by_test_id('image-card-select').nth(1).click(modifiers=['Control'])
    expect(page.get_by_test_id('bulk-summary')).to_be_visible(timeout=ACTION_TIMEOUT_MS)
    page.get_by_test_id('assign-selected').click()
    page.get_by_role('combobox', name='Search classes').fill('widget')
    page.get_by_role('dialog').get_by_role('option', name=re.compile('widget')).first.click()
    expect(page.get_by_test_id('selection-count')).to_contain_text('4', timeout=ACTION_TIMEOUT_MS)
    assert label_calls[0]['selection']['filter']['image_ids'] == ['img-0', 'img-1']
    errors = [c for c in stub.console_errors if c.startswith('pageerror')]
    assert not errors, errors[:3]


def test_details_opens_the_image_with_an_open_image_link(stub, page, app_url):
    serve(stub, [])
    page.set_viewport_size({'width': 1600, 'height': 900})
    page.goto(f'{app_url}/p/default/workspace?view=gallery')
    expect(page.get_by_test_id('image-card').first).to_be_visible(timeout=ACTION_TIMEOUT_MS)
    page.get_by_test_id('image-card-open').nth(1).click()
    expect(page.get_by_test_id('image-details')).to_be_visible(timeout=ACTION_TIMEOUT_MS)
    expect(page.get_by_test_id('open-image')).to_have_attribute(
        'href', re.compile('/images/img-1$')
    )
    shot(page, 'gallery-details', widths=(1600, 430))


def test_no_sideways_scroll_at_430(stub, page, app_url):
    serve(stub, [])
    page.set_viewport_size({'width': 430, 'height': 900})
    page.goto(f'{app_url}/p/default/workspace?view=gallery')
    expect(page.get_by_test_id('image-card').first).to_be_visible(timeout=ACTION_TIMEOUT_MS)
    assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth')
