"""Workspace polish: equal gallery card heights, the hidden-count note in Clusters, select-all in every view.

- Gallery cards in one row have the same height, with or without the "+ more boxes" chip
"""

from __future__ import annotations

from fixtures.contract import fill, image_card, image_page
from harness.constants import ACTION_TIMEOUT_MS
from harness.screens import shot
from playwright.sync_api import expect


def serve_gallery(stub) -> None:
    stub.on('GET', r'(?<!/stats)/classes(\?|$)', {'classes': []})
    stub.on(
        'GET',
        r'/next_steps(\?|$)',
        fill('WorkflowSteps', steps=[], next=None, workspace_default_view='gallery'),
    )
    cards = [
        image_card('img-0'),
        image_card('img-1', boxes_truncated=True, width=480, height=640),
        image_card('img-2', width=1920, height=1080),
        *[image_card(f'img-{n}', boxes_truncated=n % 3 == 0) for n in range(3, 20)],
    ]
    stub.on('GET', r'/images(\?|$)', lambda _r, _m: (200, image_page(*cards)))


def test_gallery_cards_have_equal_heights(stub, page, app_url):
    serve_gallery(stub)
    for width in (1600, 430):
        page.set_viewport_size({'width': width, 'height': 900})
        page.goto(f'{app_url}/p/default/workspace?view=gallery')
        expect(page.get_by_test_id('image-card').first).to_be_visible(timeout=ACTION_TIMEOUT_MS)
        heights = page.eval_on_selector_all(
            '[data-testid="image-card"]',
            'els => els.map(e => Math.round(e.getBoundingClientRect().height))',
        )
        assert len(set(heights)) == 1, (width, heights)
        shot(page, f'gallery-card-heights-{width}')
