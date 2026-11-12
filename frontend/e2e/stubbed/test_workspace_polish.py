"""Workspace polish: equal gallery card heights, the hidden-count note in Clusters, select-all in every view.

- Gallery cards in one row have the same height, with or without the "+ more boxes" chip
- The Clusters view shows "N matching" with the hidden note from the served total and hidden counts
- "Diverse sample…" asks the server for k diverse items over the Workspace filter and selects them
- Ctrl+A selects the loaded page in Gallery and Detections; a second Ctrl+A on a full page asks for all
  matching with the served total
"""

from __future__ import annotations

from fixtures.contract import cluster_card, cluster_list, fill, image_card, image_page
from fixtures.wire import make_item
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


def serve_detections(stub, total: int) -> None:
    stub.on('GET', r'(?<!/stats)/classes(\?|$)', {'classes': []})
    stub.on(
        'GET',
        r'/next_steps(\?|$)',
        fill('WorkflowSteps', steps=[], next=None, workspace_default_view='detections'),
    )
    items = [
        make_item(
            crop_id=f'crop-{i}', image_id=f'img-{i}', thumbnail_url=f'/crops/crop-{i}/thumbnail'
        )
        for i in range(3)
    ]
    stub.on(
        'GET',
        r'/crops(\?|$)',
        {'total': total, 'next_cursor': 'more', 'limit': 3, 'crops': items},
    )


def test_select_all_in_detections_asks_for_all_matching_on_a_second_press(stub, page, app_url):
    serve_detections(stub, 40)
    page.set_viewport_size({'width': 1600, 'height': 900})
    page.goto(f'{app_url}/p/default/workspace?view=detections')
    expect(page.get_by_test_id('detections-total')).to_have_text('40', timeout=ACTION_TIMEOUT_MS)
    page.get_by_test_id('detections-view').click(position={'x': 5, 'y': 5})
    page.keyboard.press('Control+a')
    expect(page.get_by_test_id('bulk-count')).to_have_text('3 selected', timeout=ACTION_TIMEOUT_MS)
    page.keyboard.press('Control+a')
    expect(page.get_by_test_id('select-all-matching')).to_have_text(
        'Select all 40 matching', timeout=ACTION_TIMEOUT_MS
    )
    shot(page, 'select-all-detections-40', widths=(1600, 430))


def test_select_all_in_gallery_selects_the_loaded_images(stub, page, app_url):
    serve_gallery(stub)
    page.set_viewport_size({'width': 1600, 'height': 900})
    page.goto(f'{app_url}/p/default/workspace?view=gallery')
    expect(page.get_by_test_id('image-card').first).to_be_visible(timeout=ACTION_TIMEOUT_MS)
    page.keyboard.press('Control+a')
    expect(page.get_by_test_id('bulk-count')).to_have_text('20 selected', timeout=ACTION_TIMEOUT_MS)


def test_clusters_show_the_matching_note_with_served_hidden_counts(stub, page, app_url):
    stub.on('GET', r'(?<!/stats)/classes(\?|$)', {'classes': []})
    stub.on(
        'GET',
        r'/clusters(\?|$)',
        cluster_list(
            cluster_card('cl-1', size=3),
            cluster_card('cl-2', size=2),
            hidden={'test_holdout': 25, 'excluded': 4},
        ),
    )
    page.set_viewport_size({'width': 1600, 'height': 900})
    page.goto(f'{app_url}/p/default/workspace?view=clusters')
    note = page.get_by_test_id('matching-note')
    expect(note).to_contain_text('2', timeout=ACTION_TIMEOUT_MS)
    expect(note).to_contain_text('25 test holdout and 4 ignored not listed')
    shot(page, 'clusters-matching-note', widths=(1600, 430))


def test_diverse_sample_selects_the_served_pick(stub, page, app_url):
    serve_detections(stub, 40)
    bodies: list[dict] = []

    def diverse(request, _match):
        bodies.append(request.post_data_json or {})
        return (
            200,
            {
                'crop_ids': ['crop-0', 'crop-2'],
                'method': 'kcenter_greedy',
                'version': 'v1',
                'n_pool': 40,
            },
        )

    stub.on('POST', r'/select/diverse$', diverse)
    page.set_viewport_size({'width': 1600, 'height': 900})
    page.goto(f'{app_url}/p/default/workspace?view=detections&class_name=widget')
    expect(page.get_by_test_id('detections-total')).to_have_text('40', timeout=ACTION_TIMEOUT_MS)
    page.get_by_test_id('diverse-sample-open').click()
    page.get_by_test_id('diverse-sample-k').fill('2')
    shot(page, 'diverse-sample-open', widths=(1600, 430))
    page.set_viewport_size({'width': 1600, 'height': 900})
    page.get_by_test_id('diverse-sample-start').click()
    expect(page.get_by_test_id('bulk-count')).to_have_text('2 selected', timeout=ACTION_TIMEOUT_MS)
    expect(page.get_by_test_id('diverse-sample-message')).to_have_text('Picked 2 of 40')
    assert bodies[0]['k'] == 2
    assert bodies[0]['scope']['item_filter']['class_names'] == ['widget']
    shot(page, 'diverse-sample-picked', widths=(1600, 430))
