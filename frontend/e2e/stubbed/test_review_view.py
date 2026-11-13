"""Review is a Workspace view: Enter on a grid card opens it at that card, Esc returns to the
same card, and a deep link lands on its item with one locate request.

- grid -> focus the third card -> Enter -> confirm it -> Esc: the fourth card (the next
  unhandled) is focused and the URL is `view=detections&focus=item:<id>`
- `?view=review&queue=to_label&focus=item:<id>` locates the item with ONE request and no
  paging scan
- the rail's queues are the served ones with their served counts
- the four Review states render (machine label with its proposal, a region item, an empty queue,
  the shared class picker); `OP_E2E_SHOT_DIR` saves them at 1600 and 430 px
"""

from __future__ import annotations

import re

from fixtures.contract import crop_label_written, fill
from fixtures.wire import make_item, review_tab, review_tabs
from harness.constants import ACTION_TIMEOUT_MS, review_url
from harness.screens import shot
from playwright.sync_api import expect
from test_region_confirm import CLASSES as REGION_CLASSES, tag_item


CLASSES = [
    {
        'class_id': 1,
        'class_name': 'ducati',
        'kind': 'item',
        'group': 'moto',
        'hotkey_letter': 'k',
        'sample_count': 10,
        'validated_count': 5,
        'deprecated': False,
    },
]


def item(i: int) -> dict:
    return make_item(
        crop_id=f'crop-{i}',
        image_id=f'img-{i}',
        class_id=1,
        class_name='ducati',
        proposed_class_id=1,
        proposed_class_name='ducati',
        label_validated=False,
        label_source='model_suggestion',
        thumbnail_url=f'/curation/crops/crop-{i}/thumbnail',
    )


def serve(stub) -> None:
    stub.on('GET', r'(?<!/stats)/classes(\?|$)', {'classes': CLASSES})
    items = [item(i) for i in range(6)]

    def crops(request, _match):
        # A page cursor starts the list at the located item (index 2), as the server does.
        rows = items[2:] if 'cursor=cb' in request.url else items
        return (200, {'total': 6, 'next_cursor': None, 'limit': 48, 'crops': rows})

    stub.on('GET', r'/crops(\?|$)', crops)
    # The details panel reads the focused item and its history.
    stub.on('GET', r'/crops/[^/]+/history$', {'crop_id': 'crop-0', 'entries': []})
    stub.on('GET', r'/crops/(crop-\d+)$', lambda r, m: (200, item(int(m.group(1)[5:]))))
    stub.on(
        'GET',
        r'/review/tabs(\?|$)',
        review_tabs(
            review_tab('to_label', 'To label', count=412),
            review_tab('check_machine', 'Check machine labels', count=7),
        ),
    )


def test_enter_opens_review_at_the_card_and_esc_returns_to_it(stub, page, app_url):
    serve(stub)
    stub.on(
        'GET',
        r'/crops/locate(\?|$)',
        {
            'crop_id': 'crop-2',
            'in_view': True,
            'index': 2,
            'total': 6,
            'cursor_before': 'cb',
            'cursor_after': 'ca',
            'reason': None,
            'reason_message': None,
        },
    )
    stub.on(
        'PUT',
        r'/crops/([^/]+)/label$',
        lambda r, m: (200, crop_label_written('crop-2', op_id='op-crop-2')),
    )

    page.goto(f'{app_url}/p/default/workspace?view=detections')
    cards = page.locator('[data-crop-id]')
    cards.nth(2).wait_for(timeout=ACTION_TIMEOUT_MS)
    cards.nth(2).focus()
    page.keyboard.press('Enter')

    expect(page.get_by_test_id('review-view')).to_be_visible(timeout=ACTION_TIMEOUT_MS)
    page.wait_for_url('**view=review**')
    assert 'focus=item%3Acrop-2' in page.url or 'focus=item:crop-2' in page.url, page.url
    expect(page.get_by_test_id('queue-counter').first).to_contain_text('#3')

    with page.expect_response(
        lambda r: r.request.method == 'PUT' and r.url.endswith('/label'), timeout=ACTION_TIMEOUT_MS
    ):
        page.keyboard.press('Enter')
    expect(page.get_by_test_id('queue-counter').first).to_contain_text('#3')

    page.keyboard.press('Escape')
    expect(page.get_by_test_id('detections-view')).to_be_visible(timeout=ACTION_TIMEOUT_MS)
    assert 'view=detections' in page.url, page.url
    assert 'focus=item%3Acrop-3' in page.url or 'focus=item:crop-3' in page.url, page.url
    expect(page.locator('[data-crop-id="crop-3"]')).to_be_focused(timeout=ACTION_TIMEOUT_MS)


def test_a_deep_link_locates_its_item_with_one_request(stub, page, app_url):
    serve(stub)
    locate_calls: list[str] = []
    queue_calls: list[str] = []

    def locate(request, _match):
        locate_calls.append(request.url)
        return (
            200,
            {
                'crop_id': 'crop-4',
                'in_queue': True,
                'index': 4,
                'total': 412,
                'cursor_before': 'cb',
                'cursor_after': 'ca',
                'reason': None,
                'sort_applied': 'detector_confidence:desc',
                'sort_fallback_reason': None,
            },
        )

    def queue(request, _match):
        queue_calls.append(request.url)
        return (
            200,
            {
                'items': [dict(item(4), reason='no class yet'), dict(item(5), reason='')],
                'total': 412,
                'next_cursor': None,
                'limit': 30,
                'sort_applied': 'detector_confidence:desc',
                'sort_fallback_reason': None,
                'empty_reason': None,
                'hidden': {'test_holdout': 0, 'excluded': 0},
                'join_truncated': False,
            },
        )

    stub.on('GET', r'/review/to_label(\?|$)', queue)
    stub.on('GET', r'/review/to_label/locate(\?|$)', locate)

    page.goto(review_url(app_url, 'to_label', 'crop-4'))
    expect(page.get_by_test_id('queue-counter').first).to_contain_text(
        '#5', timeout=ACTION_TIMEOUT_MS
    )
    assert len(locate_calls) == 1, locate_calls
    assert 'crop_id=crop-4' in locate_calls[0]
    # The first page is read once to seed the buffer and once at the located cursor; never a scan.
    assert len(queue_calls) <= 2, queue_calls
    expect(page.get_by_test_id('filmstrip-position')).to_have_text('5 of 412')


def test_the_rail_lists_the_served_queues_with_their_counts(stub, page, app_url):
    serve(stub)
    page.goto(f'{app_url}/p/default/workspace?view=detections')
    rail = page.get_by_test_id('workspace-rail')
    expect(rail.get_by_test_id('queue-to_label')).to_contain_text('412', timeout=ACTION_TIMEOUT_MS)
    expect(rail.get_by_test_id('queue-check_machine')).to_contain_text('7')
    assert rail.locator('[data-testid^="queue-"]').count() == 2


SORT_OPTIONS = [
    {'id': 'detector_confidence:desc', 'label': 'Detector confidence (high first)'},
    {'id': 'recent', 'label': 'Recently updated'},
    {'id': 'atypicality', 'label': 'Atypicality (outlier-first)'},
]


def _queue(items: list[dict], **over) -> dict:
    return {
        'sort_options': SORT_OPTIONS,
        'sort_applied': 'detector_confidence:desc',
        'items': items,
        'total': len(items),
        'next_cursor': None,
        'limit': 30,
        'sort_fallback_reason': None,
        'empty_reason': None,
        'hidden': {'test_holdout': 0, 'excluded': 0},
        'join_truncated': False,
        **over,
    }


def test_the_review_states_render_at_1600_and_430(stub, page, app_url):
    stub.on(
        'GET',
        r'(?<!/stats)/classes(\?|$)',
        {'classes': [*CLASSES, *({**c, 'class_id': 99} for c in REGION_CLASSES)]},
    )
    stub.on(
        'GET',
        r'/review/tabs(\?|$)',
        review_tabs(
            review_tab('to_label', 'To label', count=0),
            review_tab('check_machine', 'Check machine labels', count=3),
            review_tab('regions', 'Plates', count=1),
        ),
    )
    machine = [dict(item(i), reason='model proposal, no human has confirmed it') for i in range(3)]
    stub.on('GET', r'/review/check_machine(\?|$)', _queue(machine))
    stub.on('GET', r'/review/regions(\?|$)', _queue([tag_item()]))
    stub.on(
        'GET',
        r'/review/to_label(\?|$)',
        _queue([], empty_reason='every item has a class; nothing is waiting'),
    )

    page.set_viewport_size({'width': 1600, 'height': 1000})
    page.goto(review_url(app_url, 'check_machine'))
    expect(page.get_by_test_id('queue-counter').first).to_be_visible(timeout=ACTION_TIMEOUT_MS)
    expect(page.get_by_test_id('details-panel').locator('dt').first).to_be_visible(
        timeout=ACTION_TIMEOUT_MS
    )
    shot(page, 'review-machine-label', widths=(1600, 430))

    page.set_viewport_size({'width': 1600, 'height': 1000})
    # The keys register a tick after the item renders: press until the picker answers.
    picker = page.locator('#class-picker-list')
    for _ in range(10):
        page.keyboard.press('/')
        if picker.is_visible():
            break
        page.wait_for_timeout(300)
    expect(picker).to_be_visible(timeout=ACTION_TIMEOUT_MS)
    shot(page, 'review-class-picker', widths=(1600, 430))
    page.keyboard.press('Escape')

    page.set_viewport_size({'width': 1600, 'height': 1000})
    page.goto(review_url(app_url, 'regions'))
    expect(page.get_by_test_id('queue-counter').first).to_be_visible(timeout=ACTION_TIMEOUT_MS)
    shot(page, 'review-region-item', widths=(1600, 430))

    # An empty queue is not a dead end: the filter row names the served next step.
    step = {
        'id': 'review_disagreements',
        'label': 'Review the disagreements',
        'detail': '3 items where the models disagree',
        'done': False,
        'section': 'workspace',
        'query': 'view=review&queue=check_machine',
    }
    stub.on(
        'GET',
        r'/next_steps(\?|$)',
        fill('WorkflowSteps', steps=[step], next=step, workspace_default_view='review'),
    )
    page.set_viewport_size({'width': 1600, 'height': 1000})
    page.goto(review_url(app_url, 'to_label'))
    expect(page.get_by_test_id('queue-empty')).to_be_visible(timeout=ACTION_TIMEOUT_MS)
    expect(page.get_by_test_id('workspace-filter-bar')).to_contain_text(
        'Review the disagreements', timeout=ACTION_TIMEOUT_MS
    )
    shot(page, 'review-empty-queue', widths=(1600, 430))


def test_enter_on_a_table_row_opens_review_at_that_row_and_esc_returns_to_the_table(
    stub, page, app_url
):
    from test_table_view import png, serve as serve_table

    serve_table(stub)
    serve(stub)
    # The filmstrip also draws the rows past the table's four.
    stub.on('GET', r'/crops/crop-\d/thumbnail', (200, png(60), 'image/png'))
    stub.on(
        'GET',
        r'/crops/locate(\?|$)',
        {
            'crop_id': 'crop-1',
            'in_view': True,
            'index': 1,
            'total': 6,
            'cursor_before': 'cb',
            'cursor_after': 'ca',
            'reason': None,
            'reason_message': None,
        },
    )
    page.goto(f'{app_url}/p/default/workspace?view=table')
    rows = page.get_by_test_id('data-table-row')
    expect(rows).to_have_count(4, timeout=ACTION_TIMEOUT_MS)
    rows.nth(1).click()
    page.keyboard.press('Enter')

    expect(page.get_by_test_id('review-view')).to_be_visible(timeout=ACTION_TIMEOUT_MS)
    assert 'view=review' in page.url, page.url
    assert 'focus=item%3Acrop-1' in page.url or 'focus=item:crop-1' in page.url, page.url

    page.keyboard.press('Escape')
    expect(page.get_by_test_id('data-table-row').first).to_be_visible(timeout=ACTION_TIMEOUT_MS)
    assert 'view=table' in page.url, page.url


def test_the_review_sort_is_the_served_one_and_belongs_to_its_queue(stub, page, app_url):
    from urllib.parse import parse_qs, urlparse

    serve(stub)
    seen: list[dict] = []

    def queue(request, _match):
        q = {k: v[0] for k, v in parse_qs(urlparse(request.url).query).items()}
        seen.append(q)
        applied = q.get('sort', 'detector_confidence:desc')
        return (200, _queue([item(0), item(1)], sort_applied=applied))

    stub.on('GET', r'/review/to_label(\?|$)', queue)
    stub.on('GET', r'/review/check_machine(\?|$)', queue)

    page.goto(review_url(app_url, 'to_label'))
    select = page.get_by_test_id('review-sort')
    expect(select).to_be_visible(timeout=ACTION_TIMEOUT_MS)
    assert [o.inner_text() for o in select.locator('option').all()] == [
        o['label'] for o in SORT_OPTIONS
    ]
    assert select.input_value() == 'detector_confidence:desc'

    with page.expect_response(
        lambda r: 'sort=atypicality' in r.url and '/review/to_label' in r.url,
        timeout=ACTION_TIMEOUT_MS,
    ):
        select.select_option('atypicality')
    expect(select).to_have_value('atypicality', timeout=ACTION_TIMEOUT_MS)
    assert seen[-1]['sort'] == 'atypicality', seen
    expect(page).to_have_url(re.compile(r'sort_review=atypicality'), timeout=ACTION_TIMEOUT_MS)

    page.get_by_test_id('queue-check_machine').click()
    expect(page.get_by_test_id('review-sort')).to_have_value(
        'detector_confidence:desc', timeout=ACTION_TIMEOUT_MS
    )
    expect(page).not_to_have_url(re.compile(r'sort_review'), timeout=ACTION_TIMEOUT_MS)


def test_text_search_in_review_uses_the_workspace_query_and_swaps_the_queue(stub, page, app_url):
    from urllib.parse import parse_qs, urlparse

    serve(stub)
    stub.on(
        'GET',
        r'/methods(\?|$)',
        {
            'strategies': [
                {'id': 'semantic_search', 'axis': 'overlay', 'label': 'S', 'status': 'stable'}
            ],
            'flags': {},
        },
    )
    seen: list[dict] = []

    def queue(_request, _match):
        return (200, _queue([item(0), item(1), item(2)], sort_applied='detector_confidence:desc'))

    def text_search(request, _match):
        seen.append(dict(parse_qs(urlparse(request.url).query)))
        hit = {**item(7), 'semantic_score': 0.88}
        return {
            'items': [hit],
            'total': 1,
            'page': 1,
            'page_size': 200,
            'unembedded_in_scope': 0,
            'space': None,
            'fuse': [],
        }

    stub.on('GET', r'/review/to_label(\?|$)', queue)
    stub.on('GET', r'/search/text(\?|$)', text_search)

    page.goto(review_url(app_url, 'to_label', class_name='ducati', import_id='imp_1'))
    expect(page.get_by_test_id('queue-counter').first).to_contain_text(
        '3 total', timeout=ACTION_TIMEOUT_MS
    )
    box = page.get_by_placeholder('Search, e.g. "blurry closeup"')
    expect(box).to_be_visible(timeout=ACTION_TIMEOUT_MS)
    box.fill('red widget')
    box.press('Enter')

    expect(page.get_by_test_id('queue-counter').first).to_contain_text(
        '1 total', timeout=ACTION_TIMEOUT_MS
    )
    q = seen[-1]
    assert q['q'] == ['red widget'] and q['tab'] == ['to_label'], q
    assert q['class_name'] == ['ducati'] and q['import_id'] == ['imp_1'], q
    assert 'review_tab' not in q, q
    expect(page.get_by_text('match', exact=False).first).to_be_visible()
    expect(page).to_have_url(re.compile(r'q=red(\+|%20)widget'), timeout=ACTION_TIMEOUT_MS)

    page.get_by_title('Clear search').click()
    expect(page.get_by_test_id('queue-counter').first).to_contain_text(
        '3 total', timeout=ACTION_TIMEOUT_MS
    )
    assert 'q=' not in page.url, page.url


def test_each_fact_of_the_review_item_appears_once(stub, page, app_url):
    """The Review pane keeps the review facts (reason, proposal, model, actions); the Workspace
    details panel is the one place for provenance, history and the rest, without repeating the
    class, label source and confidence rows or the source image the pane already shows."""
    serve(stub)
    rich = dict(
        item(0),
        vlm_endpoint='env@abc123',
        vlm_model='example/vision-model',
        label_confidence=0.73,
        reason='model proposal, no human has confirmed it',
    )
    stub.on('GET', r'/crops/crop-0$', (200, rich))
    stub.on('GET', r'/review/check_machine(\?|$)', _queue([rich]))
    stub.on(
        'GET',
        r'/review/tabs(\?|$)',
        review_tabs(review_tab('check_machine', 'Check machine labels', count=1)),
    )
    page.set_viewport_size({'width': 1600, 'height': 1000})
    page.goto(review_url(app_url, 'check_machine'))
    panel = page.get_by_test_id('details-panel')
    expect(panel.get_by_test_id('vlm-provenance-endpoint')).to_have_text(
        'env@abc123', timeout=ACTION_TIMEOUT_MS
    )
    assert page.get_by_test_id('vlm-provenance-endpoint').count() == 1
    assert page.get_by_role('button', name='▸ Details').count() == 0
    rows = [t.strip() for t in panel.locator('dt').all_inner_texts()]
    assert 'Class' not in rows and 'Label source' not in rows, rows
    assert 'Confidence' not in rows and 'Detector score' not in rows, rows
    assert panel.get_by_test_id('overlay-box').count() == 0
    meta = page.get_by_test_id('review-meta-pane')
    assert 'Current label' in meta.inner_text()
