"""The review queue's region-text (plate) filter and its region-status select, together.

A region profile that reads text (the served `reads_text`) gives the region tab a "Text" input
(`textFilter` in `servedRegionSlot.ts`); what is typed reaches `GET /review/regions` as `text`,
beside the served `region_status` enum, and a cleared input drops the param. The backend's
`text` filter is plate text matched against the box text, so the stub answers it by the box text
the item carries: the rendered queue must be what the server returned, not a client-side filter.

`test_review_region_status_filter.py` covers the status select on its own; this file covers the
text input and the pair.
"""

from __future__ import annotations

from urllib.parse import parse_qs, urlparse

from fixtures.wire import REGION_CLASS, REGION_TAB_LABEL, REGION_TAB_URL_ID, make_item
from harness.constants import ACTION_TIMEOUT_MS
from harness.screens import save_screens


CLASSES = [
    {
        'class_id': 1,
        'class_name': REGION_CLASS,
        'kind': 'region',
        'group': 'widgets',
        'hotkey_letter': 'l',
        'sample_count': 40,
        'validated_count': 12,
        'cluster_size': 44,
        'deprecated': False,
    },
]

REVIEW_TABS = {
    'empty_state': {'has_probe_predictions': True, 'has_item_scores': True},
    'tabs': [
        {
            'id': 'regions',
            'label': REGION_TAB_LABEL,
            'description': '',
            'filters': ['text', 'region_status'],
            'filter_defaults': {'region_status': 'all'},
            'filter_specs': [
                {
                    'param': 'region_status',
                    'kind': 'enum',
                    'label': 'Status',
                    'options': [
                        {'value': 'all', 'label': 'All'},
                        {'value': 'detected', 'label': 'Detected only'},
                    ],
                    'min': None,
                    'max': None,
                    'description': '',
                    'default': 'all',
                    'allows_unset': False,
                },
            ],
        },
    ],
}

VOCABULARY = {
    'detectors': [],
    'region_sources': [],
    'chain_actors': [],
    'text_choices': [],
    'text_rules': None,
    'rejection_reasons': [],
}

PLATES = {'plate-a': 'KLM4821', 'plate-b': 'ZQX9000'}


def _item(crop_id: str) -> dict:
    return make_item(
        crop_id=crop_id,
        image_id=f'img-{crop_id}',
        class_id=1,
        class_name=REGION_CLASS,
        thumbnail_url=f'/curation/crops/{crop_id}/thumbnail',
        region_status='detected',
    )


def _query(url: str) -> dict[str, list[str]]:
    return parse_qs(urlparse(url).query)


def test_region_text_input_reaches_the_queue_with_the_status_filter(stub, page, app_url):
    calls: list[str] = []

    stub.on('GET', r'(?<!/stats)/classes(\?|$)', {'classes': CLASSES})
    stub.on('GET', r'/review/tabs(\?|$)', REVIEW_TABS)
    stub.on('GET', r'/regions/vocabulary(\?|$)', VOCABULARY)
    stub.on('GET', r'/crops/[^/]+/(image|thumbnail)', (200, b'', 'image/jpeg'))

    def regions(request, _match):
        calls.append(request.url)
        wanted = (_query(request.url).get('text') or [''])[0]
        ids = [cid for cid, plate in PLATES.items() if not wanted or wanted in plate]
        return (
            200,
            {'items': [_item(c) for c in ids], 'total': len(ids), 'page': 1, 'page_size': 30},
        )

    stub.on('GET', r'/review/regions(\?|$)', regions)

    page.goto(f'{app_url}/p/default/review?tab={REGION_TAB_URL_ID}')
    page.get_by_test_id('queue-counter').first.wait_for(timeout=ACTION_TIMEOUT_MS)
    assert all('text' not in _query(u) for u in calls), calls
    first_counter = page.get_by_test_id('queue-counter').first.inner_text()

    text_input = page.locator('label:has-text("Text") input[type="text"]').first
    text_input.wait_for(timeout=ACTION_TIMEOUT_MS)
    with page.expect_response(
        lambda r: '/review/regions' in r.url and 'text=KLM' in r.url, timeout=ACTION_TIMEOUT_MS
    ) as sent:
        text_input.fill('KLM')
    query = _query(sent.value.url)
    assert query['text'] == ['KLM']
    # The served default (`all`) is not sent; only a chosen value is.
    assert 'region_status' not in query
    page.wait_for_function(
        "(before) => document.querySelector('[data-testid=queue-counter]')?.textContent !== before",
        arg=first_counter,
        timeout=ACTION_TIMEOUT_MS,
    )

    select = page.locator('label:has-text("Status") select')
    with page.expect_response(
        lambda r: (
            '/review/regions' in r.url and 'region_status=detected' in r.url and 'text=KLM' in r.url
        ),
        timeout=ACTION_TIMEOUT_MS,
    ):
        select.select_option('detected')

    with page.expect_response(
        lambda r: '/review/regions' in r.url and 'text=' not in r.url, timeout=ACTION_TIMEOUT_MS
    ) as cleared:
        text_input.fill('')
    assert 'text' not in _query(cleared.value.url)
    assert _query(cleared.value.url)['region_status'] == ['detected']

    errors = [c for c in stub.console_errors if c.startswith('pageerror')]
    assert not errors, errors[:3]


MATCH_TABS = {
    **REVIEW_TABS,
    'tabs': [
        {
            **REVIEW_TABS['tabs'][0],
            'filters': ['text', 'region_text_match', 'region_status'],
            'filter_specs': [
                *REVIEW_TABS['tabs'][0]['filter_specs'],
                {
                    'param': 'region_text_match',
                    'kind': 'enum',
                    'label': 'Text match',
                    'options': [
                        {'value': '', 'label': 'Any'},
                        {'value': 'exact', 'label': 'Exact (letters and digits)'},
                        {'value': 'fuzzy', 'label': 'Fuzzy (one edit)'},
                        {'value': 'fragment', 'label': 'Contains fragment'},
                    ],
                    'min': None,
                    'max': None,
                    'description': '',
                    'default': None,
                    'allows_unset': True,
                },
            ],
        },
    ],
}


def test_served_text_match_mode_reaches_the_queue(stub, page, app_url):
    stub.on('GET', r'(?<!/stats)/classes(\?|$)', {'classes': CLASSES})
    stub.on('GET', r'/review/tabs(\?|$)', MATCH_TABS)
    stub.on('GET', r'/regions/vocabulary(\?|$)', VOCABULARY)
    stub.on('GET', r'/crops/[^/]+/(image|thumbnail)', (200, b'', 'image/jpeg'))
    stub.on(
        'GET',
        r'/review/regions(\?|$)',
        {'items': [_item('plate-a')], 'total': 1, 'page': 1, 'page_size': 30},
    )

    page.goto(f'{app_url}/p/default/review?tab={REGION_TAB_URL_ID}')
    select = page.locator('label:has-text("Text match") select')
    select.wait_for(timeout=ACTION_TIMEOUT_MS)
    assert select.input_value() == ''
    with page.expect_response(
        lambda r: '/review/regions' in r.url and 'region_text_match=fuzzy' in r.url,
        timeout=ACTION_TIMEOUT_MS,
    ):
        select.select_option('fuzzy')
    save_screens(page, 'wf2b', 'review-text-match')
    with page.expect_response(
        lambda r: '/review/regions' in r.url and 'region_text_match' not in r.url,
        timeout=ACTION_TIMEOUT_MS,
    ):
        select.select_option('')
