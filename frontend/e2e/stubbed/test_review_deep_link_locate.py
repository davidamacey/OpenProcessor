"""DQ-M7: `/workspace?view=review&focus=item:<id>`
deep link at a deep page used to page 1..loc.page sequentially via
`queue.loadMore()` — 103 requests / 5.7s to reach rank 3000 (page 101) —
and for that whole window rendered page 1's item #1 with its keybindings
live, so a keypress acted on the wrong crop.

Fixed by `queue.loadAt(loc.cursor_before, loc.index)` (one request, src/lib/pager.svelte.ts)
and a new `awaitingDeepLink` state that keeps the tab-action keybindings
unregistered and renders a "Locating crop…" placeholder instead of item #1
until the target page lands (src/lib/workspace/views/ReviewView.svelte).

This test proves both halves against a real page mount:
  1. only ONE `/review/to_label` page request is made before the located page's
     request (page 1's own load, made before the deep link — never every
     page from 2..101);
  2. the page shows "Locating crop…" (not item #1) while the deep link
     resolves, and a keypress fired in that window never labels crop-0
     (item #1) — the regression this test reproduces.

Note on technique: point 2's keypress is fired while the locate response
is deliberately held open (a `threading.Event` a background thread
releases after a short real delay). This harness's Playwright sync API
serializes route-handler dispatch with other `page.*` calls made from the
test body, so a `page.*` call issued right after firing the keypress
can't reliably "observe" intermediate DOM state without itself blocking
on the held-open handler — instead of asserting on timing-fragile DOM
reads, the invariant checked is on the plain Python `label_calls` list
(no `page.*` call needed to inspect it) and is timing-independent: no
matter whether the keypress is delivered to the browser before or after
the locate resolves, it must never label crop-0 — either the
keybindings aren't registered yet (awaitingDeepLink still true) and
nothing happens, or the target crop (crop-target) is already current by
the time the key lands.
"""

from __future__ import annotations

import threading
from urllib.parse import parse_qs, urlparse

from fixtures.contract import crop_label_written
from fixtures.wire import make_item
from harness.constants import ACTION_TIMEOUT_MS, PLACEHOLDER_PHOTO, review_url
from harness.waits import wait_for_paint
from playwright.sync_api import expect


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

METHODS = {'strategies': [], 'flags': {}}
LOCATE_CURSOR = 'cursor-before-target'
AFTER_LOCATED = 'cursor-after-located-page'
AFTER_PREFETCH = 'cursor-after-prefetch'


def wait_until_requested(page, requests: list, cursor: str) -> None:
    for _ in range(40):
        if cursor in requests:
            return
        page.wait_for_timeout(250)
    raise AssertionError(f'{cursor} was never requested: {requests}')


def review_item(i: int) -> dict:
    item = make_item(
        crop_id=f'crop-{i}',
        image_id=f'img-{i}',
        class_id=1,
        class_name='ducati',
        proposed_class_id=1,
        proposed_class_name='ducati',
        label_validated=False,
        class_validated=False,
        label_source='model_suggestion',
        thumbnail_url=f'/curation/crops/crop-{i}/thumbnail',
    )
    item['reason'] = 'uncertainty'
    return item


def test_deep_link_fetches_only_the_located_page_and_ignores_early_keys(stub, page, app_url):
    review_page_requests: list[str | None] = []
    label_calls: list[str] = []
    locate_gate = threading.Event()

    stub.on('GET', r'(?<!/stats)/classes(\?|$)', {'classes': CLASSES})
    stub.on('GET', r'/methods(\?|$)', METHODS)
    for route in (r'/images/[^/]+/file(\?|$)', r'/crops/[^/]+/thumbnail(\?|$)'):
        stub.on('GET', route, lambda _req, _m: (200, PLACEHOLDER_PHOTO, 'image/svg+xml'))

    def review_handler(request, _match):
        qparams = parse_qs(urlparse(request.url).query)
        cursor = qparams.get('cursor', [None])[0]
        review_page_requests.append(cursor)
        # The first page serves item #1 (crop-0) so the "shows item #1 while
        # locating" regression is provable. The page the locate cursor starts
        # serves the actual target, crop-target.
        if cursor == LOCATE_CURSOR:
            item = review_item(999)
            # mapRawCrop() maps Crop.id from the wire's `crop_id` field,
            # not `id` -- both are set so the frontend actually resolves
            # this row's id to "crop-target" (matching the locate
            # response's crop_id and the URL's ?crop_id=).
            item['id'] = 'crop-target'
            item['crop_id'] = 'crop-target'
            return (
                200,
                {'items': [item], 'total': 3010, 'next_cursor': AFTER_LOCATED, 'limit': 30},
            )
        if cursor == AFTER_LOCATED:
            # the rows after the located one: 3002 onward
            rows = [review_item(1000 + i) for i in range(6)]
            return (
                200,
                {'items': rows, 'total': 3010, 'next_cursor': AFTER_PREFETCH, 'limit': 30},
            )
        return (200, {'items': [review_item(0)], 'total': 3010, 'next_cursor': None, 'limit': 30})

    stub.on('GET', r'/review/to_label(\?|$)', review_handler)

    def locate_handler(_request, _match):
        # Held open until the test explicitly releases it, simulating the
        # locate round trip's real latency — this is the window during
        # which the regression showed item #1 with live keys.
        locate_gate.wait(timeout=10)
        return (
            200,
            {
                'crop_id': 'crop-target',
                'in_queue': True,
                'index': 3000,
                'total': 3010,
                'cursor_before': LOCATE_CURSOR,
                'cursor_after': 'after-target',
                'reason': None,
                'sort_applied': 'atypicality',
                'sort_fallback_reason': None,
            },
        )

    stub.on('GET', r'/review/to_label/locate(\?|$)', locate_handler)

    def label_handler(request, match):
        label_calls.append(match.string)
        class_id = (request.post_data_json or {}).get('class_id')
        return (200, crop_label_written('crop-0', class_id, op_id='op-crop-0'))

    stub.on('PUT', r'/crops/([^/]+)/label$', label_handler)

    page.goto(review_url(app_url, 'to_label', 'crop-target'))

    # This is the one page.* call proven to complete BEFORE a held-open
    # route handler can block the driver's dispatch loop for later
    # page.* calls (it polls the already-rendered DOM, same as the
    # locate request itself firing only after this text is on screen).
    page.wait_for_selector('text=Locating crop', timeout=ACTION_TIMEOUT_MS)

    # Fire the keypress while the locate response is still held open.
    # Whether the browser processes it now or only once the driver loop
    # frees up after the gate is released below, it must never reach
    # crop-0 — see the module docstring for why this check doesn't rely
    # on further page.* calls to prove it.
    page.keyboard.press('Enter')

    locate_gate.set()
    page.wait_for_selector('text=Locating crop', state='detached', timeout=ACTION_TIMEOUT_MS)
    # There should be no in-flight PUT to wait FOR — this settles the
    # event loop via a real paint tick (same rationale as the module
    # docstring: a fixed sleep here would be exactly the kind of
    # timing-fragile check this test deliberately avoids elsewhere).
    wait_for_paint(page)

    assert not any('crop-0' in c for c in label_calls), (
        f'a keypress fired while locating must never label crop-0 (item #1): {label_calls}'
    )

    # The initial first-page load, the one request the locate cursor starts (never a walk
    # through the pages in between), then the queue's own prefetch with the served next_cursor.
    wait_until_requested(page, review_page_requests, AFTER_LOCATED)
    assert review_page_requests == [None, LOCATE_CURSOR, AFTER_LOCATED], review_page_requests

    # Rows remain after the located one: the filmstrip shows them after the located item.
    expect(page.get_by_test_id('filmstrip-item').nth(1)).to_be_visible(timeout=ACTION_TIMEOUT_MS)

    # F8 D6: the counter shows the served queue position (index 3000 ->
    # #3001), not the cursor's index within the one loaded page.
    counter = page.get_by_test_id('queue-counter').first.inner_text()
    assert counter.startswith('#3,001 '), counter

    errors = [c for c in stub.console_errors if c.startswith('pageerror')]
    assert not errors, f'no pageerror expected: {errors[:3]}'
