"""Import leftovers: the URL-seeded `combine_conflict` queue filter, lock badges, and image
Reprocess from a browse card. Payloads follow the vendored contract in the neutral widget domain;
the fail-closed stub means any request the page makes that a test did not expect fails the test.

  1. `test_combine_conflict_filter_from_the_url_is_sent` -- a served queue-only filter in the
     address is shown and sent; a queue that does not serve it never sends it.
  2. `test_lock_badge_only_on_a_locked_label`.
  3. `test_image_reprocess_from_a_cluster_card` -- body asserted, served items
     adopted into the grid.
  4. `test_import_provenance_rows_in_details`.
"""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import parse_qs, urlparse

from fixtures.contract import item_cluster
from fixtures.wire import make_item, review_tab, review_tabs
from harness.constants import ACTION_TIMEOUT_MS, review_url
from harness.waits import expect_handled
from playwright.sync_api import expect
from test_dataset_import import CLUSTER_UID, FORMATS, serve_cluster


CLASSES = [
    {
        'class_id': 2,
        'class_name': 'widget',
        'kind': 'item',
        'group': None,
        'hotkey_letter': None,
        'sample_count': 10,
        'validated_count': 5,
        'deprecated': False,
    },
    {
        'class_id': 5,
        'class_name': 'gadget',
        'kind': 'item',
        'group': None,
        'hotkey_letter': None,
        'sample_count': 4,
        'validated_count': 1,
        'deprecated': False,
    },
]

EMPTY_STATE = {
    'has_probe_predictions': True,
    'has_item_scores': True,
    'has_imported_labels': True,
}

IMPORT_ID = 'imp_20260927T120000_1a2b3c4d'


COMBINE_CONFLICT_SPEC = {
    'param': 'combine_conflict',
    'kind': 'bool',
    'label': 'Project-combine conflicts only',
    'options': [],
    'min': None,
    'max': None,
    'description': '',
    'default': None,
    'allows_unset': True,
}


def served_queues(**check_machine_over: Any) -> dict[str, Any]:
    return review_tabs(
        review_tab('to_label', 'To label'),
        review_tab('check_machine', 'Check machine labels', **check_machine_over),
        empty_state=EMPTY_STATE,
    )


def review_item(i: int, **over: Any) -> dict[str, Any]:
    item = make_item(
        crop_id=f'crop-{i}',
        image_id=f'img-{i}',
        class_id=2,
        class_name='widget',
        label_validated=True,
        label_source='import',
        **over,
    )
    item['reason'] = 'imported'
    return item


def serve_review(stub: Any, tabs: dict[str, Any]) -> list[dict[str, list[str]]]:
    """Registers the shell reads and a `/review/{tab}` handler that records
    each request's path and query."""
    seen: list[dict[str, list[str]]] = []
    stub.on('GET', r'(?<!/stats)/classes(\?|$)', {'classes': CLASSES})
    stub.on('GET', r'/review/tabs(\?|$)', tabs)

    def review_handler(request: Any, _m: Any):
        u = urlparse(request.url)
        seen.append({'path': [u.path], **parse_qs(u.query)})
        items = [review_item(i) for i in range(2)]
        return (200, {'items': items, 'total': 2, 'next_cursor': None, 'limit': 30})

    stub.on('GET', r'/review/(?!tabs)', review_handler)
    return seen


def review_requests(seen: list[dict[str, list[str]]], tab: str) -> list[dict[str, list[str]]]:
    return [s for s in seen if s['path'][0].endswith(f'/review/{tab}')]


def test_combine_conflict_filter_from_the_url_is_sent(stub, page, app_url):
    seen = serve_review(stub, served_queues(filter_specs=[COMBINE_CONFLICT_SPEC]))
    page.goto(review_url(app_url, 'check_machine', combine_conflict='true'))
    page.get_by_test_id('queue-counter').first.wait_for(timeout=ACTION_TIMEOUT_MS)
    expect(page.get_by_test_id('queue-filters')).to_contain_text(
        'Project-combine conflicts only', timeout=ACTION_TIMEOUT_MS
    )
    page.wait_for_load_state('networkidle', timeout=ACTION_TIMEOUT_MS)
    reqs = review_requests(seen, 'check_machine')
    assert reqs and reqs[0]['combine_conflict'] == ['true'], reqs


def test_queue_filter_the_queue_does_not_serve_is_not_sent(stub, page, app_url):
    seen = serve_review(stub, served_queues())
    page.goto(review_url(app_url, 'check_machine', combine_conflict='true'))
    page.get_by_test_id('queue-counter').first.wait_for(timeout=ACTION_TIMEOUT_MS)
    page.wait_for_load_state('networkidle', timeout=ACTION_TIMEOUT_MS)
    reqs = review_requests(seen, 'check_machine')
    assert reqs, seen
    assert all('combine_conflict' not in r for r in reqs), reqs
    expect(page.get_by_test_id('queue-filters')).to_have_count(0)


def serve_locked_cluster(stub: Any, locked_ids: set[str]) -> None:
    serve_cluster(stub)
    stub.on('GET', r'(?<!/stats)/classes(\?|$)', {'classes': CLASSES})
    crops = [
        make_item(
            crop_id=f'crop-{i}',
            image_id=f'img-{i}',
            class_id=2,
            class_name='widget',
            cluster=item_cluster(CLUSTER_UID),
            label_validated=False,
            label_source='model_suggestion',
            label_locked=f'crop-{i}' in locked_ids,
            thumbnail_url=f'/curation/crops/crop-{i}/thumbnail',
        )
        for i in range(3)
    ]
    stub.on('GET', r'/crops(\?|$)', {'total': 3, 'next_cursor': None, 'limit': 60, 'crops': crops})


def test_lock_badge_only_on_a_locked_label(stub, page, app_url):
    serve_locked_cluster(stub, {'crop-1'})
    page.goto(f'{app_url}/p/default/workspace?view=clusters&scope=cluster:{CLUSTER_UID}')
    page.wait_for_selector('img', timeout=ACTION_TIMEOUT_MS)
    badges = page.get_by_test_id('label-locked-badge')
    expect(badges).to_have_count(1, timeout=ACTION_TIMEOUT_MS)
    # The tooltip is the served lock rule (the reprocess vocabulary), not a
    # guessed single reason.
    expect(badges.first).to_have_attribute(
        'title',
        re.compile(r'^Locked\. Locked when: Human label: .*Test holdout: '),
        timeout=ACTION_TIMEOUT_MS,
    )


def test_image_reprocess_from_a_cluster_card(stub, page, app_url):
    serve_locked_cluster(stub, set())
    stub.on('GET', r'/datasets/formats(\?|$)', FORMATS)
    bodies: list[tuple[str, Any]] = []

    def reprocess(request: Any, _m: Any):
        bodies.append((urlparse(request.url).path, request.post_data_json))
        updated = make_item(
            crop_id='crop-0',
            image_id='img-0',
            class_id=5,
            class_name='gadget',
            cluster=item_cluster(CLUSTER_UID),
            label_validated=False,
            label_source='model_suggestion',
            thumbnail_url='/curation/crops/crop-0/thumbnail',
        )
        return (
            200,
            {
                'dry_run': False,
                'scopes': [
                    {
                        'scope': 'detect',
                        'selected': 1,
                        'locked_skipped': 0,
                        'queued': 1,
                        'breakdown': [],
                    }
                ],
                'job': None,
                'items': [updated],
            },
        )

    stub.on('POST', r'/images/img-0/reprocess$', reprocess)

    page.goto(f'{app_url}/p/default/workspace?view=clusters&scope=cluster:{CLUSTER_UID}')
    page.wait_for_selector('img', timeout=ACTION_TIMEOUT_MS)
    names = page.get_by_test_id('class-name')
    expect(names).to_have_text(['widget', 'widget', 'widget'], timeout=ACTION_TIMEOUT_MS)

    card = page.locator('div.group').filter(has=page.locator('button[aria-pressed]')).first
    card.hover()
    card.get_by_role('button', name='Expand').click()
    page.get_by_test_id('card-reprocess-image').get_by_test_id('reprocess-open').click()
    dialog = page.get_by_role('dialog', name='Reprocess')
    expect(dialog.get_by_role('heading', name='Reprocess image')).to_be_visible()
    dialog.get_by_label('Find objects', exact=True).check()
    with expect_handled(
        page, lambda r: r.method == 'POST' and r.url.endswith('/images/img-0/reprocess')
    ):
        dialog.get_by_role('button', name='Reprocess', exact=True).click()
    assert [b for _p, b in bodies] == [{'scopes': ['detect'], 'dry_run': False}], bodies
    assert bodies[0][0].endswith('/images/img-0/reprocess'), bodies
    expect(dialog.get_by_test_id('reprocess-result')).to_be_visible(timeout=ACTION_TIMEOUT_MS)

    # The served item replaced the grid's copy of that crop only.
    dialog.get_by_role('button', name='Close').click()
    page.get_by_role('button', name='Close', exact=True).click()
    expect(names).to_have_text(['gadget', 'widget', 'widget'], timeout=ACTION_TIMEOUT_MS)


def test_import_provenance_rows_in_details(stub, page, app_url):
    serve_locked_cluster(stub, set())
    stub.on('GET', r'/datasets/formats(\?|$)', FORMATS)
    stub.on('GET', r'/crops/[^/]+/history$', {'crop_id': 'crop-0', 'entries': []})
    imported = make_item(
        crop_id='crop-0',
        image_id='img-0',
        class_id=2,
        class_name='widget',
        cluster=item_cluster(CLUSTER_UID),
        label_validated=True,
        label_source='import',
        label_locked=True,
        import_ids=[IMPORT_ID],
        dataset_split='train',
        imported_at='2026-09-27T12:00:03Z',
        on_negative_frame=True,
        proposal_chain=['import:' + IMPORT_ID],
        thumbnail_url='/curation/crops/crop-0/thumbnail',
    )
    others = [
        make_item(
            crop_id=f'crop-{i}',
            image_id=f'img-{i}',
            class_id=2,
            class_name='widget',
            cluster=item_cluster(CLUSTER_UID),
            thumbnail_url=f'/curation/crops/crop-{i}/thumbnail',
        )
        for i in (1, 2)
    ]
    stub.on(
        'GET',
        r'/crops(\?|$)',
        {'total': 3, 'next_cursor': None, 'limit': 60, 'crops': [imported, *others]},
    )

    page.goto(f'{app_url}/p/default/workspace?view=clusters&scope=cluster:{CLUSTER_UID}')
    page.wait_for_selector('img', timeout=ACTION_TIMEOUT_MS)
    card = page.locator('div.group').filter(has=page.locator('button[aria-pressed]')).first
    card.hover()
    card.get_by_role('button', name='Show crop details').click()
    expect(page.get_by_test_id('import-label-locked')).to_have_text(
        'Locked', timeout=ACTION_TIMEOUT_MS
    )
    expect(page.get_by_test_id('import-split')).to_have_text('train')
    link = page.get_by_test_id('import-ids').get_by_role('link', name=IMPORT_ID)
    expect(link).to_have_attribute('href', f'/p/default/datasets/imports/{IMPORT_ID}')
    expect(page.get_by_test_id('import-negative-frame')).to_be_visible()
    expect(page.get_by_test_id('import-proposal-chain')).to_contain_text('import:' + IMPORT_ID)
