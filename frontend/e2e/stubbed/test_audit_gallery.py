"""Data quality › Accuracy audit galleries: the waiting sample as a strip of crops, and every
outcome count and off-diagonal confusion cell as a link to the matching items in Detections."""

from __future__ import annotations

from fixtures.wire import make_item
from harness.constants import ACTION_TIMEOUT_MS
from harness.screens import save_screens
from playwright.sync_api import expect
from test_app_shell import DESKTOP
from test_label_confirmation import REPORT, _stub_audit


def _crops(n: int) -> dict:
    items = [
        make_item(
            crop_id=f'crop-{i}',
            image_id=f'img-{i}',
            class_id=2,
            class_name='widget_a',
            label_validated=False,
            thumbnail_url=f'/curation/crops/crop-{i}/thumbnail',
        )
        for i in range(n)
    ]
    return {'total': n, 'next_cursor': None, 'limit': 24, 'crops': items}


def _serve(stub) -> list[str]:
    _stub_audit(stub)
    seen: list[str] = []

    def crops(request, _match):
        seen.append(request.url)
        return (200, _crops(24))

    stub.on('GET', r'/crops(\?|$)', crops)
    return seen


def test_the_waiting_sample_is_a_strip_of_the_audit_queue_crops(stub, page, app_url):
    seen = _serve(stub)
    page.set_viewport_size(DESKTOP)
    page.goto(f'{app_url}/p/default/quality/audit')
    summary = page.get_by_test_id('audit-pending-summary')
    expect(summary).to_have_text('83 drawn · 52 labelled · 31 waiting', timeout=ACTION_TIMEOUT_MS)
    expect(page.get_by_test_id('audit-pending-strip').locator('li')).to_have_count(24)
    assert any('review_tab=audit' in u for u in seen), seen
    expect(page.get_by_test_id('audit-open-queue')).to_have_attribute(
        'href', '/p/default/workspace?view=review&queue=audit'
    )
    expect(page.get_by_test_id('audit-see-all')).to_have_attribute(
        'href', '/p/default/workspace?view=detections&queue=audit'
    )
    save_screens(page, 'uxr-d', 'audit-pending')


def test_outcome_counts_and_confusion_cells_link_to_detections(stub, page, app_url):
    _serve(stub)
    page.goto(f'{app_url}/p/default/quality/audit')
    outcomes = page.get_by_test_id('audit-outcomes')
    expect(outcomes.locator('a')).to_have_count(len(REPORT['outcomes']), timeout=ACTION_TIMEOUT_MS)
    href = outcomes.locator('a[data-outcome="detector_wrong"]').get_attribute('href')
    assert href == '/p/default/workspace?view=detections&audit_outcome=detector_wrong', href
    cell = page.locator(
        '[data-testid="confusion-cell"][data-detector="widget_a"][data-human="widget_b"] a'
    )
    href = cell.get_attribute('href') or ''
    for part in (
        'view=detections',
        'audit_outcome=detector_wrong',
        'audit_outcome=both_wrong',
        'detector_class_name=widget_a',
        'class_name=widget_b',
    ):
        assert part in href, href
    diagonal = page.locator(
        '[data-testid="confusion-cell"][data-detector="widget_a"][data-human="widget_a"] a'
    )
    expect(diagonal).to_have_count(0)
