"""New coverage (test audit recommendation 5):
`/review` Enter-assign followed by Z-undo.

Enter -> confirmAndAdvance() -> assign(classId) -> `PUT
{API_PREFIX}/crops/{id}/label` (src/routes/review/+page.svelte).
The write is recorded as a label operation (the PUT answers its `op_id`), so Z ->
undoLast() -> `POST {API_PREFIX}/label_ops/{op_id}/undo`, the same served undo the Label
history uses; the per-crop `.../label/undo` route is not used.
"""

from __future__ import annotations

from fixtures.wire import make_item
from harness.constants import ACTION_TIMEOUT_MS, review_url
from harness.label_ops import stub_crop_label, stub_label_op_undo
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


def test_review_assign_then_undo(stub, page, app_url):
    label_calls: list[tuple[str, dict]] = []
    undo_calls: list[dict] = []

    stub.on('GET', r'(?<!/stats)/classes(\?|$)', {'classes': CLASSES})
    stub.on('GET', r'/methods(\?|$)', METHODS)
    # The detail lightbox/hover path fetches a crop's full image + siblings.
    stub.on(
        'GET',
        r'/images/[^/]+/file$',
        lambda req, m: (
            200,
            {
                'image': {'path': '/nas/img-0.jpg', 'width': 640, 'height': 480},
                'items': [review_item(0)],
            },
        ),
    )

    def review_handler(_request, _match):
        items = [review_item(i) for i in range(3)]
        return (200, {'items': items, 'total': 3, 'next_cursor': None, 'limit': 30})

    stub.on('GET', r'/review/(?!tabs)', review_handler)

    stub_crop_label(stub, calls=label_calls)
    stub_label_op_undo(stub, 'op-crop-0', calls=undo_calls)
    # The restored item is read back after the undo.
    stub.on('GET', r'/crops(\?|$)', {'crops': [review_item(0)], 'total': 1, 'next_cursor': None})
    legacy: list[str] = []
    stub.on(
        'POST',
        r'/crops/[^/]+/label/undo$',
        lambda request, match: legacy.append(match.string) or (200, {}),
    )

    page.goto(review_url(app_url, 'to_label'))
    counter = page.get_by_test_id('queue-counter')
    counter.first.wait_for(timeout=ACTION_TIMEOUT_MS)
    before = counter.first.inner_text()

    with page.expect_response(
        lambda r: r.request.method == 'PUT' and r.url.endswith('/label'), timeout=ACTION_TIMEOUT_MS
    ):
        page.keyboard.press('Enter')

    assert len(label_calls) == 1, f'Enter should PUT exactly one label: {label_calls}'
    crop_id, body = label_calls[0]
    assert crop_id == 'crop-0'
    assert body.get('class_id') == 1, body

    expect(counter.first).not_to_have_text(before, timeout=ACTION_TIMEOUT_MS)
    after_label = counter.first.inner_text()
    assert after_label != before, 'the queue should visibly advance after a successful assign'

    with page.expect_response(
        lambda r: r.request.method == 'POST' and r.url.endswith('/label_ops/op-crop-0/undo'),
        timeout=ACTION_TIMEOUT_MS,
    ):
        page.keyboard.press('z')

    assert undo_calls == [{'dry_run': False}], f'Z should queue one undo by op id: {undo_calls}'
    assert legacy == [], f'the single write is undone by its op, not per crop: {legacy}'

    errors = [c for c in stub.console_errors if c.startswith('pageerror')]
    assert not errors, f'no pageerror expected in the assign/undo flow: {errors[:3]}'
